import logging
from typing import Annotated, Sequence
from aiogram import Bot
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from config import settings
from retriever import OpenSearchRetriever

logger = logging.getLogger(__name__)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class ChatState(MessagesState):
    """State for the chat workflow."""
    telegram_chat_id: int
    messages: Annotated[Sequence[HumanMessage | AIMessage], "messages"]
    documents: Annotated[Sequence[Document], "documents"]

class ChatChain:
    """Class to handle chat processing logic."""
    def __init__(self, bot: Bot):
        self.bot = bot
        self.opensearch_retriever = OpenSearchRetriever()
        
        # Retrieval variables   
        _retrieval_grader_prompt = PromptTemplate(
            input_variables=["question", "document"],
            template="""
            You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Retrieved document: \n\n {document} \n\n User question: {question}
            """
        )

        _retrieval_grader_llm = ChatOllama(
            base_url=settings.retrieval_grader_ollama_base_url,
            model=settings.retrieval_grader_ollama_model,
            temperature=settings.retrieval_grader_ollama_temperature,
            timeout=settings.retrieval_grader_ollama_timeout,
            streaming=True,
            max_tokens=settings.retrieval_grader_ollama_max_tokens
        )

        _structured_retrieval_grader_llm = _retrieval_grader_llm.with_structured_output(GradeDocuments)

        self.retrieval_grader = _retrieval_grader_prompt | _structured_retrieval_grader_llm


        # Answer variables
        _answer_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Based on the user's query and the logs provided, determine the appropriate response.

            USER QUESTION: {question}

            LOGS CONTEXT:
            {context}

            Please provide:
            1. A direct answer to the user's question, if possible (e.g., confirmation of an event or status). If the logs contain information that directly answers the user's question, state it clearly.
            2. A concise description of what these logs represent, suitable for a business user.
            3. Technical context from the codebase, if applicable (relevant files, functions, or code paths).
            4. Exact IDs affected by the error, if applicable.

            Focus on providing a clear and direct response to the user's question, supplemented by technical insights when necessary.
            """
        )

        _answer_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            timeout=settings.ollama_timeout,
            streaming=True,
            max_tokens=settings.ollama_max_tokens
            # temperature=0.1,    # Low temperature for consistent pattern matching
            # top_k=3,           # Very limited options for precise matching
            # top_p=0.1,         # High precision in pattern identification
            # num_ctx=8192,      # Large context to analyze many logs at once
            # repeat_penalty=1.2  # Higher penalty to avoid repetitive patterns
        )

        self.answer_chain = _answer_prompt | _answer_llm | StrOutputParser()





        
        

    
    async def retrieve_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents for the query."""
        try:
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Retrieving documents..."
            )

            # Get the last user message
            question = state["messages"][0].content
            # print(f"================ CHAIN QUERY: {question}")
            
            # Retrieve documents
            docs = self.opensearch_retriever.invoke(question)
            await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Retrieved {len(docs)} documents"
                        )
            # Update state with retrieved documents
            state["documents"] = docs
            
            return state
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            raise
    

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["messages"][0].content
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for doc in documents:
            print("--------------------------------")
            print(f"---DOC: {doc.page_content} {doc.metadata}---")
            grade = self.retrieval_grader.invoke(
                {"question": question, "document": f"{doc.page_content} \n\n {doc.metadata}"}
            )
            grade = grade.binary_score

            print(f"---GRADE: {grade}---")

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    async def answer_question(self, state: ChatState) -> ChatState:
        """Process a message and update the state."""
        try:
            # Get the user message from history
            first_message = state["messages"][0].content

            # Send initial "processing" message
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text=f"Answering your question: {first_message}"
            )

            current_sentence = ""
            full_response = ""
            
            # Stream the response
            formatted_docs = "\n\n".join(doc.page_content for doc in state["documents"])
            async for chunk in self.answer_chain.astream({"context": formatted_docs, "question": first_message}):
                if chunk:
                    current_sentence += chunk
                    
                    # Check if we have a complete sentence
                    if any(current_sentence.strip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                        full_response += current_sentence
                        current_sentence = ""
                        # Update the message with the accumulated response
                        await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=full_response
                        )
            
            # Add any remaining text
            if current_sentence:
                full_response += current_sentence
            
            # Add the AI response to message history
            state["messages"].append(AIMessage(content=full_response))
            
            return state
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            raise

class WorkflowGraph:
    """Class to manage the workflow graph."""
    def __init__(self, bot: Bot):
        self.chat_chain = ChatChain(bot)
        self.workflow = StateGraph(ChatState)
        
        # Add nodes
        self.workflow.add_node("retrieve_documents", self.chat_chain.retrieve_documents)
        self.workflow.add_node("grade_documents", self.chat_chain.grade_documents)  
        self.workflow.add_node("answer_question", self.chat_chain.answer_question)
        
        # Add edges
        self.workflow.add_edge(START, "retrieve_documents")
        self.workflow.add_edge("retrieve_documents", "grade_documents")
        self.workflow.add_edge("grade_documents", "answer_question")
        self.workflow.add_edge("answer_question", END)
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    async def process(self, initial_state: ChatState):
        """Process a message through the workflow."""
        return await self.app.ainvoke(initial_state) 
    