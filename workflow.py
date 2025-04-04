import logging
from typing import Annotated, Sequence
from aiogram import Bot
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel, Field

from config import settings
from open_search_retriever import OpenSearchRetriever
from opensearch_retrieval_grader import OpenSearchRetrievalGrader
from question_rewriter import QuestionRewriter
from answerer import Answerer

logger = logging.getLogger(__name__)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class ChatState(MessagesState):
    """State for the chat workflow."""
    telegram_chat_id: int
    question: str
    rewrite_question_attempts: int
    documents: Annotated[Sequence[Document], "documents"]

class ChatChain:
    """Class to handle chat processing logic."""
    def __init__(self, bot: Bot):
        self.bot = bot
        self.opensearch_retriever = OpenSearchRetriever()
        self.retrieval_grader = OpenSearchRetrievalGrader()
        self.question_rewriter = QuestionRewriter()
        self.answerer = Answerer()
        
    async def retrieve_opensearch_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents for the query."""
        try:
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Retrieving documents..."
            )

            # Get the last user message
            question = state["question"]
            # print(f"================ CHAIN QUERY: {question}")
            
            # Retrieve documents
            docs = self.opensearch_retriever.invoke(question)
            await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Retrieved {len(docs)} documents"
                        )
            
            return {**state, "documents": docs}
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            raise
    
    async def grade_opensearch_documents(self, state):
        """Determines whether the retrieved documents are relevant to the question."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Checking document relevance to question..."
            )
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for doc in documents:
            print("--------------------------------")
            print(f"---DOC: {doc.page_content} {doc.metadata}---")
            grade = self.retrieval_grader.invoke(
                question=question,
                document=f"{doc.page_content} \n\n {doc.metadata}"
            )

            print(f"---GRADE: {grade}---")

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        
        print(f"---FILTERED {len(filtered_docs)} documents")
        await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"{len(filtered_docs)} documents graded as relevant"
                        )
        
        return {**state, "documents": filtered_docs}

    
    async def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Assessing graded documents..."
        )
        filtered_documents = state["documents"]
        rewrite_question_attempts = state['rewrite_question_attempts']

        if not filtered_documents and rewrite_question_attempts > 1:
            print(f"---RETRIEVE ATTEMPTS: {rewrite_question_attempts}---")
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"All documents are not relevant to question, transforming query..."
                        )
            return "rewrite_question"
        else:
            print("---DECISION: GENERATE---")
            await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Decision: Generate answer"
                        )
            return "answer_question"
    
    
    async def rewrite_question(self, state):
        print("---REWRITE QUESTION---")
        msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Decision: Rewrite question"
        )

        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        print(f"---NEW QUESTION: {better_question}---")
        await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Rewritten question: {better_question}"
                        )
        return {"documents": documents, "question": better_question, "rewrite_question_attempts": state["rewrite_question_attempts"] - 1}

    async def answer_question(self, state: ChatState) -> ChatState:
        """Process a message and update the state."""
        try:
            # Get the user message from history
            question = state["question"]

            # Send initial "processing" message
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text=f"Answering your question: {question}"
            )

            current_sentence = ""
            full_response = ""
            
            # Stream the response
            formatted_docs = "\n\n".join(doc.page_content for doc in state["documents"])
            
            async for chunk in self.answerer.astream({"context": formatted_docs, "question": question}):
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
        self.workflow.add_node("retrieve_opensearch_documents", self.chat_chain.retrieve_opensearch_documents)
        self.workflow.add_node("grade_opensearch_documents", self.chat_chain.grade_opensearch_documents)  
        self.workflow.add_node("rewrite_question", self.chat_chain.rewrite_question)
        self.workflow.add_node("answer_question", self.chat_chain.answer_question)
        
        # Add edges
        self.workflow.add_edge(START, "retrieve_opensearch_documents")
        self.workflow.add_edge("retrieve_opensearch_documents", "grade_opensearch_documents")


        self.workflow.add_conditional_edges(
            "grade_opensearch_documents",
            self.chat_chain.decide_to_generate,
            {
                "rewrite_question": "rewrite_question",
                "answer_question": "answer_question",
            },
        )
        self.workflow.add_edge("rewrite_question", "retrieve_opensearch_documents")


        # self.workflow.add_edge("grade_opensearch_documents", "answer_question")
        self.workflow.add_edge("answer_question", END)
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    async def process(self, initial_state: ChatState):
        """Process a message through the workflow."""
        return await self.app.ainvoke(initial_state) 
    