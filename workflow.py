import logging
from typing import Annotated, Sequence
from aiogram import Bot
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel, Field
import asyncio

from config import settings
from open_search_retriever import OpenSearchRetriever
from opensearch_retrieval_grader import OpenSearchRetrievalGrader
from question_rewriter import QuestionRewriter
from answerer import Answerer
from hallucination_grader import HallucinationGrader
from answer_grader import AnswerGrader

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
        self.hallucination_grader = HallucinationGrader()
        self.answer_grader = AnswerGrader()
        
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
            docs = await self.opensearch_retriever.ainvoke(question)
            await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Retrieved {len(docs)} documents"
                        )
            
            return {**state, "documents": docs}
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            raise
    
    async def _grade_single_document(self, question: str, doc: Document) -> tuple[Document, bool]:
        """Grade a single document for relevance.
        
        Args:
            question: The user's question
            doc: The document to grade
            
        Returns:
            Tuple of (document, is_relevant)
        """
        try:
            grade = await self.retrieval_grader.ainvoke(
                question=question,
                document=f"{doc.page_content} \n\n {doc.metadata}"
            )
            print(f"---DOC: {doc.page_content[:100]}... GRADE: {grade}---")
            return (doc, grade == "yes")
        except Exception as e:
            print(f"Error grading document: {e}")
            return (doc, False)

    async def grade_opensearch_documents(self, state):
        """Determines whether the retrieved documents are relevant to the question."""
        try:
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            
            status_msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Checking document relevance to question..."
            )

            # Grade all documents concurrently
            graded_docs = await asyncio.gather(
                *[self._grade_single_document(state["question"], doc) 
                  for doc in state["documents"]]
            )

            # Filter relevant documents
            filtered_docs = [doc for doc, is_relevant in graded_docs if is_relevant]
            
            # Update status message
            await self.bot.edit_message_text(
                chat_id=state["telegram_chat_id"],
                message_id=status_msg.message_id,
                text=f"{len(filtered_docs)} out of {len(state['documents'])} documents graded as relevant"
            )
            
            return {**state, "documents": filtered_docs}

        except Exception as e:
            logger.error(f"Error in grade_opensearch_documents: {e}")
            # Try to notify user about the error
            try:
                await self.bot.edit_message_text(
                    chat_id=state["telegram_chat_id"],
                    message_id=status_msg.message_id,
                    text="Error occurred while grading documents"
                )
            except:
                pass
            raise
    
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
            return "generate_answer"
    
    
    async def rewrite_question(self, state):
        print("---REWRITE QUESTION---")
        msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Decision: Rewrite question"
        )

        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = await self.question_rewriter.ainvoke({"question": question})
        print(f"---NEW QUESTION: {better_question}---")
        await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Rewritten question: {better_question}"
                        )
        return {"documents": documents, "question": better_question, "rewrite_question_attempts": state["rewrite_question_attempts"] - 1}

    async def generate_answer(self, state: ChatState) -> ChatState:
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
            # state["messages"].append(AIMessage(content=full_response))
            
            return {**state, "generation": full_response}
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}")
            raise

    async def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        grade = await self.hallucination_grader.ainvoke(documents=documents, generation=generation)
        print(f"---HALLUCINATION GRADE: {grade}---")

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            grade = await self.answer_grader.ainvoke(question=question, generation=generation)
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"


class WorkflowGraph:
    """Class to manage the workflow graph."""
    def __init__(self, bot: Bot):
        self.chat_chain = ChatChain(bot)
        self.workflow = StateGraph(ChatState)
        
        # Add nodes
        self.workflow.add_node("retrieve_opensearch_documents", self.chat_chain.retrieve_opensearch_documents)
        self.workflow.add_node("grade_opensearch_documents", self.chat_chain.grade_opensearch_documents)  
        self.workflow.add_node("rewrite_question", self.chat_chain.rewrite_question)
        self.workflow.add_node("generate_answer", self.chat_chain.generate_answer)
        
        # Add edges
        self.workflow.add_edge(START, "retrieve_opensearch_documents")
        self.workflow.add_edge("retrieve_opensearch_documents", "grade_opensearch_documents")


        self.workflow.add_conditional_edges(
            "grade_opensearch_documents",
            self.chat_chain.decide_to_generate,
            {
                "rewrite_question": "rewrite_question",
                "generate_answer": "generate_answer",
            },
        )
        self.workflow.add_edge("rewrite_question", "retrieve_opensearch_documents")


        self.workflow.add_conditional_edges(
            "generate_answer",
            self.chat_chain.grade_generation_v_documents_and_question,
            {
                "not supported": "generate_answer",
                "useful": END,
                "not useful": "rewrite_question",
            },
        )

        # Compile the graph
        self.app = self.workflow.compile()
    
    async def process(self, initial_state: ChatState):
        """Process a message through the workflow."""
        return await self.app.ainvoke(initial_state) 
    