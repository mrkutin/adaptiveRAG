import logging
from typing import Annotated, Sequence
from aiogram import Bot
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
import asyncio

from config import settings
from open_search_retriever import OpenSearchRetriever
from opensearch_retrieval_grader import OpenSearchRetrievalGrader
from question_rewriter import QuestionRewriter
from answerer import Answerer
from log_summarizer import LogSummarizer
from code_base_retriever import CodeBaseRetriever
# from hallucination_grader import HallucinationGrader
# from answer_grader import AnswerGrader


logger = logging.getLogger(__name__)


class ChatState(MessagesState):
    """State for the chat workflow."""
    telegram_chat_id: int
    question: str
    rewrite_question_attempts: int
    # regenerate_answer_attempts: int  
    documents: Annotated[Sequence[Document], "documents"]
    generation: str
    log_summary: str = ""  # Add log summary to state
    stack_traces_text: str = ""
    code_docs: Annotated[Sequence[Document], "code_docs"] = []

class ChatChain:
    """Class to handle chat processing logic."""
    def __init__(self, bot: Bot):
        self.bot = bot
        self.opensearch_retriever = OpenSearchRetriever()
        self.retrieval_grader = OpenSearchRetrievalGrader()
        self.question_rewriter = QuestionRewriter()
        self.answerer = Answerer()
        self.log_summarizer = LogSummarizer()
        self.code_base_retriever = CodeBaseRetriever()
        # self.hallucination_grader = HallucinationGrader()
        # self.answer_grader = AnswerGrader()
        
    async def retrieve_opensearch_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents for the query."""
        try:
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="🔍 Retrieving documents..."
            )

            # Get the last user message
            question = state["question"]
            
            # Retrieve documents
            docs = await self.opensearch_retriever.ainvoke(question)
            await self.bot.edit_message_text(
                chat_id=state["telegram_chat_id"],
                message_id=msg.message_id,
                text=f"📚 Retrieved {len(docs)} documents"
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
            print(f"---DOC: {doc.metadata['time']} {doc.page_content[:100]}... GRADE: {grade}---")
            return (doc, grade == "yes")
        except Exception as e:
            print(f"Error grading document: {e}")
            return (doc, False)

    async def grade_opensearch_documents(self, state):
        """Determines whether the retrieved documents are relevant to the question."""
        try:
            status_msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="🤔 Checking document relevance to question..."
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
                text=f"📊 {len(filtered_docs)} out of {len(state['documents'])} documents graded as relevant"
            )
            
            return {**state, "documents": filtered_docs}

        except Exception as e:
            logger.error(f"Error in grade_opensearch_documents: {e}")
            # Try to notify user about the error
            try:
                await self.bot.edit_message_text(
                    chat_id=state["telegram_chat_id"],
                    message_id=status_msg.message_id,
                    text="❌ Error occurred while grading documents"
                )
            except:
                pass
            raise
    
    async def decide_to_generate(self, state):
        msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="🔄 Assessing graded documents..."
        )
        filtered_documents = state["documents"]
        rewrite_question_attempts = state['rewrite_question_attempts']

        if not filtered_documents and rewrite_question_attempts > 1:
            await self.bot.edit_message_text(
                chat_id=state["telegram_chat_id"],
                message_id=msg.message_id,
                text=f"❌ All documents are not relevant to question, transforming query..."
            )
            return "rewrite_question"
        else:
            await self.bot.edit_message_text(
                chat_id=state["telegram_chat_id"],
                message_id=msg.message_id,
                text=f"✨ Decision: Generate answer"
            )
            return "generate_answer"
    
    
    async def rewrite_question(self, state):
        msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="✏️ Rewriting question..."
        )

        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = await self.question_rewriter.ainvoke({"question": question})
        print(f"---NEW QUESTION: {better_question}---")
        await self.bot.edit_message_text(
            chat_id=state["telegram_chat_id"],
            message_id=msg.message_id,
            text=f"✅ Rewritten question: {better_question}"
        )
        return {**state, "documents": documents, "question": better_question, "rewrite_question_attempts": state["rewrite_question_attempts"] - 1}

    async def retrieve_code_docs(self, state: ChatState) -> ChatState:
        """Retrieve relevant code documents for the stack traces."""
        try:
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="🔍 Searching codebase for relevant files..."
            )

            # Get stack traces from state
            stack_traces = state["stack_traces_text"]
            
            # Retrieve relevant code documents
            code_docs = await self.code_base_retriever.ainvoke(stack_traces)
            
            # Update status message
            await self.bot.edit_message_text(
                chat_id=state["telegram_chat_id"],
                message_id=msg.message_id,
                text=f"📚 Found {len(code_docs)} relevant code files"
            )
            
            return {**state, "code_docs": code_docs}
        except Exception as e:
            logger.error(f"Error in retrieve_code_docs: {str(e)}")
            raise

    async def generate_answer(self, state: ChatState) -> ChatState:
        """Process a message and update the state."""
        try:
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="🤖 Generating answer..."
            )

            # Format documents for context
            formatted_docs = "\n\n".join(f"Log Entry:\n{doc.page_content}" for doc in state["documents"])
            
            # Generate response using the documents directly
            response = await self.answerer.ainvoke({
                "context": formatted_docs,
                "question": state["question"]
            })

            # Update message to indicate completion
            await self.bot.edit_message_text(
                chat_id=state["telegram_chat_id"],
                message_id=msg.message_id,
                text="✅ Answer generated successfully"
            )
            
            return {**state, "generation": response}
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}")
            raise

    # async def summarize_logs(self, state: ChatState) -> ChatState:
    #     """Summarize the retrieved logs."""
    #     try:
    #         msg = await self.bot.send_message(
    #             chat_id=state["telegram_chat_id"],
    #             text="📊 Analyzing logs..."
    #         )

    #         # Extract log content from documents
    #         logs = [doc.page_content for doc in state["documents"]]
    #         i = 0
    #         for log in logs:
    #             print(f"---LOG {i}: {log[:100]}...---")
    #             i += 1
    #         # Get summary
    #         summary = await self.log_summarizer.ainvoke(logs)
    #         logger.info(f"---SUMMARY: {summary}---")
            

    #         # Format summary for display
    #         key_events_text = "\n".join(f"• {event}" for event in summary.key_events)
    #         stack_traces_text = "\n".join(f"```\n{trace}\n```" for trace in summary.stack_traces) if summary.stack_traces else "No stack traces found"
            
    #         summary_text = f"""
    #         📝 Log Analysis Summary:
    #         {summary.summary}

    #         🔍 Key Events:
    #         {key_events_text}

    #         ⚠️ Statistics:
    #         • Errors: {summary.error_count}
    #         • Warnings: {summary.warning_count}

    #         🔧 Stack Traces:
    #         {stack_traces_text}
    # """
            
    #         # Update status message
    #         await self.bot.edit_message_text(
    #             chat_id=state["telegram_chat_id"],
    #             message_id=msg.message_id,
    #             text=summary_text
    #         )
            
    #         return {**state, "log_summary": summary_text, "stack_traces_text": stack_traces_text}
    #     except Exception as e:
    #         logger.error(f"Error in summarize_logs: {str(e)}")
    #         raise


class WorkflowGraph:
    """Class to manage the workflow graph."""
    def __init__(self, bot: Bot):
        self.chat_chain = ChatChain(bot)
        self.workflow = StateGraph(ChatState)
        
        # Add nodes
        self.workflow.add_node("retrieve_opensearch_documents", self.chat_chain.retrieve_opensearch_documents)
        self.workflow.add_node("grade_opensearch_documents", self.chat_chain.grade_opensearch_documents)  
        # self.workflow.add_node("summarize_logs", self.chat_chain.summarize_logs)
        # self.workflow.add_node("retrieve_code_docs", self.chat_chain.retrieve_code_docs)
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
                "generate_answer": "generate_answer",  # Changed from "summarize_logs" to "generate_answer"
            },
        )
        # self.workflow.add_edge("summarize_logs", "retrieve_code_docs")
        # self.workflow.add_edge("retrieve_code_docs", "generate_answer")
        self.workflow.add_edge("rewrite_question", "retrieve_opensearch_documents")
        self.workflow.add_edge("generate_answer", END)

        # Compile the graph
        self.app = self.workflow.compile()
        self.app.get_graph().print_ascii()
        self.app.get_graph().draw_png(
                    output_file_path="workflow_graph.png",
                    fontname="Arial",
                )
    
    async def process(self, initial_state: ChatState):
        """Process a message through the workflow."""
        return await self.app.ainvoke(initial_state) 
    