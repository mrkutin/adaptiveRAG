import logging
from typing import Sequence, List
from aiogram import Bot
from langchain_core.documents import Document
import asyncio

from config import settings
from open_search_retriever import OpenSearchRetriever
from opensearch_retrieval_grader import OpenSearchRetrievalGrader
from answerer import Answerer
from code_base_retriever import CodeBaseRetriever

logger = logging.getLogger(__name__)

class ChatChain:
    """Class to handle chat processing logic."""
    def __init__(self, bot: Bot):
        self.bot = bot
        self.opensearch_retriever = OpenSearchRetriever()
        self.retrieval_grader = OpenSearchRetrievalGrader()
        self.answerer = Answerer()
        self.code_base_retriever = CodeBaseRetriever()

    async def process_message(self, telegram_chat_id: int, question: str) -> None:
        """Main processing function that handles the entire workflow."""
        # Step 1: Retrieve documents
        documents = await self.retrieve_opensearch_documents(telegram_chat_id, question)
        
        if not documents:
            await self.bot.send_message(
                chat_id=telegram_chat_id,
                text="No documents found for your question."
            )
            return
        
        # Step 2: Process each document
        relevant_docs_found = False
        for i, doc in enumerate(documents, 1):
            # Check relevance
            is_relevant = await self._grade_single_document(question, doc)
            
            if not is_relevant:
                await self.bot.send_message(
                    chat_id=telegram_chat_id,
                    text=f"📄 Document {i}/{len(documents)} is not relevant to your question, skipping..."
                )
                continue
            
            relevant_docs_found = True
            # Get code snippets if available
            code_docs = []
            if doc.metadata.get('stack_trace'):
                code_docs = await self.retrieve_code_docs(telegram_chat_id, doc.metadata['stack_trace'])
            
            # Generate and send answer for this document
            await self.generate_and_send_answer(telegram_chat_id, question, doc, code_docs, i, len(documents))
        
        if not relevant_docs_found:
            await self.bot.send_message(
                chat_id=telegram_chat_id,
                text="No relevant documents found to answer your question."
            )

    async def retrieve_opensearch_documents(self, telegram_chat_id: int, question: str) -> List[Document]:
        """Retrieve relevant documents for the query."""
        msg = await self.bot.send_message(
            chat_id=telegram_chat_id,
            text="🔍 Retrieving documents..."
        )

        docs = await self.opensearch_retriever.ainvoke(question)
        await self.bot.edit_message_text(
            chat_id=telegram_chat_id,
            message_id=msg.message_id,
            text=f"📚 Retrieved {len(docs)} documents"
        )
        
        return docs

    async def _grade_single_document(self, question: str, doc: Document) -> bool:
        """Grade a single document for relevance."""
        grade = await self.retrieval_grader.ainvoke(
            question=question,
            document=f"{doc.page_content} \n\n {doc.metadata}"
        )
        print(f"---DOC: {doc.metadata['time']} {doc.page_content[:100]}... GRADE: {grade}---")
        return grade == "yes"

    async def retrieve_code_docs(self, telegram_chat_id: int, stack_trace: str) -> List[Document]:
        """Retrieve relevant code documents for the stack trace."""
        msg = await self.bot.send_message(
            chat_id=telegram_chat_id,
            text="🔍 Searching codebase for relevant files..."
        )

        code_docs = await self.code_base_retriever.ainvoke(stack_trace)
        
        # Get list of unique file names
        file_names = set(doc.metadata.get('filename', 'Unknown') for doc in code_docs)
        file_list = "\n".join(f"• {name}" for name in sorted(file_names))
        
        # Format stack trace for display
        stack_trace_lines = stack_trace.split('\n')
        formatted_stack = "\n".join(f"  {line}" for line in stack_trace_lines)
        
        await self.bot.edit_message_text(
            chat_id=telegram_chat_id,
            message_id=msg.message_id,
            text=f"🔍 Searching for code related to stack trace:\n```\n{formatted_stack}\n```\n\n📚 Found {len(code_docs)} relevant code files:\n{file_list}"
        )
        
        return code_docs

    async def generate_and_send_answer(self, telegram_chat_id: int, question: str, doc: Document, code_docs: List[Document], doc_num: int, total_docs: int) -> None:
        """Generate and send answer for a single document and its associated code snippets."""
        msg = await self.bot.send_message(
            chat_id=telegram_chat_id,
            text=f"🤖 Generating answer for document {doc_num}/{total_docs}..."
        )

        # Prepare context and code snippets
        context = f"Log Entry:\n{doc.page_content}"
        code_context = ""
        stack_trace = doc.metadata.get('stack_trace', '')
        
        if code_docs:
            code_context = "\n\n".join(f"Code Snippet:\n{code_doc.page_content}" for code_doc in code_docs)
        
        # Prepare input for answerer
        answerer_input = {
            "context": context,
            "question": question,
            "code_context": code_context,
            "stack_trace": stack_trace
        }
        
        response = await self.answerer.ainvoke(answerer_input)

        # Send the answer
        await self.bot.send_message(
            chat_id=telegram_chat_id,
            text=f"📝 Answer for document {doc_num}/{total_docs}:\n\n{response}"
        )
        
        # Update status message
        await self.bot.edit_message_text(
            chat_id=telegram_chat_id,
            message_id=msg.message_id,
            text=f"✅ Processed document {doc_num}/{total_docs}"
        )
        