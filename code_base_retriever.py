from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_ollama import ChatOllama
from pydantic import Field, PrivateAttr
import re

from config import settings


class CodeBaseRetriever(BaseRetriever):
    """A custom retriever that searches documents in codebase using vector store."""

    # Use private attributes
    _loader: GenericLoader = PrivateAttr()
    _embeddings: OllamaEmbeddings = PrivateAttr()
    _vector_store: Chroma = PrivateAttr()
    _retriever: BaseRetriever = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize loader
        self._loader = GenericLoader.from_filesystem(
            settings.codebase_path,
            glob=settings.codebase_file_pattern,
            suffixes=settings.codebase_file_extensions,
            parser=LanguageParser(language=settings.codebase_language)
        )
        
        # Initialize embeddings
        self._embeddings = OllamaEmbeddings(model=settings.codebase_embedding_model)
        
        # Initialize vector store (in-memory)
        self._vector_store = Chroma(
            collection_name="codebase",
            embedding_function=self._embeddings
        )
        
        #TODO  Load documents
        docs = self._loader.load()
        print(f"Codebase loaded successfully! Docs loaded: {len(docs)}")
        
        # Add filename to metadata
        for doc in docs:
            source = doc.metadata.get("source", "")
            doc.metadata["filename"] = source.split("/")[-1] if source else ""
        
        self._vector_store.add_documents(documents=docs)
        print(f"Documents added to vector store successfully!")
        
        # Initialize LLM for query construction
        llm = ChatOllama(
            model=settings.retriever_ollama_model,
            temperature=settings.retriever_ollama_temperature
        )
        
        # Define metadata field info
        metadata_field_info = [
            {
                "name": "source",
                "description": "The full file path of the code, e.g. 'code_base/enterprise-service-bus/services/one-c.service.js'",
                "type": "string"
            },
            {
                "name": "filename",
                "description": "The name of the file, e.g. 'one-c.service.js'",
                "type": "string"
            },
            {
                "name": "language",
                "description": "The programming language of the code",
                "type": "string"
            }
        ]
        
        # Define document content description
        document_content_description = "Code files from the codebase. When users ask about specific files, filter by the filename field. For example, if they ask 'what is in one-c.service.js', filter for filename equal to 'one-c.service.js'."
        
        # Create query constructor
        prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info
        )
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | llm | output_parser
        
        # Initialize self-query retriever
        self._retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=self._vector_store,
            structured_query_translator=ChromaTranslator(),
            verbose=True
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in codebase using vector store.
        
        Args:
            query: The natural language query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        # If query contains stack trace, extract files and search for them
        if "stack:" in query or "at " in query:
            print("\nExtracting files from stack trace...")
            # Extract file paths from stack trace
            file_paths = set()  # Use set to deduplicate
            # Find all filenames between / and .js
            matches = re.findall(r'/([^/]+\.js)', query)
            for filename in matches:
                print(f"Found file: {filename}")
                file_paths.add(filename)
            
            if file_paths:
                print(f"\nFound files in stack trace: {sorted(file_paths)}")
                # Create a query that searches for all files
                file_query = " OR ".join(f"filename:{filepath}" for filepath in sorted(file_paths))
                print(f"Searching for: {file_query}")
                return self._retriever.invoke(file_query)
            else:
                print("No JavaScript files found in stack trace")
        
        # If not a stack trace or no files found, use normal search
        return self._retriever.invoke(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in codebase using vector store asynchronously.
        
        Args:
            query: The natural language query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        # If query contains stack trace, extract files and search for them
        if "stack:" in query or "at " in query:
            print("\nExtracting files from stack trace...")
            # Extract file paths from stack trace
            file_paths = set()  # Use set to deduplicate
            # Find all filenames between / and .js
            matches = re.findall(r'/([^/]+\.js)', query)
            for filename in matches:
                print(f"Found file: {filename}")
                file_paths.add(filename)
            
            if file_paths:
                print(f"\nFound files in stack trace: {sorted(file_paths)}")
                # Create a query that searches for all files
                file_query = " OR ".join(f"filename:{filepath}" for filepath in sorted(file_paths))
                print(f"Searching for: {file_query}")
                return await self._retriever.ainvoke(file_query)
            else:
                print("No JavaScript files found in stack trace")
        
        # If not a stack trace or no files found, use normal search
        return await self._retriever.ainvoke(query)


# Example usage:
if __name__ == "__main__":
    retriever = CodeBaseRetriever()
    
    test_queries = [
        # "what is in one-c.service.js?",
        # "what generatorLoop does in generator.mixin.js?",
        # "how to handle errors?",
        # "what is the main function?",
        # "where is this error handled? MoleculerServerError: AxiosError: Request failed with status code 400 at Service.handler (/app/services/crm.service.js:199:13)?",
        """stack: MoleculerServerError: MoleculerServerError: crmRes.data.success: false
            at Service.handler (/app/services/crm.service.js:199:13)
            at process.processTicksAndRejections (node:internal/process/task_queues:95:5)
            at async /app/middlewares/metricsMiddleware.js:16:17
            at async /app/services/crm.service.js:119:6
            at async Promise.all (index 0)
            at async Service.crmLoop (/app/services/crm.service.js:117:4)
            at async Service.crmStartLoop (/app/services/crm.service.js:40:5)"""
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("--------------------------------")
        docs = retriever.invoke(query)
        if docs:
            print(f"Found {len(docs)} documents")
            print("--------------------------------")
        for doc in docs:
            # print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            print("--------------------------------")
