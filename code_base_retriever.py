# from langchain_community.document_loaders import GitLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr


class CodeBaseRetriever(BaseRetriever):
    """A custom retriever that searches documents in codebase using vector store."""

    # Configuration
    codebase_path: str = Field(default="./code_base/enterprise-service-bus")
    file_pattern: str = Field(default="**/*")
    file_extensions: List[str] = Field(default=[".js"])
    language: str = Field(default="js")
    embedding_model: str = Field(default="unclemusclez/jina-embeddings-v2-base-code")
    k: int = Field(default=1)
    persist_directory: str = Field(default="./chroma_db")
    
    # Use private attributes
    _loader: GenericLoader = PrivateAttr()
    _embeddings: OllamaEmbeddings = PrivateAttr()
    _vector_store: Chroma = PrivateAttr()
    _retriever: BaseRetriever = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize loader
        self._loader = GenericLoader.from_filesystem(
            self.codebase_path,
            glob=self.file_pattern,
            suffixes=self.file_extensions,
            parser=LanguageParser(language=self.language)
        )
        
        # Initialize embeddings
        self._embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Initialize vector store
        self._vector_store = Chroma(
            persist_directory=self.persist_directory,
            collection_name="codebase",
            embedding_function=self._embeddings
        )
        
        #TODO  Load documents
        docs = self._loader.load()
        print(f"Codebase loaded successfully! Docs loaded: {len(docs)}")
        self._vector_store.add_documents(documents=docs)
        print(f"Documents added to vector store successfully!")
        
        # Initialize retriever
        self._retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
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
        return self._retriever.invoke(query, filter={"source": "code_base/enterprise-service-bus/services/one-c.service.js"})

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
        return await self._retriever.ainvoke(query)


# Example usage:
if __name__ == "__main__":
    retriever = CodeBaseRetriever()
    
    test_queries = [
        "what is in one-c.service.js?",
        # "what generatorLoop does?",
        # "how to handle errors?",
        # "what is the main function?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("--------------------------------")
        docs = retriever.invoke(query)
        for doc in docs:
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            print("--------------------------------")
