from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from opensearchpy import OpenSearch, AsyncOpenSearch
from pprint import pformat
from config import settings


class OpenSearchRetriever(BaseRetriever):
    """A custom retriever that searches documents in OpenSearch with query translation."""

    # OpenSearch configuration from settings
    host: str = Field(default=settings.opensearch_host)
    port: int = Field(default=settings.opensearch_port)
    username: str = Field(default=settings.opensearch_username)
    password: str = Field(default=settings.opensearch_password)
    index: str = Field(default=settings.opensearch_index)
    use_ssl: bool = Field(default=settings.opensearch_use_ssl)
    verify_certs: bool = Field(default=settings.opensearch_verify_certs)
    opensearch_query_size: int = Field(default=settings.opensearch_query_size)
    
    # Ollama configuration for query translation
    ollama_base_url: str = Field(default=settings.retriever_ollama_base_url)
    ollama_model: str = Field(default=settings.retriever_ollama_model)
    ollama_temperature: float = Field(default=settings.retriever_ollama_temperature)
    ollama_timeout: int = Field(default=settings.retriever_ollama_timeout)
    ollama_max_tokens: int = Field(default=settings.retriever_ollama_max_tokens)
    
    # Use private attributes
    _client: OpenSearch = PrivateAttr()
    _aclient: AsyncOpenSearch = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize OpenSearch client
        self._client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=(self.username, self.password) if self.username else None,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            ssl_show_warn=False
        )

        # Initialize AsyncOpenSearch client
        self._aclient = AsyncOpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=(self.username, self.password) if self.username else None,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            ssl_show_warn=False
        )
        

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in OpenSearch using translated query.
        
        Args:
            query: The natural language query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        # Use OpenSearchQueryConstructor to build the query
        opensearch_query = self._query_constructor.construct_query(query)
        print(f"OpenSearch query: {pformat(opensearch_query)}")
        print("--------------------------------")
        
        # Execute search
        result = self._client.search(
            index=self.index,
            body={
                "query": opensearch_query,
                "size": self.opensearch_query_size,
            }
        )

        print(f"--- RETRIEVED {len(result['hits']['hits'])} documents")
        
        # Convert OpenSearch results to Documents
        docs = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            docs.append(
                Document(
                    page_content=source.get("msg", ""),
                    metadata={
                        "level": source.get("level"),
                        "ns": source.get("ns"),
                        "svc": source.get("svc"),
                        "time": source.get("time"),
                        "score": hit["_score"]
                    }
                )
            )
        
        return docs


    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in OpenSearch using translated query.
        
        Args:
            query: The natural language query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        # Use OpenSearchQueryConstructor to build the query
        opensearch_query = await self._query_constructor.aconstruct_query(query)
        print(f"OpenSearch query: {pformat(opensearch_query)}")
        print("--------------------------------")
        
        # Execute search
        result = await self._aclient.search(
            index=self.index,
            body={
                "query": opensearch_query,
                "size": self.opensearch_query_size,
            }
        )

        print(f"--- RETRIEVED {len(result['hits']['hits'])} documents")
        
        # Convert OpenSearch results to Documents
        docs = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            docs.append(
                Document(
                    page_content=source.get("msg", ""),
                    metadata={
                        "level": source.get("level"),
                        "ns": source.get("ns"),
                        "svc": source.get("svc"),
                        "time": source.get("time"),
                        "score": hit["_score"]
                    }
                )
            )
        
        return docs



# Example usage:
if __name__ == "__main__":
    retriever = OpenSearchRetriever()
    
    tests = [
        # "What are errors in prod today?",
        "What are crm errors in prod today?",
        # "What are Mindbox upload server errors in topic id-authorize-customer-topic?",
        # "What are errors in prod last hour?",
        # "What are errors in prod last 20 hours?",
        # "What is wrong with order PSV-745559?",
        # "What is wrong with order PSV-737844-К0015742?",
        # "What happened with item NM0086817 on test?",
        # "What are steps of item NM0098877?",
        # "What are errors in prod from 2025-03-20 to 2025-03-21?",
        # "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
        # "What happened with order PSV-745559 from 2025-03-20 10:00:00 to 2025-03-21 11:35:56?",
        # "What are logs from 16:00:00 to now?",
        # "What are logs on prod from 16:35:11 to 16:36:56?",
        # "What are warnings in prod this month?",
        # "What are errors in test last month?",
        # "What are Mindbox upload errors in test this week?",
        # "What are info messages in prod last week?",
    ]

    for test in tests:
        print(f"Test: {test}")
        print("--------------------------------")
        docs = retriever.invoke(test)
        for doc in docs:
            # print(f"Content: {pformat(doc.page_content)}")
            print(f"Metadata: {pformat(doc.metadata)}")
            print("--------------------------------")
    