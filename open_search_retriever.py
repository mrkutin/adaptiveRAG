from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from opensearchpy import OpenSearch
from langchain_ollama import OllamaLLM
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from open_search_translator import CustomOpenSearchTranslator
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
    _query_constructor = PrivateAttr()
    _translator: CustomOpenSearchTranslator = PrivateAttr()

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
        
        # Initialize query translation components
        query_model = OllamaLLM(
            base_url=self.ollama_base_url,
            model=self.ollama_model,
            temperature=self.ollama_temperature,
            timeout=self.ollama_timeout,
            max_tokens=self.ollama_max_tokens,
        )

        # Define query contents
        document_contents = "Log storage containing system events and error messages. Use only attributes 'time', 'level', 'ns', 'svc' in filter."

        # Define query attributes
        attribute_info = [
            AttributeInfo(
                name="time",
                description="The datetime in ISO 8601 format (if exact time is needed) or relative time (e.g. 'now', 'now/m', 'now/w', 'now/d', 'now-1h', 'now-24h'). When exact time is specified, substract 3 hours from the time to get the Greenwich Mean Time (GMT). now/m means start of the current month, now/w means start of the current week, now/d means start of the current day.",
                type="string",
            ),
            AttributeInfo(
                name="level",
                description="The severity level of the log entry (e.g., info, warn, error, debug, trace, panic, fatal).",
                type="string",
            ),
            AttributeInfo(
                name="ns",
                description="The namespace of the log entry, prod or test. Use prod if not specified",
                type="string",
            ),
        ]

        # Define query examples
        examples = [
            (
                "What are Mindbox upload server errors in topic id-authorize-customer-topic?",
                {       
                    "query": "mindbox upload server error id-authorize-customer-topic",
                    "filter": "and(eq('level', 'error'))",
                },  
            ),
            (
                "What are errors in prod last hour?",
                {       
                    "query": "error",
                    "filter": "and(eq('level', 'error'), eq('ns', 'prod'), gte('time', 'now-1h'))",
                },  
            ),
            (
                "What is wrong with order PSV-745559?",
                {       
                    "query": "PSV-745559",
                    "filter": "and(or(eq('level', 'error'), eq('level', 'warn')), eq('ns', 'prod'))",
                },  
            ),
            (
                "What is wrong with order PSV-737844-К0015742?",
                {       
                    "query": "PSV-737844-К0015742",
                    "filter": "and(or(eq('level', 'error'), eq('level', 'warn')), eq('ns', 'prod'))",
                },  
            ),
            (
                "What happened with item NM0086817 on test?",
                {       
                    "query": "NM0086817",
                    "filter": "and(eq('ns', 'test'))",
                },  
            ),
            (
                "What are errors in prod from 2025-03-20 to 2025-03-21?",
                {       
                    "query": "error",
                    "filter": "and(eq('level', 'error'), eq('ns', 'prod'), gte('time', '2025-03-20'), lte('time', '2025-03-21'))",
                },  
            ),
            (
                "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
                {       
                    "query": "mindbox upload error",
                    "filter": "and(eq('level', 'error'), eq('ns', 'test'), eq('svc', 'mindbox'), gte('time', '2025-03-20T07:00:00Z'), lte('time', '2025-03-21T07:00:00Z'))",
                },  
            ),
            (
                "What happened with order PSV-745559 from 2025-03-20 10:00:00 to 2025-03-21 11:35:56?",
                {       
                    "query": "PSV-745559",
                    "filter": "and(gte('time', '2025-03-20T07:00:00Z'), lte('time', '2025-03-21T08:35:56Z'))",
                },  
            ),
            (
                "What are logs from 16:00:00 to now?",
                {       
                    "query": "NO_FILTER",
                    "filter": "and(gte('time', 'now/d+13h'))",
                },  
            ),
            (
                "What are logs on prod from 16:35:11 to 16:36:56?",
                {       
                    "query": "NO_FILTER",
                    "filter": "and(eq('ns', 'prod'), gte('time', 'now/d+13h+35m+11s'), lte('time', 'now/d+13h+36m+56s'))",
                },  
            ),
            (
                "What are Mindbox upload errors in test this week?",
                {       
                    "query": "mindbox upload error",
                    "filter": "and(eq('level', 'error'), eq('ns', 'test'), eq('svc', 'mindbox'), gte('time', 'now/w'))",
                },  
            ),
        ]


        # Create the query constructor chain
        constructor_prompt = get_query_constructor_prompt(
            document_contents,
            attribute_info,
            examples=examples,
        )
        
        output_parser = StructuredQueryOutputParser.from_components()
        self._query_constructor = constructor_prompt | query_model | output_parser
        self._translator = CustomOpenSearchTranslator()

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
        # Translate query using the same flow as in self_query_example.py
        structured_query = self._query_constructor.invoke({"query": query})
        print(f"Structured query: {pformat(structured_query)}") 
        print("--------------------------------")
        opensearch_query = self._translator.visit_structured_query(structured_query)
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

        print(f"result length: {len(result['hits']['hits'])}")
        
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
    
    
    
    
    
    
    
    
    