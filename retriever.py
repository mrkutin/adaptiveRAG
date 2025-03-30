from typing import List, Optional, Dict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from opensearchpy import OpenSearch

class OpenSearchRetriever(BaseRetriever):
    """A custom retriever that searches documents in OpenSearch."""

    # OpenSearch configuration
    host: str = Field(default="opensearch-data.prod.prosv.yc")
    port: int = Field(default=9200)
    username: str = Field(default="bus-admin")
    password: str = Field(default="WZJ7WKimoLFWCzV")
    index: str = Field(default="bus-prod-info-*")
    use_ssl: bool = Field(default=True)
    verify_certs: bool = Field(default=False)
    query_size: int = Field(default=10)

    # Use PrivateAttr for the client instance
    _client: OpenSearch = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=(self.username, self.password) if self.username else None,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            ssl_show_warn=False
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in OpenSearch.
        
        Args:
            query: The search query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        result = self._client.search(
            index=self.index,
            body={
                "query": {
                    "match": {
                        "msg": query
                    }
                },
                "size": self.query_size
            }
        )
        
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
    results = retriever.invoke("successfully posted to Mindbox")
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")
