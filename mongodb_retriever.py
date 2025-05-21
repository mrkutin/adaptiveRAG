from typing import List, Dict, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from pprint import pformat
from config import settings
import certifi


class MongoDBRetriever(BaseRetriever):
    """A custom retriever that searches documents in MongoDB collections."""

    # MongoDB configuration from settings
    hosts: List[str] = Field(default=settings.mongodb_hosts)
    username: str = Field(default=settings.mongodb_username)
    password: str = Field(default=settings.mongodb_password)
    database: str = Field(default=settings.mongodb_database)
    replica_set: str = Field(default=settings.mongodb_replica_set)
    auth_source: str = Field(default=settings.mongodb_auth_source)
    mongodb_query_limit: int = Field(default=settings.mongodb_query_limit)
    use_ssl: bool = Field(default=settings.mongodb_use_ssl)
    verify_certs: bool = Field(default=settings.mongodb_verify_certs)
    ca_cert_path: str = Field(default=settings.mongodb_ca_cert_path)
        
    # Use private attributes
    _client: MongoClient = PrivateAttr()
    _aclient: AsyncIOMotorClient = PrivateAttr()
    _collection_configs: Dict[str, Dict[str, Any]] = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize MongoDB client with SSL/TLS
        connection_string = f"mongodb://{self.username}:{self.password}@{','.join(self.hosts)}/{self.database}?replicaSet={self.replica_set}&authSource={self.auth_source}"
        print(f"Connecting to MongoDB with connection string: {connection_string}")
        
        # Configure SSL/TLS options
        tls_options = {}
        if self.use_ssl:
            tls_options = {
                "tls": True,
                "tlsAllowInvalidCertificates": not self.verify_certs,
                "tlsCAFile": self.ca_cert_path
            }
        
        print(f"Connecting to MongoDB with options: {tls_options}")
        self._client = MongoClient(connection_string, **tls_options)
        self._aclient = AsyncIOMotorClient(connection_string, **tls_options)

        # Define collection configurations
        self._collection_configs = {
            "items": {
                "search_fields": [
                    "itemid",
                    "namealias",
                    "inventcontent.namealias",
                    "inventcontent.authors",
                    "inventcontent.annotation",
                    "inventcontent.notes",
                    "inventcontentgroup.namealias",
                    "inventedition.isbn",
                    "inventedition.inventeditionid"
                ],
                "metadata_fields": [
                    "itemid",
                    "namealias",
                    "inventcontent.namealias",
                    "inventcontent.authors",
                    "inventcontent.inventcontentcode",
                    "inventcontent.inventcontentid",
                    "inventedition.isbn",
                    "inventedition.inventeditionid",
                    "inventedition.inventeditioncode",
                    "inventedition.inventtitleyearid",
                    "inventcontentgroup.namealias",
                    "inventcontentgroup.inventcontentgroupcode",
                    "inventcontentgroup.language",
                    "inventcontentgroup.numberparts",
                    "inventcontentgroup.partnumber",
                    "updated_at"
                ],
                "content_field": "inventcontent.notes",
                "query_patterns": {
                    "isbn": {
                        "pattern": "isbn",
                        "extract_after": "ISBN",
                        "search_fields": ["inventedition.isbn"]
                    },
                    "author": {
                        "pattern": "author",
                        "extract_after": "author",
                        "search_fields": ["inventcontent.authors"]
                    },
                    "topic": {
                        "pattern": "about",
                        "extract_after": "about",
                        "search_fields": [
                            "namealias",
                            "inventcontent.namealias",
                            "inventcontent.annotation",
                            "inventcontent.notes"
                        ]
                    }
                }
            }
        }

    def _build_query(self, collection: str, query: str) -> Dict[str, Any]:
        """Build MongoDB query based on collection configuration."""
        config = self._collection_configs.get(collection)
        if not config:
            raise ValueError(f"Unknown collection: {collection}")

        query_lower = query.lower()
        
        # Check each query pattern in the configuration
        for pattern_config in config["query_patterns"].values():
            if pattern_config["pattern"] in query_lower:
                # Extract search term after the specified text
                search_term = query.split(pattern_config["extract_after"])[-1].strip()
                
                # If only one search field, return simple query
                if len(pattern_config["search_fields"]) == 1:
                    return {pattern_config["search_fields"][0]: {"$regex": search_term, "$options": "i"}}
                
                # If multiple search fields, return OR query
                return {
                    "$or": [
                        {field: {"$regex": search_term, "$options": "i"}}
                        for field in pattern_config["search_fields"]
                    ]
                }
        
        # If no pattern matches, use all search fields
        return {
            "$or": [
                {field: {"$regex": query, "$options": "i"}}
                for field in config["search_fields"]
            ]
        }

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in MongoDB.
        
        Args:
            query: The natural language query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        all_docs = []
        
        # Search in each configured collection
        for collection_name, config in self._collection_configs.items():
            print(f"\nSearching in collection: {collection_name}")
            
            # Build query for this collection
            mongo_query = self._build_query(collection_name, query)
            print(f"MongoDB query: {pformat(mongo_query)}")
            print("--------------------------------")
            
            # Execute search
            collection = self._client[self.database][collection_name]
            cursor = collection.find(mongo_query).limit(self.mongodb_query_limit)

            # Convert MongoDB results to Documents
            for doc in cursor:
                # Extract content and metadata based on collection config
                content = doc.get(config["content_field"], "")
                if not content and "inventcontent" in doc:
                    content = doc["inventcontent"].get("notes", "")
                
                metadata = {}
                for field in config["metadata_fields"]:
                    if "." in field:
                        # Handle nested fields
                        parts = field.split(".")
                        value = doc
                        for part in parts:
                            if isinstance(value, dict):
                                value = value.get(part, "")
                            else:
                                value = ""
                                break
                        metadata[field] = value
                    else:
                        metadata[field] = doc.get(field, "")
                
                metadata["collection"] = collection_name
                
                all_docs.append(
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                )
        
        print(f"--- RETRIEVED {len(all_docs)} documents total")
        return all_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in MongoDB asynchronously.
        
        Args:
            query: The natural language query
            run_manager: Callback manager
            
        Returns:
            List of found documents
        """
        all_docs = []
        
        # Search in each configured collection
        for collection_name, config in self._collection_configs.items():
            print(f"\nSearching in collection: {collection_name}")
            
            # Build query for this collection
            mongo_query = self._build_query(collection_name, query)
            print(f"MongoDB query: {pformat(mongo_query)}")
            print("--------------------------------")
            
            # Execute search
            collection = self._aclient[self.database][collection_name]
            cursor = collection.find(mongo_query).limit(self.mongodb_query_limit)

            # Convert MongoDB results to Documents
            async for doc in cursor:
                # Extract content and metadata based on collection config
                content = doc.get(config["content_field"], "")
                if not content and "inventcontent" in doc:
                    content = doc["inventcontent"].get("notes", "")
                
                metadata = {}
                for field in config["metadata_fields"]:
                    if "." in field:
                        # Handle nested fields
                        parts = field.split(".")
                        value = doc
                        for part in parts:
                            if isinstance(value, dict):
                                value = value.get(part, "")
                            else:
                                value = ""
                                break
                        metadata[field] = value
                    else:
                        metadata[field] = doc.get(field, "")
                
                metadata["collection"] = collection_name
                
                all_docs.append(
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                )
        
        print(f"--- RETRIEVED {len(all_docs)} documents total")
        return all_docs


# Example usage:
if __name__ == "__main__":
    retriever = MongoDBRetriever()
    
    tests = [
        "Find item with ISBN 978-5-9963-5562-4",
        "Show items by author Репкин",
        "Find items about Букварь",
        # Add more test queries here
    ]

    for test in tests:
        print(f"\nTest: {test}")
        print("--------------------------------")
        docs = retriever.invoke(test)
        for doc in docs:
            print(f"Collection: {doc.metadata.get('collection')}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {pformat(doc.metadata)}")
            print("--------------------------------") 