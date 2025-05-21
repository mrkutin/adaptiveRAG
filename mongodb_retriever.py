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
from mongodb_query_constructor import MongoDBQueryConstructor


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
    _query_constructor: MongoDBQueryConstructor = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize query constructor
        self._query_constructor = MongoDBQueryConstructor()

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

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search documents in MongoDB."""
        all_docs = []
        
        # Search in each configured collection
        for collection_name in self._query_constructor.collection_configs.keys():
            print(f"\nSearching in collection: {collection_name}")
            
            # Get collection config
            config = self._query_constructor.get_collection_config(collection_name)
            
            # Build query for this collection
            mongo_query = self._query_constructor.construct_query(query, collection_name)
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
        """Search documents in MongoDB asynchronously."""
        all_docs = []
        
        # Search in each configured collection
        for collection_name in self._query_constructor.collection_configs.keys():
            print(f"\nSearching in collection: {collection_name}")
            
            # Get collection config
            config = self._query_constructor.get_collection_config(collection_name)
            
            # Build query for this collection
            mongo_query = await self._query_constructor.aconstruct_query(query, collection_name)
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
        # Items collection tests
        # "Find item with ISBN 978-5-9963-5562-4",
        # "Show items by author Репкин",
        # "Find items about Букварь",
        
        # CRM Agreements collection tests
        # "Find agreement with code PSVK460534",
       str("Show agreements for customer К0015433"),
        # "Find agreements with document title A0053335",
        # "Show agreements in classification Бюджетные заказы",
        # "Find agreements with GAK 39286",
        # "Show agreements with CFO 020002",
        # "Find agreements created on 2023-05-05",
        # "Show agreements with delivery date 2023-09-30",
        # "Find agreements with signing status 4",
        # "Show agreements with CRM status success",
        # "Find agreements with document reference Контракт № A0053335"
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