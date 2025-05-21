from typing import Dict, Any, List, Tuple, Union
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import json
from config import settings


class MongoDBQueryConstructor:
    """Constructs MongoDB queries from natural language questions."""
    
    def __init__(self):
        self.llm = Ollama(
            base_url=settings.mongodb_retriever_ollama_base_url,
            model=settings.mongodb_retriever_ollama_model,
            temperature=settings.mongodb_retriever_ollama_temperature,
            num_ctx=settings.mongodb_retriever_ollama_num_ctx
        )
        
        self.query_analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analyzer for a MongoDB database. Your task is to analyze natural language queries and determine:
1. The search intent (what the user is looking for)
2. The relevant search terms
3. The most appropriate fields to search in

IMPORTANT: You MUST preserve the EXACT characters from the input query in the search_term. 
Do not convert or transliterate any characters (e.g., keep Cyrillic 'К' as is, don't convert it to Latin 'K').

Available fields for the 'items' collection:
{fields}

You must respond with a valid JSON object in this exact format:
{{
    "intent": "isbn|author|topic|general",
    "search_term": "the actual term to search for (preserve exact characters)",
    "fields": ["list", "of", "relevant", "fields"]
}}

Do not include any other text or explanation, just the JSON object."""),
            ("user", "{query}")
        ])

        # Define collection configurations
        self.collection_configs = {
            "items": {
                "exact_match_fields": [
                    "itemid",
                    "inventedition.isbn",
                    "inventedition.inventeditionid"
                ],
                "regex_match_fields": [
                    "namealias",
                    "inventcontent.namealias",
                    "inventcontent.authors",
                    "inventcontent.annotation",
                    "inventcontent.notes",
                    "inventcontentgroup.namealias"
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
                "content_field": "inventcontent.notes"
            },
            "crm-agreements": {
                "exact_match_fields": [
                    "code",
                    "recid",
                    "cust_account_code",
                    "gak",
                    "cfo",
                    "agreementid_pik",
                    "agreementiid_kz",
                    "header_rec_id"
                ],
                "regex_match_fields": [
                    "documentExternalReference",
                    "document_title",
                    "classification_name",
                    "owner_executor"
                ],
                "metadata_fields": [
                    "code",
                    "recid",
                    "channel_code",
                    "classification_name",
                    "client_type_code",
                    "company_source",
                    "created_at",
                    "created_date",
                    "currency",
                    "cust_account_code",
                    "default_effective_date",
                    "delivery_date",
                    "documentExternalReference",
                    "document_date",
                    "document_title",
                    "edo_type_code",
                    "end_date",
                    "gak",
                    "header_rec_id",
                    "owner_executor",
                    "payment_schedule",
                    "sales_district_code",
                    "signing_date",
                    "signing_status_code",
                    "source_code",
                    "status_code",
                    "updated_at",
                    "vat_amount",
                    "sent_to_crm",
                    "cfo",
                    "management_accouting_article",
                    "agreementid_pik",
                    "agreementiid_kz",
                    "attempts",
                    "crm_status",
                    "is_correct_efu"
                ],
                "content_field": "documentExternalReference"
            }
        }

    def _analyze_query(self, query: str, collection: str) -> Tuple[str, str, List[str]]:
        """Analyze the query using AI to determine intent, search term, and relevant fields."""
        config = self.collection_configs.get(collection)
        if not config:
            raise ValueError(f"Unknown collection: {collection}")
            
        # Format available fields for the prompt
        all_fields = config["exact_match_fields"] + config["regex_match_fields"]
        fields_str = "\n".join(f"- {field}" for field in all_fields)
        
        # Get AI analysis
        response = self.llm.invoke(
            self.query_analyzer_prompt.format(
                query=query,
                fields=fields_str
            )
        )
        
        # Parse the response
        try:
            # Extract JSON from the response (in case there's any extra text)
            json_str = response[response.find("{"):response.rfind("}")+1]
            analysis = json.loads(json_str)
            return (
                analysis["intent"],
                analysis["search_term"],
                analysis["fields"]
            )
        except:
            # Fallback to general search if analysis fails
            return "general", query, all_fields

    def construct_query(self, query: str, collection: str) -> Dict[str, Any]:
        """Construct a MongoDB query from a natural language question."""
        # Use AI to analyze the query
        intent, search_term, fields = self._analyze_query(query, collection)
        config = self.collection_configs[collection]

        # Build the query based on the analysis
        if len(fields) == 1:
            # Use exact matching for identifiers and case-insensitive for text
            if fields[0] in config["exact_match_fields"]:
                return {fields[0]: search_term}
            return {fields[0]: {"$regex": search_term, "$options": "i"}}
        
        # For multiple fields, use exact matching for identifiers
        exact_matches = []
        regex_matches = []
        
        for field in fields:
            if field in config["exact_match_fields"]:
                exact_matches.append({field: search_term})
            else:
                regex_matches.append({field: {"$regex": search_term, "$options": "i"}})
        
        if exact_matches and regex_matches:
            return {
                "$or": [
                    {"$and": exact_matches},
                    *regex_matches
                ]
            }
        elif exact_matches:
            return {"$or": exact_matches}
        else:
            return {"$or": regex_matches}

    async def aconstruct_query(self, query: str, collection: str) -> Dict[str, Any]:
        """Construct a MongoDB query from a natural language question asynchronously."""
        # For now, use synchronous version since Ollama doesn't support async
        return self.construct_query(query, collection)

    def get_collection_config(self, collection: str) -> Dict[str, Any]:
        """Get configuration for a specific collection."""
        config = self.collection_configs.get(collection)
        if not config:
            raise ValueError(f"Unknown collection: {collection}")
        return config 