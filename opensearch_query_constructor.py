from typing import Dict, Any, List, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import json
from pprint import pprint
from config import settings

class OpenSearchQuery(BaseModel):
    """OpenSearch query structure."""
    bool: Dict[str, Any] = Field(
        description="The bool query structure containing filter and must clauses"
    )

class OpenSearchQueryConstructor:
    """Constructs OpenSearch queries from natural language questions."""
    
    def __init__(self):
        self.llm = ChatOllama(
            base_url=settings.retriever_ollama_base_url,
            model=settings.retriever_ollama_model,
            temperature=settings.retriever_ollama_temperature,
            timeout=settings.retriever_ollama_timeout,
            # max_tokens=settings.retriever_ollama_max_tokens,
            num_ctx=8192
        )
        
        self.base_template = (
            "You are an assistant that converts user questions into OpenSearch queries. "
            "The query must have a 'bool' structure with 'filter' and 'must' sections. "
            "The 'filter' section should ONLY contain 'term' clauses for 'level', 'ns', and 'time' fields. "
            "The 'must' section should contain 'match' or 'term' clauses for the 'msg' field. "
            "If searching for specific values like order numbers, item numbers, customer numbers, or topic IDs, use 'term' in the 'must' section. "
            "For general text matching, use 'match' in the 'must' section. "
            "NEVER put 'match' or 'term' clauses for 'msg' field in the 'filter' section. "
            "ALWAYS put 'msg' field clauses in the 'must' section. "
            "Skip clauses if they are not needed. "
            "Output a valid OpenSearch query as JSON. "
            "The output must be a valid JSON object, nothing else. "
            "When using exact time (not relative like 'now'), add 'time_zone': '+03:00' to the time range. "
            "Here are some examples:\n\n"
            "{examples}\n"
            "User: {question}\n"
            "OpenSearch query:\n"
        )
        
        self.examples: List[Tuple[str, Dict[str, Any]]] = [
            (
                "What were API service errors on April 11 2025?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {
                                "range": {
                                    "time": {
                                        "gte": "2025-04-11T00:00:00Z", 
                                        "lte": "2025-04-11T23:59:59Z",
                                        "time_zone": "+03:00"
                                    }
                                }
                            }
                        ],
                        "must": [
                            {"match": {"msg": "API service error"}}
                        ]
                    }
                }
            ),
            (
                "What are warnings in prod this week?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "warn"}},
                            {"term": {"ns": "prod"}},
                            {"range": {"time": {"gte": "now/w"}}}
                        ]
                    }
                }
            ),
            (
                "What are errors in prod last month?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {"term": {"ns": "prod"}},
                            {"range": {"time": {"gte": "now-1M"}}}
                        ]
                    }
                }
            ),
            (
                "What are errors in prod last hour?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {"term": {"ns": "prod"}},
                            {"range": {"time": {"gte": "now-1h"}}}
                        ]
                    }
                }
            ),
            (
                "What are errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {"term": {"ns": "test"}},
                            {
                                "range": {
                                    "time": {
                                        "gte": "2025-03-20T10:00:00Z", 
                                        "lte": "2025-03-21T10:00:00Z",
                                        "time_zone": "+03:00"   
                                    }
                                }
                            }
                        ]
                    }
                }
            ),
            (
                "What happened with order PSV-745559?",
                {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "msg": "PSV-745559"
                                }
                            }
                        ]
                    }
                }
            ),
            (
                "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {"term": {"ns": "test"}},
                            {
                                "range": {
                                    "time": {
                                        "gte": "2025-03-20T10:00:00Z", 
                                        "lte": "2025-03-21T10:00:00Z",
                                        "time_zone": "+03:00"   
                                    }
                                }
                            }
                        ],
                        "must": [
                            {"match": {"msg": "mindbox upload error"}}
                        ]
                    }
                }
            ),
            (
                "What are crm errors in prod today?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {"term": {"ns": "prod"}},
                            {"range": {"time": {"gte": "now/d"}}}
                        ],
                        "must": [
                            {"match": {"msg": "crm error"}}
                        ]
                    }
                }
            ),
            (
                "What are Mindbox upload server errors in topic id-authorize-customer-topic?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"level": "error"}},
                            {"term": {"ns": "test"}}
                        ],
                        "must": [
                            {"match": {"msg": "mindbox upload server error"}},
                            {"term": {"msg": "id-authorize-customer-topic"}}
                        ]
                    }
                }
            ),
            (
                "What are logs from 16:00:00 to now?",
                {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "time": {
                                        "gte": "now/d+16h",
                                        "lte": "now",
                                        "time_zone": "+03:00"
                                    }
                                }
                            }
                        ]
                    }
                }
            ),
            (
                "What are logs on prod from 16:35:11 to 16:36:56?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"ns": "prod"}},
                            {
                                "range": {
                                    "time": {
                                        "gte": "now/d+16h35m11s",
                                        "lte": "now/d+16h36m56s",
                                        "time_zone": "+03:00"
                                    }
                                }
                            }
                        ]
                    }
                }
            ),
            (
                "What are steps of item NM0098877?",
                {
                    "bool": {
                        "must": [
                            {"term": {"msg": "NM0098877"}}
                        ]
                    }
                }
            ),
            (
                "What happened with item NM0086817 on test?",
                {
                    "bool": {
                        "filter": [
                            {"term": {"ns": "test"}}
                        ],
                        "must": [
                            {"term": {"msg": "NM0086817"}}
                        ]
                    }
                }
            )
        ]
        
        # Create structured output chain
        structured_llm = self.llm.with_structured_output(OpenSearchQuery)
        self.chain = PromptTemplate(
            input_variables=["examples", "question"],
            template=self.base_template
        ) | structured_llm
    
    def _format_examples(self) -> str:
        """Format examples for the prompt template."""
        formatted_examples = []
        for question, query in self.examples:
            # Convert the query dict to a nicely formatted JSON string
            query_str = json.dumps(query, indent=2)
            formatted_examples.append(f"User: {question}\nOpenSearch query:\n{query_str}\n")
        return "\n".join(formatted_examples)
    
    def construct_query(self, question: str) -> Dict[str, Any]:
        """Construct an OpenSearch query from a natural language question."""
        # Format the prompt with examples
        result = self.chain.invoke({
            "examples": self._format_examples(),
            "question": question
        })
        
        # Convert the Pydantic model to a dict
        return result.model_dump()

    async def aconstruct_query(self, question: str) -> Dict[str, Any]:
        """Construct an OpenSearch query from a natural language question asynchronously."""
        # Format the prompt with examples
        result = await self.chain.ainvoke({
            "examples": self._format_examples(),
            "question": question
        })
        
        # Convert the Pydantic model to a dict
        return result.model_dump()

if __name__ == "__main__":
    constructor = OpenSearchQueryConstructor()
    
    test_queries = [
        "What were API service errors on April 11 2025?",
        "What are warnings in prod this week?",
        "What are warnings in prod this month?",
        "What are errors in test last hour?",
        "What happened with order PSV-745559?",
        "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
        "What are errors in prod today?",
        "What are crm errors in prod today?",
        "What are Mindbox upload server errors in topic id-authorize-customer-topic?",
        "What are errors in prod last hour?",
        "What are errors in prod last 20 hours?",
        "What is wrong with order PSV-745559?",
        "What is wrong with order PSV-737844-К0015742?",
        "What happened with item NM0086817 on test?",
        "What are steps of item NM0098877?",
        "What are errors in prod from 2025-03-20 to 2025-03-21?",
        "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
        "What happened with order PSV-745559 from 2025-03-20 10:00:00 to 2025-03-21 11:35:56?",
        "What are logs from 16:00:00 to now?",
        "What are logs on prod from 16:35:11 to 16:36:56?",
        "What are warnings in prod this month?",
        "What are errors in test last month?",
        "What are Mindbox upload errors in test this week?",
        "What are info messages in prod last week?",
    ]


    print("--------------------------------")
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("Generated OpenSearch query:")
        try:
            result = constructor.construct_query(query)
            pprint(result, indent=2)
        except Exception as e:
            print(f"Error processing query: {e}") 
        print("--------------------------------")