from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_ollama import OllamaLLM


from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.opensearch import OpenSearchTranslator
from pprint import pformat

document_contents = "Log storage containing system events and error messages. Use only attributes 'time', 'level', 'ns', 'svc' in filter."


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
     AttributeInfo(
        name="svc",
        description="Service name, mindbox, esb, etc.",
        type="string",
    ),
]

query_model = OllamaLLM(
    base_url="http://127.0.0.1:11434",
    model="qwen2.5-coder",
    temperature=0,
    timeout=30,
    max_tokens=8192,
)

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
            "query": "NO_FILTER",
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
            "query": "NO_FILTER",
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


constructor_prompt = get_query_constructor_prompt(
    document_contents,
    attribute_info,
    examples=examples,
)

# print(prompt.format(query="What are steps of item FG987867456879?"))

# # Instead of trying to print the formatted prompt, let's just use it
output_parser = StructuredQueryOutputParser.from_components()
translator = OpenSearchTranslator()

query_constructor = constructor_prompt | query_model | output_parser 


# tests = [
#     "What are errors in prod today?",
#     "What are Mindbox upload server errors in topic id-authorize-customer-topic?",
#     "What are errors in prod last hour?",
#     "What are errors in prod last 20 hours?",
#     "What is wrong with order PSV-745559?",
#     "What is wrong with order PSV-737844-К0015742?",
#     "What happened with item NM0086817 on test?",
#     "What are steps of item NM0098877?",
#     "What are errors in prod from 2025-03-20 to 2025-03-21?",
#     "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?",
#     "What happened with order PSV-745559 from 2025-03-20 10:00:00 to 2025-03-21 11:35:56?",
#     "What are logs from 16:00:00 to now?",
#     "What are logs on prod from 16:35:11 to 16:36:56?",
#     "What are warnings in prod this month?",
#     "What are errors in test last month?",
#     "What are Mindbox upload errors in test this week?",
#     "What are info messages in prod last week?",
# ]


# for test in tests:
#     print(f"Test: {test} ================================")
#     structured_query = query_constructor.invoke({"query": test})
#     print(structured_query)


structured_query = query_constructor.invoke({"query": "What are Mindbox upload errors in test this week?"})
# print(structured_query)

translator = OpenSearchTranslator()
open_search_query = translator.visit_structured_query(structured_query)
print(pformat(open_search_query))




# retriever = SelfQueryRetriever(
#             query_constructor=query_constructor,
#             vectorstore=vectorstore,
#             structured_query_translator=OpenSearchTranslator(),
#             search_kwargs={'k': 10}
        # )

# retriever.invoke(
#     "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
# )