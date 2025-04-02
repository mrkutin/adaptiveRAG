from opensearchpy import OpenSearch
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import (
    ElasticSearchBM25Retriever,
)
# import elasticsearch

OPENSEARCH_HOST='opensearch-data.prod.prosv.yc'
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME='bus-admin'
OPENSEARCH_PASSWORD='WZJ7WKimoLFWCzV'
OPENSEARCH_INDEX='bus-prod-info-*'
OPENSEARCH_USE_SSL=True
OPENSEARCH_VERIFY_CERTS=False
OPENSEARCH_QUERY_SIZE=10



client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_USERNAME else None,
    use_ssl=OPENSEARCH_USE_SSL,
    verify_certs=OPENSEARCH_VERIFY_CERTS,
    ssl_show_warn=False
)

result = client.search(index="bus-prod-info-*", body={
    "min_score": 10.0,
    "size": 501,
    "query": {
        "bool": {
            "must": {
                "match": {
                    "msg": "mindbox upload error"
                },
            },

            "filter": [
                {'term': {'level': 'error'}},
                {'term': {'ns': 'prod'}},
                {'range': {'time': {'gte': 'now/d'}}}
            ]
        }
    }
})



# result = client.search(index="bus-prod-info-*", body={
#     "size": 1000,
#     "query": {
#         "bool": {
#             "should": [
#                 # { "match": { "msg": "mindbox" } },
#                 { "match": { "msg": "uploaded" } },
#                 { "match": { "msg": "error" } }
#             ],
#             "minimum_should_match": 2
#         }
#       }
#     })



print(f"result length: {len(result['hits']['hits'])}")

""" Came from the opensearch translator
('mindbox upload error',
{'filter': {'bool': {'must': [{'term': {'metadata.level.keyword': 'error'}},
    {'term': {'metadata.ns.keyword': 'test'}},
    {'term': {'metadata.svc.keyword': 'mindbox'}},
    {'range': {'metadata.time': {'gte': 'now/w'}}}]}}})
"""
