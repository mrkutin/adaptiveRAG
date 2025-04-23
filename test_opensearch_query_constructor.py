# USAGE: pytest test_opensearch_query_constructor.py

import pytest
from opensearch_query_constructor import OpenSearchQueryConstructor

@pytest.fixture
def constructor():
    return OpenSearchQueryConstructor()

def test_api_service_errors_specific_date(constructor):
    query = "What were API service errors on April 11 2025?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {
                    "range": {
                        "time": {
                            "gte": "2025-04-11T00:00:00",
                            "lte": "2025-04-11T23:59:59",
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
    
    print("\nTest: API service errors specific date")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any("time_zone" in tr and tr["time_zone"] == "+03:00" for tr in time_ranges)
    assert any(tr.get("gte") == "2025-04-11T00:00:00" for tr in time_ranges)
    assert any(tr.get("lte") == "2025-04-11T23:59:59" for tr in time_ranges)
    
    musts = result["bool"]["must"]
    assert any(m.get("match", {}).get("msg") == "API service error" for m in musts)

def test_warnings_in_prod_this_week(constructor):
    query = "What are warnings in prod this week?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "warn"}},
                {"term": {"ns": "prod"}},
                {"range": {"time": {"gte": "now/w"}}}
            ]
        }
    }
    
    print("\nTest: Warnings in prod this week")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "warn" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now/w" for tr in time_ranges)

def test_warnings_in_prod_this_month(constructor):
    query = "What are warnings in prod this month?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "warn"}},
                {"term": {"ns": "prod"}},
                {"range": {"time": {"gte": "now-1M"}}}
            ]
        }
    }
    
    print("\nTest: Warnings in prod this month")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "warn" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now-1M" for tr in time_ranges)

def test_errors_in_test_last_hour(constructor):
    query = "What are errors in test last hour?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "test"}},
                {"range": {"time": {"gte": "now-1h"}}}
            ]
        }
    }
    
    print("\nTest: Errors in test last hour")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "test" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now-1h" for tr in time_ranges)

def test_errors_in_test_last_month(constructor):
    query = "What are errors in test last month?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "test"}},
                {"range": {"time": {"gte": "now-1M"}}}
            ]
        }
    }
    
    print("\nTest: Errors in test last month")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "test" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now-1M" for tr in time_ranges)

def test_order_psv_745559(constructor):
    query = "What happened with order PSV-745559?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "must": [
                {"term": {"msg": "PSV-745559"}}
            ]
        }
    }
    
    print("\nTest: Order PSV-745559")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "must" in result["bool"]
    
    musts = result["bool"]["must"]
    assert any(m.get("term", {}).get("msg") == "PSV-745559" for m in musts)

def test_mindbox_upload_errors_date_range(constructor):
    query = "What are Mindbox upload errors in test from 2025-03-20 10:00:00 to 2025-03-21 10:00:00?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "test"}},
                {
                    "range": {
                        "time": {
                            "gte": "2025-03-20T10:00:00",
                            "lte": "2025-03-21T10:00:00",
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
    
    print("\nTest: Mindbox upload errors date range")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "test" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any("time_zone" in tr and tr["time_zone"] == "+03:00" for tr in time_ranges)
    assert any(tr.get("gte") == "2025-03-20T10:00:00" for tr in time_ranges)
    assert any(tr.get("lte") == "2025-03-21T10:00:00" for tr in time_ranges)
    
    musts = result["bool"]["must"]
    assert any("mindbox" in m.get("match", {}).get("msg", "").lower() for m in musts)

def test_errors_in_prod_today(constructor):
    query = "What are errors in prod today?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "prod"}},
                {"range": {"time": {"gte": "now/d"}}}
            ]
        }
    }
    
    print("\nTest: Errors in prod today")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now/d" for tr in time_ranges)

def test_crm_errors_in_prod_today(constructor):
    query = "What are crm errors in prod today?"
    result = constructor.construct_query(query)
    
    expected = {
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
    
    print("\nTest: CRM errors in prod today")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now/d" for tr in time_ranges)
    
    musts = result["bool"]["must"]
    assert any(m.get("match", {}).get("msg") == "crm error" for m in musts)

def test_mindbox_upload_server_errors(constructor):    
    query = "What are Mindbox upload server errors in topic id-authorize-customer-topic?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
            ],
            "must": [
                {"match": {"msg": "mindbox upload server error"}},
                {"term": {"msg": "id-authorize-customer-topic"}}
            ]
        }
    }
    
    print("\nTest: Mindbox upload server errors")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    
    musts = result["bool"]["must"]
    assert any("mindbox" in m.get("match", {}).get("msg", "").lower() for m in musts)
    assert any(m.get("term", {}).get("msg") == "id-authorize-customer-topic" for m in musts)

def test_errors_in_prod_last_hour(constructor):
    query = "What are errors in prod last hour?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "prod"}},
                {"range": {"time": {"gte": "now-1h"}}}
            ]
        }
    }
    
    print("\nTest: Errors in prod last hour")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now-1h" for tr in time_ranges)

def test_errors_in_prod_last_20_hours(constructor):
    query = "What are errors in prod last 20 hours?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "prod"}},
                {"range": {"time": {"gte": "now-20h"}}}
            ]
        }
    }
    
    print("\nTest: Errors in prod last 20 hours")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now-20h" for tr in time_ranges)

def test_order_psv_745559_wrong(constructor):
    query = "What is wrong with order PSV-745559?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "must": [
                {"term": {"msg": "PSV-745559"}}
            ]
        }
    }
    
    print("\nTest: Order PSV-745559 wrong")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "must" in result["bool"]
    
    musts = result["bool"]["must"]
    assert any(m.get("term", {}).get("msg") == "PSV-745559" for m in musts)

def test_order_psv_737844_wrong(constructor):
    query = "What is wrong with order PSV-737844-К0015742?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "must": [
                {"term": {"msg": "PSV-737844-К0015742"}}
            ]
        }
    }
    
    print("\nTest: Order PSV-737844 wrong")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "must" in result["bool"]
    
    musts = result["bool"]["must"]
    assert any(m.get("term", {}).get("msg") == "PSV-737844-К0015742" for m in musts)

def test_item_nm0086817(constructor):
    query = "What happened with item NM0086817 on test?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"ns": "test"}}
            ],
            "must": [
                {"term": {"msg": "NM0086817"}}
            ]
        }
    }
    
    print("\nTest: Item NM0086817")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("ns") == "test" for f in filters)
    
    musts = result["bool"]["must"]
    assert any(m.get("term", {}).get("msg") == "NM0086817" for m in musts)

def test_item_nm0098877_steps(constructor):
    query = "What are steps of item NM0098877?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "must": [
                {"term": {"msg": "NM0098877"}}
            ]
        }
    }
    
    print("\nTest: Item NM0098877 steps")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "must" in result["bool"]
    
    musts = result["bool"]["must"]
    assert any(m.get("term", {}).get("msg") == "NM0098877" for m in musts)

def test_errors_in_prod_date_range(constructor):
    query = "What are errors in prod from 2025-03-20 to 2025-03-21?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "prod"}},
                {
                    "range": {
                        "time": {
                            "gte": "2025-03-20T00:00:00",
                            "lte": "2025-03-21T23:59:59",
                            "time_zone": "+03:00"
                        }
                    }
                }
            ]
        }
    }
    
    print("\nTest: Errors in prod date range")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any("time_zone" in tr and tr["time_zone"] == "+03:00" for tr in time_ranges)
    assert any(tr.get("gte") == "2025-03-20T00:00:00" for tr in time_ranges)
    assert any(tr.get("lte") == "2025-03-21T23:59:59" for tr in time_ranges)

def test_mindbox_upload_errors_this_week(constructor):
    query = "What are Mindbox upload errors in test this week?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "error"}},
                {"term": {"ns": "test"}},
                {"range": {"time": {"gte": "now/w"}}}
            ],
            "must": [
                {"match": {"msg": "mindbox upload error"}}
            ]
        }
    }
    
    print("\nTest: Mindbox upload errors this week")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "error" for f in filters)
    assert any(f.get("term", {}).get("ns") == "test" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now/w" for tr in time_ranges)
    
    musts = result["bool"]["must"]
    assert any("mindbox" in m.get("match", {}).get("msg", "").lower() for m in musts)

def test_order_psv_745559_time_range(constructor):
    query = "What happened with order PSV-745559 from 2025-03-20 10:00:00 to 2025-03-21 11:35:56?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {
                    "range": {
                        "time": {
                            "gte": "2025-03-20T10:00:00",
                            "lte": "2025-03-21T11:35:56",
                            "time_zone": "+03:00"
                        }
                    }
                }
            ],
            "must": [
                {"term": {"msg": "PSV-745559"}}
            ]
        }
    }
    
    print("\nTest: Order PSV-745559 time range")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    assert "must" in result["bool"]
    
    filters = result["bool"]["filter"]
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any("time_zone" in tr and tr["time_zone"] == "+03:00" for tr in time_ranges)
    assert any(tr.get("gte") == "2025-03-20T10:00:00" for tr in time_ranges)
    assert any(tr.get("lte") == "2025-03-21T11:35:56" for tr in time_ranges)
    
    musts = result["bool"]["must"]
    assert any(m.get("term", {}).get("msg") == "PSV-745559" for m in musts)

def test_logs_from_16_to_now(constructor):
    query = "What are logs from 16:00:00 to now?"
    result = constructor.construct_query(query)
    
    expected = {
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
    
    print("\nTest: Logs from 16:00 to now")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now/d+16h" for tr in time_ranges)
    assert any(tr.get("lte") == "now" for tr in time_ranges)
    assert any("time_zone" in tr and tr["time_zone"] == "+03:00" for tr in time_ranges)

def test_logs_on_prod_time_range(constructor):
    query = "What are logs on prod from 16:35:11 to 16:36:56?"
    result = constructor.construct_query(query)
    
    expected = {
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
    
    print("\nTest: Logs on prod time range")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now/d+16h35m11s" for tr in time_ranges)
    assert any(tr.get("lte") == "now/d+16h36m56s" for tr in time_ranges)
    assert any("time_zone" in tr and tr["time_zone"] == "+03:00" for tr in time_ranges)

def test_info_messages_in_prod_last_week(constructor):
    query = "What are info messages in prod last week?"
    result = constructor.construct_query(query)
    
    expected = {
        "bool": {
            "filter": [
                {"term": {"level": "info"}},
                {"term": {"ns": "prod"}},
                {
                    "range": {
                        "time": {
                            "gte": "now-1w"
                        }
                    }
                }
            ]
        }
    }
    
    print("\nTest: Info messages in prod last week")
    print("Actual output:")
    print(result)
    print("\nExpected output:")
    print(expected)
    
    assert "bool" in result
    assert "filter" in result["bool"]
    
    filters = result["bool"]["filter"]
    assert any(f.get("term", {}).get("level") == "info" for f in filters)
    assert any(f.get("term", {}).get("ns") == "prod" for f in filters)
    
    time_ranges = [f["range"]["time"] for f in filters if "range" in f and "time" in f["range"]]
    assert any(tr.get("gte") == "now-1w" for tr in time_ranges)