from langchain_community.query_constructors.opensearch import OpenSearchTranslator
from typing import Dict, Tuple
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    StructuredQuery
)

class CustomOpenSearchTranslator(OpenSearchTranslator):
    def visit_comparison(self, comparison: Comparison) -> Dict:
        # Skip if value is NO_FILTER
        if comparison.value == "NO_FILTER":
            return None

        if comparison.comparator in [
            Comparator.LT,
            Comparator.LTE,
            Comparator.GT,
            Comparator.GTE,
        ]:
            return {
                "range": {
                    comparison.attribute: {
                        self._format_func(comparison.comparator): comparison.value
                    }
                }
            }

        return {self._format_func(comparison.comparator): {comparison.attribute: comparison.value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:  # Changed return type to match base class
        """Convert structured query to OpenSearch query format."""
        query_dict = {"bool": {}}
        
        # Get the query text
        query_text = structured_query.query if structured_query.query != "NO_FILTER" else ""
        
        # Add text query if present
        if query_text:
            query_dict["bool"]["must"] = {"match": {"msg": query_text}}

        # Add filters if present
        if structured_query.filter:
            filter_dict = structured_query.filter.accept(self)
            if filter_dict:
                if "bool" in filter_dict and "must" in filter_dict["bool"]:
                    # If we have a list of filters, add them directly
                    query_dict["bool"]["filter"] = filter_dict["bool"]["must"]
                else:
                    # If we have a single filter, wrap it in a list
                    query_dict["bool"]["filter"] = [filter_dict]

        # If we have an empty bool query, use match_all
        if not query_dict["bool"]:
            return query_text, {"match_all": {}}
            
        return query_text, query_dict  