from typing import Dict, Union
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

class CustomOpenSearchTranslator(Visitor):
    """Translate structured queries to OpenSearch format without field name modifications."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
    ]
    """Subset of allowed logical comparators."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        comp_operator_map = {
            Comparator.EQ: "term",
            Comparator.LT: "lt",
            Comparator.LTE: "lte",
            Comparator.GT: "gt",
            Comparator.GTE: "gte",
            Operator.AND: "must",
            Operator.OR: "should",
            Operator.NOT: "must_not",
        }
        return comp_operator_map[func]

    def visit_operation(self, operation: Operation) -> Dict:
        """Convert operation (AND/OR/NOT) to OpenSearch bool query."""
        args = [arg.accept(self) for arg in operation.arguments]
        # Filter out None values (skipped filters)
        args = [arg for arg in args if arg is not None]
        
        if not args:
            return None
            
        return {"bool": {self._format_func(operation.operator): args}}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        """Convert comparison to OpenSearch query term."""
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

    def visit_structured_query(self, structured_query: StructuredQuery) -> Dict:
        """Convert structured query to OpenSearch query format."""
        query_dict = {"bool": {}}
        
        # Add text query if present and not NO_FILTER
        if structured_query.query and structured_query.query != "NO_FILTER":
            query_dict["bool"]["must"] = {"match": {"msg": structured_query.query}}

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

        # If we have an empty bool query, return just the parts we have
        if not query_dict["bool"]:
            return {"match_all": {}}
            
        return query_dict 