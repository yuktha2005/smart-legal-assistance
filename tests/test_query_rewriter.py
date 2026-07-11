from src.query_rewriter import QueryRewriter


def test_rewriter_where_defined():
    r = QueryRewriter()
    out = r.rewrite("Where is it defined?", {"current_section": "376DA"})
    assert "section 376da" in out.rewritten_query.lower()

