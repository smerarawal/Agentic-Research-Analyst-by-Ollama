from workflow import (
    _allow_by_robots,
    _cached_page,
    _index_sources,
    _route_clarify,
    _route_critique,
    _route_verify,
    check_report,
    chunk_text,
    verify_quote,
)


def test_quote_check_normalizes_whitespace():
    assert verify_quote("the  quick\n brown fox", "The quick brown fox jumps")
    assert not verify_quote("quick fox", "The quick brown fox jumps")


def test_chunker_uses_overlap():
    chunks = chunk_text("a b c d e f", size=4, overlap=1)
    assert chunks == ["a b c d", "d e f"]


def test_chunker_rejects_invalid_overlap():
    try:
        chunk_text("words", size=3, overlap=3)
    except ValueError:
        return
    raise AssertionError("expected invalid chunk settings to raise")


def test_clarification_router():
    assert _route_clarify({"specific": True}) == "plan"
    assert _route_clarify({"specific": False}) == "end"


def test_critique_router_caps_search_rounds():
    assert _route_critique({"gaps": [{"search_query": "more"}], "iteration": 1}) == "search"
    assert _route_critique({"gaps": [{"search_query": "more"}], "iteration": 2}) == "write"
    assert _route_critique({"gaps": [], "iteration": 1}) == "write"


def test_verify_router_caps_writer_retry():
    assert _route_verify({"check": {"passed": False}, "attempts": 1}) == "retry"
    assert _route_verify({"check": {"passed": False}, "attempts": 2}) == "end"
    assert _route_verify({"check": {"passed": True}, "attempts": 1}) == "end"


def test_report_check_flags_invalid_citations_and_uncited_claims():
    report = "\n".join(f"## {heading}" for heading in [
        "Executive Summary", "Core Strategy", "Technical Analysis",
        "Competitive Landscape", "Risks and Challenges", "Future Outlook",
    ]) + "\nThis sentence contains an unsupported claim without a source."
    result = check_report(report, {})
    assert not result["passed"]
    assert result["uncited_claim_lines"]


def test_robots_check_uses_declared_user_agent(monkeypatch):
    import requests

    seen = {}

    class Response:
        status_code = 200
        text = "User-agent: *\nDisallow: /private"

        def raise_for_status(self):
            pass

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)
    assert _allow_by_robots("https://example.com/public")
    assert not _allow_by_robots("https://example.com/private/report")
    assert seen["headers"]["User-Agent"]


def test_page_timeout_returns_empty_for_snippet_fallback(monkeypatch, tmp_path):
    import requests
    import workflow

    monkeypatch.setattr(workflow, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(workflow, "_allow_by_robots", lambda _url: True)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ReadTimeout("slow page")))
    assert _cached_page("https://example.com/slow") == ""


def test_indexing_batches_documents(monkeypatch):
    import workflow

    class Collection:
        def __init__(self):
            self.batch_sizes = []

        def add(self, documents, ids, metadatas):
            self.batch_sizes.append(len(documents))
            assert len(documents) == len(ids) == len(metadatas)

    collection = Collection()
    monkeypatch.setattr(workflow, "_get_collection", lambda _name: (None, collection))
    text = " ".join(f"word{i}" for i in range(3000))
    _index_sources("test-run", {1: {"url": "https://example.com", "title": "Example", "text": text}})
    assert len(collection.batch_sizes) > 1
    assert max(collection.batch_sizes) <= 8

