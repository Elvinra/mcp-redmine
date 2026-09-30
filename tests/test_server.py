"""Tests for mcp_redmine.server. Run with: make tests-run"""
import os

# Server module reads config from environment at import time
os.environ.setdefault("REDMINE_URL", "https://redmine.example.org")
os.environ.setdefault("REDMINE_API_KEY", "test-api-key-123")

import pytest

from mcp_redmine import server


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, content=b""):
        self.status_code = status_code
        self._json_body = json_body
        self.content = content if not json_body else b"x"

    def raise_for_status(self):
        pass

    def json(self):
        if self._json_body is None:
            raise ValueError("no json")
        return self._json_body


@pytest.fixture
def capture_requests(monkeypatch):
    """Capture outgoing requests instead of hitting the network."""
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse(json_body={"ok": True})

    monkeypatch.setattr(server._http_client, "request", fake_request)
    return calls


# Security: model-controlled `path` must never escape REDMINE_URL (issue #44)

def test_normal_path_stays_on_redmine(capture_requests):
    result = server.request("issues.json")
    assert result["status_code"] == 200
    assert capture_requests[0]["url"] == "https://redmine.example.org/issues.json"


def test_leading_slash_path_stays_on_redmine(capture_requests):
    server.request("/issues.json")
    assert capture_requests[0]["url"] == "https://redmine.example.org/issues.json"


@pytest.mark.parametrize("path", [
    "https://attacker.example/collect",
    "http://attacker.example/collect",
    "https://redmine.example.org.attacker.example/collect",
    "ftp://attacker.example/collect",
])
def test_escaping_paths_are_refused(capture_requests, path):
    result = server.request(path)
    assert result["status_code"] == 0
    assert "escapes REDMINE_URL" in result["error"]
    assert capture_requests == []  # nothing left the building


@pytest.mark.parametrize("path", [
    "//attacker.example/collect",  # neutralized by lstrip('/')
    "../../../collect",  # dot segments resolved by urljoin, can't climb above the host
    "/../collect",
])
def test_tricky_paths_stay_on_redmine_host(capture_requests, path):
    server.request(path)
    assert len(capture_requests) == 1
    assert capture_requests[0]["url"].startswith("https://redmine.example.org/")


def test_api_key_only_sent_to_redmine(capture_requests):
    server.request("issues.json")
    assert capture_requests[0]["headers"]["X-Redmine-API-Key"] == "test-api-key-123"


# Read-only mode (REDMINE_READ_ONLY)

def test_read_only_blocks_writes(capture_requests, monkeypatch):
    monkeypatch.setattr(server, "REDMINE_READ_ONLY", True)
    for method in ["post", "put", "PATCH", "delete"]:
        result = server.request("issues.json", method=method)
        assert result["status_code"] == 0
        assert "REDMINE_READ_ONLY" in result["error"]
    assert capture_requests == []


def test_read_only_allows_get(capture_requests, monkeypatch):
    monkeypatch.setattr(server, "REDMINE_READ_ONLY", True)
    result = server.request("issues.json", method="get")
    assert result["status_code"] == 200
    assert len(capture_requests) == 1


def test_writes_allowed_by_default(capture_requests):
    result = server.request("issues.json", method="post", data={"issue": {}})
    assert result["status_code"] == 200


# Tool plumbing

def test_redmine_request_tool_wraps_insecure_content(capture_requests):
    result = server.redmine_request("issues.json")
    assert "<insecure-content-" in result
    assert "status_code: 200" in result


def test_paths_list_returns_spec_paths():
    result = server.format_response(list(server.SPEC["paths"].keys()))
    assert "/issues.json" in result


def test_attachment_image_rejects_non_image(monkeypatch):
    def fake_request(path, method="get", **kwargs):
        return {"status_code": 200, "error": "",
                "body": {"attachment": {"content_type": "application/pdf", "filename": "a.pdf", "filesize": 10}}}

    monkeypatch.setattr(server, "request", fake_request)
    result = server.redmine_attachment_image(1)
    assert isinstance(result, str) and "not an image" in result


def test_attachment_image_rejects_oversize(monkeypatch):
    def fake_request(path, method="get", **kwargs):
        return {"status_code": 200, "error": "",
                "body": {"attachment": {"content_type": "image/png", "filename": "a.png",
                                        "filesize": server.ATTACHMENT_IMAGE_MAX_BYTES + 1}}}

    monkeypatch.setattr(server, "request", fake_request)
    result = server.redmine_attachment_image(1)
    assert isinstance(result, str) and "too large" in result


def test_attachment_image_returns_image(monkeypatch):
    png_bytes = b"\x89PNG\r\n\x1a\nfakepngdata"

    def fake_request(path, method="get", **kwargs):
        if path.endswith(".json"):
            return {"status_code": 200, "error": "",
                    "body": {"attachment": {"content_type": "image/png", "filename": "a.png", "filesize": 20}}}
        return {"status_code": 200, "error": "", "body": png_bytes}

    monkeypatch.setattr(server, "request", fake_request)
    result = server.redmine_attachment_image(1)
    assert isinstance(result, server.Image)


# Raw downloads (#46): attachments whose content is valid JSON must be saved byte-exact

JSON_BYTES = b'{"info": {"name": "collection"}, "item": []}'


def test_request_raw_returns_bytes_for_json_content(monkeypatch):
    def fake_request(method, url, **kwargs):
        response = FakeResponse(json_body={"info": {"name": "collection"}, "item": []})
        response.content = JSON_BYTES
        return response

    monkeypatch.setattr(server._http_client, "request", fake_request)
    assert server.request("attachments/download/1/collection.json", raw=True)["body"] == JSON_BYTES
    # without raw, the same response is parsed (existing behaviour for API calls)
    assert isinstance(server.request("some.json")["body"], dict)


def test_redmine_download_saves_json_attachment(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "REDMINE_ALLOWED_DIRECTORIES", [tmp_path])

    def fake_request(path, method="get", **kwargs):
        if path.endswith(".json") and not kwargs.get("raw"):
            return {"status_code": 200, "error": "",
                    "body": {"attachment": {"filename": "collection.json"}}}
        assert kwargs.get("raw") is True
        return {"status_code": 200, "error": "", "body": JSON_BYTES}

    monkeypatch.setattr(server, "request", fake_request)
    result = server.redmine_download(1, str(tmp_path / "out.json"))
    assert "saved_to" in result and "error: ''" in result
    assert (tmp_path / "out.json").read_bytes() == JSON_BYTES


# Per-user API keys: the key comes from the calling user's request headers on the HTTP transports,
# with REDMINE_API_KEY as the fallback, so one deployment can serve a whole org.

class FakeContext:
    """Stands in for mcp Context. A plain dict proves the lookup does its own case folding."""

    def __init__(self, headers=None):
        self.headers = headers


def test_no_context_falls_back_to_env_key():
    assert server.resolve_api_key() == ("test-api-key-123", "")


def test_stdio_context_without_headers_falls_back_to_env_key():
    assert server.resolve_api_key(FakeContext(None)) == ("test-api-key-123", "")


def test_header_key_overrides_env_key():
    assert server.resolve_api_key(FakeContext({"X-Redmine-API-Key": "alice-key"})) == ("alice-key", "")


def test_header_lookup_is_case_insensitive():
    assert server.resolve_api_key(FakeContext({"x-redmine-api-key": "alice-key"})) == ("alice-key", "")


@pytest.mark.parametrize("value", ["alice-key", "Bearer alice-key", "bearer alice-key", "  Bearer  alice-key  "])
def test_authorization_header_with_and_without_bearer(value):
    assert server.resolve_api_key(FakeContext({"Authorization": value})) == ("alice-key", "")


def test_dedicated_header_wins_over_authorization():
    """REDMINE_API_KEY_HEADERS order decides: the Redmine-specific header beats a generic OAuth token."""
    ctx = FakeContext({"Authorization": "Bearer oauth-token", "X-Redmine-API-Key": "alice-key"})
    assert server.resolve_api_key(ctx) == ("alice-key", "")


def test_custom_header_name_is_honoured(monkeypatch):
    monkeypatch.setattr(server, "REDMINE_API_KEY_HEADERS", ["X-Api-Key"])
    assert server.resolve_api_key(FakeContext({"X-Api-Key": "alice-key"})) == ("alice-key", "")
    # A header not on the list is ignored, so the fallback applies.
    assert server.resolve_api_key(FakeContext({"X-Redmine-API-Key": "alice-key"})) == ("test-api-key-123", "")


def test_empty_header_falls_back_to_env_key():
    assert server.resolve_api_key(FakeContext({"X-Redmine-API-Key": "   "})) == ("test-api-key-123", "")


@pytest.mark.parametrize("value", ["bad\nX-Evil: 1", "two words", "tab\tkey"])
def test_malformed_header_key_is_refused_not_forwarded(value):
    """A client-controlled value must never be able to inject into the outgoing header."""
    key, error = server.resolve_api_key(FakeContext({"X-Redmine-API-Key": value}))
    assert key == ""
    assert "Malformed API key" in error


def test_require_user_api_key_disables_the_env_fallback(monkeypatch):
    monkeypatch.setattr(server, "REDMINE_REQUIRE_USER_API_KEY", True)
    key, error = server.resolve_api_key(FakeContext(None))
    assert key == ""
    assert "REDMINE_REQUIRE_USER_API_KEY" in error
    # A user who does send their own key is unaffected.
    assert server.resolve_api_key(FakeContext({"X-Redmine-API-Key": "alice-key"})) == ("alice-key", "")


def test_no_key_anywhere_is_an_error(monkeypatch):
    monkeypatch.setattr(server, "REDMINE_API_KEY", "")
    key, error = server.resolve_api_key(FakeContext(None))
    assert key == ""
    assert "No Redmine API key available" in error


def test_request_refuses_to_call_redmine_without_a_key(capture_requests, monkeypatch):
    monkeypatch.setattr(server, "REDMINE_API_KEY", "")
    result = server.request("issues.json")
    assert result["status_code"] == 0
    assert "No Redmine API key available" in result["error"]
    assert capture_requests == []


def test_request_sends_the_supplied_key(capture_requests):
    server.request("issues.json", api_key="alice-key")
    assert capture_requests[0]["headers"]["X-Redmine-API-Key"] == "alice-key"


# Every key-using tool resolves from its own context and threads the key into each outgoing call.

def test_tools_use_the_callers_key(capture_requests, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "REDMINE_ALLOWED_DIRECTORIES", [tmp_path])
    ctx = FakeContext({"X-Redmine-API-Key": "alice-key"})

    upload_me = tmp_path / "upload.txt"
    upload_me.write_text("hello")

    server.redmine_request("issues.json", ctx=ctx)
    server.redmine_upload(str(upload_me), ctx=ctx)
    server.redmine_download(1, str(tmp_path / "out.bin"), filename="a.bin", ctx=ctx)
    server.redmine_attachment_image(1, ctx=ctx)

    assert capture_requests, "expected outgoing calls"
    assert {call["headers"]["X-Redmine-API-Key"] for call in capture_requests} == {"alice-key"}


def test_tools_report_a_key_error_without_calling_redmine(capture_requests, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "REDMINE_ALLOWED_DIRECTORIES", [tmp_path])
    monkeypatch.setattr(server, "REDMINE_REQUIRE_USER_API_KEY", True)
    ctx = FakeContext(None)

    assert "REDMINE_REQUIRE_USER_API_KEY" in server.redmine_request("issues.json", ctx=ctx)
    assert "REDMINE_REQUIRE_USER_API_KEY" in server.redmine_upload(str(tmp_path / "nope.txt"), ctx=ctx)
    assert "REDMINE_REQUIRE_USER_API_KEY" in server.redmine_download(1, str(tmp_path / "out.bin"), ctx=ctx)
    assert "REDMINE_REQUIRE_USER_API_KEY" in server.redmine_attachment_image(1, ctx=ctx)
    assert capture_requests == []


def test_download_attachment_lookup_also_uses_the_callers_key(monkeypatch, tmp_path):
    """The metadata GET and the byte download must both carry the caller's key, not the fallback."""
    monkeypatch.setattr(server, "REDMINE_ALLOWED_DIRECTORIES", [tmp_path])
    keys_sent = []

    def fake_request(method, url, **kwargs):
        keys_sent.append(kwargs["headers"]["X-Redmine-API-Key"])
        if url.endswith("attachments/1.json"):
            return FakeResponse(json_body={"attachment": {"filename": "a.bin"}})
        return FakeResponse(content=b"bytes")

    monkeypatch.setattr(server._http_client, "request", fake_request)
    result = server.redmine_download(1, str(tmp_path / "out.bin"),
                                     ctx=FakeContext({"Authorization": "Bearer alice-key"}))

    assert "saved_to" in result
    assert keys_sent == ["alice-key", "alice-key"]  # metadata lookup, then the download


# End-to-end over the real streamable-http transport: this is the deployment the per-user key exists for,
# so assert the whole path (client headers -> MCPServer Context -> outgoing X-Redmine-API-Key), not just
# resolve_api_key in isolation. Driven with anyio.run so no async pytest plugin is needed.

def test_streamable_http_sends_each_callers_own_key(monkeypatch):
    import anyio
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    keys_sent = []

    def fake_request(method, url, **kwargs):
        keys_sent.append(kwargs["headers"]["X-Redmine-API-Key"])
        return FakeResponse(json_body={"ok": True})

    monkeypatch.setattr(server._http_client, "request", fake_request)

    # json_response keeps the exchange to plain JSON; ASGITransport needs no socket. The Host header must
    # look like a real local server or the transport's DNS-rebinding protection rejects the request.
    app = server.mcp.streamable_http_app(json_response=True)
    base_url = "http://127.0.0.1:8000"

    async def call_tool_with_headers(headers):
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app), base_url=base_url,
                                      headers=headers) as http_client:
            async with streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool("redmine_request", {"path": "issues.json"})

    async def three_callers():
        async with app.router.lifespan_context(app):
            alice = await call_tool_with_headers({"X-Redmine-API-Key": "alice-key"})
            await call_tool_with_headers({"Authorization": "Bearer bob-key"})
            await call_tool_with_headers({})  # a client that sets no header at all
            return alice

    alice = anyio.run(three_callers)

    assert "status_code: 200" in alice.content[0].text
    # Each user's own key, and only the header-less caller falls back to REDMINE_API_KEY.
    assert keys_sent == ["alice-key", "bob-key", "test-api-key-123"]


def test_request_never_borrows_the_env_key_in_strict_mode(capture_requests, monkeypatch):
    """Defense in depth: even a call that skipped resolve_api_key must not fall back to the operator's key."""
    monkeypatch.setattr(server, "REDMINE_REQUIRE_USER_API_KEY", True)
    result = server.request("issues.json")
    assert result["status_code"] == 0
    assert "REDMINE_REQUIRE_USER_API_KEY" in result["error"]
    assert capture_requests == []
    # An explicit per-user key still goes through.
    assert server.request("issues.json", api_key="alice-key")["status_code"] == 200
    assert capture_requests[0]["headers"]["X-Redmine-API-Key"] == "alice-key"
