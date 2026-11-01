"""
Tests for backend run metrics.

Every /api/generate run records how long each frame took, which frames
failed and why, and what the run cost. Metrics stay on the backend:
the public response carries none of them, and they are read through
token-protected admin endpoints and an unlinked admin page.
"""

import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from main import app

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
ALL_KEYS = ["GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"]
XAI_OK = {"data": [{"b64_json": "cmVzdWx0"}]}  # base64("result")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from metrics import RunMetricsStore
    s = RunMetricsStore(tmp_path / "runs.jsonl")
    monkeypatch.setattr(main, "metrics_store", s)
    return s


@pytest.fixture
def xai_key(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "secret-xai-999")


def _patch_post(monkeypatch, fail_every=0, error_text="rate limited"):
    """Fake provider: optionally fail every Nth call."""
    calls = {"n": 0}

    async def fake_post(self, url, json=None, headers=None, **kw):
        calls["n"] += 1
        if fail_every and calls["n"] % fail_every == 0:
            raise httpx.HTTPError(error_text)
        return httpx.Response(200, json=XAI_OK, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


def _generate(client, model_id="grok-imagine-2.0"):
    return client.post(
        "/api/generate",
        files={"image": ("t.png", FAKE_PNG, "image/png")},
        data={"model_id": model_id},
    )


# ──────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────

class TestStore:

    def test_record_and_read_back(self, tmp_path):
        from metrics import RunMetricsStore
        s = RunMetricsStore(tmp_path / "runs.jsonl")
        s.record({"job_id": "a", "model_id": "m", "total_ms": 10})
        s.record({"job_id": "b", "model_id": "m", "total_ms": 20})
        assert [r["job_id"] for r in s.all()] == ["a", "b"]

    def test_survives_new_instance_on_same_file(self, tmp_path):
        from metrics import RunMetricsStore
        RunMetricsStore(tmp_path / "runs.jsonl").record({"job_id": "a", "model_id": "m"})
        assert len(RunMetricsStore(tmp_path / "runs.jsonl").all()) == 1

    def test_empty_store_reads_empty(self, tmp_path):
        from metrics import RunMetricsStore
        assert RunMetricsStore(tmp_path / "none.jsonl").all() == []


# ──────────────────────────────────────────────
# /api/generate records a run
# ──────────────────────────────────────────────

class TestGenerateRecordsRun:

    def test_one_run_recorded_per_generate(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch)
        assert _generate(client).status_code == 200
        runs = store.all()
        assert len(runs) == 1
        run = runs[0]
        assert run["model_id"] == "grok-imagine-2.0"
        assert run["provider"] == "xai"
        assert run["frames_requested"] == 13
        assert run["frames_ok"] == 13
        assert run["frames_failed"] == 0
        assert run["total_ms"] >= 0
        assert run["started_at"]

    def test_every_frame_timed(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch)
        _generate(client)
        frames = store.all()[0]["frames"]
        assert len(frames) == 13
        assert [f["hour"] for f in frames] == list(range(8, 21))
        for f in frames:
            assert f["ms"] >= 0
            assert f["status"] == "success"

    def test_failed_frames_counted_with_reason(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch, fail_every=2, error_text="429 Too Many Requests")
        _generate(client)
        run = store.all()[0]
        assert run["frames_failed"] == 6
        assert run["frames_ok"] == 7
        failed = [f for f in run["frames"] if f["status"] == "error"]
        assert len(failed) == 6
        assert all("429" in f["error"] for f in failed)

    def test_cost_counts_only_successful_frames(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch, fail_every=2)
        _generate(client)
        run = store.all()[0]
        info = main.registry.get_adapter("grok-imagine-2.0").model_info
        assert run["est_cost_usd"] == pytest.approx(7 * info.cost_per_image, abs=1e-4)

    def test_error_text_never_contains_api_key(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch, fail_every=1, error_text="bad key secret-xai-999")
        _generate(client)
        dumped = json.dumps(store.all())
        assert "secret-xai-999" not in dumped

    def test_error_text_is_bounded(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch, fail_every=1, error_text="x" * 5000)
        _generate(client)
        for f in store.all()[0]["frames"]:
            assert len(f["error"]) <= 300

    def test_run_is_logged_as_structured_line(self, client, store, xai_key, monkeypatch, caplog):
        _patch_post(monkeypatch)
        with caplog.at_level(logging.INFO):
            _generate(client)
        lines = [r.getMessage() for r in caplog.records if "run_metrics" in r.getMessage()]
        assert len(lines) == 1
        payload = json.loads(lines[0].split("run_metrics ", 1)[1])
        assert payload["frames_ok"] == 13


# ──────────────────────────────────────────────
# Metrics stay on the backend
# ──────────────────────────────────────────────

class TestPublicResponseHasNoMetrics:

    def test_generate_response_has_no_timing_or_cost(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch)
        body = _generate(client).json()
        for field in ("total_ms", "est_cost_usd", "frames"):
            assert field not in body
        for v in body["variants"]:
            assert "ms" not in v


# ──────────────────────────────────────────────
# Admin endpoints
# ──────────────────────────────────────────────

class TestAdminAccess:

    @pytest.mark.parametrize("path", ["/api/admin/runs", "/api/admin/summary"])
    def test_disabled_without_admin_token_env(self, client, store, monkeypatch, path):
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
        assert client.get(path, headers={"X-Admin-Token": "anything"}).status_code == 404

    @pytest.mark.parametrize("path", ["/api/admin/runs", "/api/admin/summary"])
    def test_wrong_or_missing_token_forbidden(self, client, store, monkeypatch, path):
        monkeypatch.setenv("ADMIN_TOKEN", "letmein")
        assert client.get(path).status_code == 403
        assert client.get(path, headers={"X-Admin-Token": "nope"}).status_code == 403

    def test_token_in_query_string_not_accepted(self, client, store, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "letmein")
        assert client.get("/api/admin/runs?token=letmein").status_code == 403

    def test_runs_listed_newest_first(self, client, store, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "letmein")
        store.record({"job_id": "old", "model_id": "m", "started_at": "2026-10-31T01:00:00Z"})
        store.record({"job_id": "new", "model_id": "m", "started_at": "2026-10-31T02:00:00Z"})
        resp = client.get("/api/admin/runs", headers={"X-Admin-Token": "letmein"})
        assert resp.status_code == 200
        assert [r["job_id"] for r in resp.json()] == ["new", "old"]


class TestAdminSummary:

    def _seed(self, store):
        def run(model, ms_list, statuses, cost):
            frames = [{"hour": 8 + i, "ms": ms, "status": st, "error": "" if st == "success" else "e"}
                      for i, (ms, st) in enumerate(zip(ms_list, statuses))]
            ok = sum(1 for s in statuses if s == "success")
            store.record({
                "job_id": f"{model}-{len(store.all())}", "model_id": model, "model_name": model,
                "provider": "p", "started_at": "2026-10-31T00:00:00Z",
                "total_ms": max(ms_list), "frames_requested": len(ms_list),
                "frames_ok": ok, "frames_failed": len(ms_list) - ok,
                "est_cost_usd": cost, "frames": frames,
            })
        run("a", [100, 200, 300, 400], ["success"] * 4, 0.4)
        run("a", [500, 600, 700, 800], ["success", "success", "error", "error"], 0.2)
        run("b", [50, 50, 50, 50], ["success"] * 4, 0.1)

    def test_summary_groups_by_model(self, client, store, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "letmein")
        self._seed(store)
        data = client.get("/api/admin/summary", headers={"X-Admin-Token": "letmein"}).json()
        by = {row["model_id"]: row for row in data}
        assert set(by) == {"a", "b"}
        a = by["a"]
        assert a["runs"] == 2
        assert a["frames_requested"] == 8
        assert a["frames_failed"] == 2
        assert a["failure_rate"] == pytest.approx(0.25)
        assert a["mean_run_ms"] == pytest.approx(600)
        # Timing covers successful frames only; a failed call is not a render time
        assert a["median_frame_ms"] == pytest.approx(350)
        assert a["mean_cost_usd"] == pytest.approx(0.3)

    def test_summary_p95_frame_ms(self, client, store, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "letmein")
        self._seed(store)
        data = client.get("/api/admin/summary", headers={"X-Admin-Token": "letmein"}).json()
        b = next(r for r in data if r["model_id"] == "b")
        assert b["p95_frame_ms"] == pytest.approx(50)


# ──────────────────────────────────────────────
# Admin page: unlinked, no baked-in token
# ──────────────────────────────────────────────

STATIC = Path(__file__).parent.parent / "static"


class TestAdminPage:

    def test_admin_page_served(self, client):
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_admin_page_sends_token_in_header(self):
        html = (STATIC / "admin.html").read_text()
        assert "X-Admin-Token" in html
        assert "?token=" not in html

    def test_main_app_does_not_link_to_admin(self):
        assert "/admin" not in (STATIC / "index.html").read_text()


# ──────────────────────────────────────────────
# Thumbnails: original plus one representative frame per run
# ──────────────────────────────────────────────

class TestRunThumbnail:

    def test_run_records_original_url(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch)
        job = _generate(client).json()["job_id"]
        run = store.all()[0]
        assert run["original_url"] == f"/api/images/{job}/original.png"

    def test_thumbnail_is_5pm_frame(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch)
        job = _generate(client).json()["job_id"]
        run = store.all()[0]
        assert run["thumb_hour"] == 17
        assert run["thumb_url"] == f"/api/images/{job}/slot_09_h17.png"

    def test_thumbnail_falls_back_to_nearest_successful_hour(self, client, store, xai_key, monkeypatch):
        # Every 2nd call fails: 9, 11, 13, 15, 17, 19 fail. 16 and 18 tie; earlier wins.
        _patch_post(monkeypatch, fail_every=2)
        _generate(client)
        run = store.all()[0]
        assert run["thumb_hour"] == 16
        assert run["thumb_url"].endswith("slot_08_h16.png")

    def test_no_thumbnail_when_every_frame_failed(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch, fail_every=1)
        _generate(client)
        run = store.all()[0]
        assert run["thumb_url"] is None
        assert run["thumb_hour"] is None

    def test_public_response_still_has_no_thumbnail_fields(self, client, store, xai_key, monkeypatch):
        _patch_post(monkeypatch)
        body = _generate(client).json()
        assert "thumb_url" not in body and "thumb_hour" not in body


class TestAdminPageThumbnails:

    def test_admin_page_renders_thumbnails(self):
        html = (STATIC / "admin.html").read_text()
        assert "thumb_url" in html
        assert "original_url" in html
        assert "createElement('img')" in html

    def test_admin_page_never_uses_inner_html(self):
        """Run data includes provider error text; render with textContent only."""
        assert "innerHTML" not in (STATIC / "admin.html").read_text()


class TestAdminPageSession:

    def _html(self):
        return (STATIC / "admin.html").read_text()

    def test_has_logout_button(self):
        assert 'id="logout"' in self._html()

    def test_logout_starts_hidden(self):
        html = self._html()
        tag = html[html.index('id="logout"') - 80: html.index('id="logout"') + 80]
        assert "hidden" in tag

    def test_login_hides_form_and_shows_logout(self):
        html = self._html()
        assert "function showSignedIn" in html
        body = html[html.index("function showSignedIn"):]
        body = body[: body.index("\n  }")]
        assert "auth" in body and "logout" in body

    def test_logout_forgets_token_and_clears_data(self):
        html = self._html()
        assert "function logOut" in html
        body = html[html.index("function logOut"):]
        body = body[: body.index("\n  }")]
        assert "sessionStorage.removeItem" in body
        assert "summary" in body and "runs" in body

    def test_rejected_token_is_forgotten(self):
        """A stale saved token must not keep auto-loading into a 403."""
        html = self._html()
        assert "Wrong token" in html
        assert html.count("sessionStorage.removeItem") >= 2


class TestAdminPageImageDates:

    def test_each_image_caption_carries_the_run_date(self):
        html = (STATIC / "admin.html").read_text()
        assert "function runDate" in html
        assert "toLocaleDateString" in html
        # both thumbnails are captioned with the date
        assert html.count("runDate(run.started_at)") >= 2
