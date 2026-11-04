from fastapi.testclient import TestClient

from app.checker import _short_text
from app.main import app


def test_short_text_keeps_number():
    assert _short_text("UAE NATIONAL DAY 55") == "UAE 55"
    assert _short_text("Welcome home") == "WELCO"
    assert _short_text("GO") == "GO"


def test_auth_disabled(monkeypatch):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    c = TestClient(app)
    assert c.get("/api/presets").status_code == 200
    assert c.get("/studio").status_code == 200


def test_auth_enabled(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "pw123")
    c = TestClient(app, follow_redirects=False)
    assert c.get("/").status_code == 200
    assert c.get("/health").status_code == 200
    assert c.get("/api/presets").status_code == 401
    assert c.get("/assets/x/y.png").status_code == 401
    r = c.get("/studio")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    r = c.post("/login", data={"username": "judge", "password": "bad"})
    assert r.headers["location"] == "/login?error=1"
    r = c.post("/login", data={"username": "nobody", "password": "pw123"})
    assert r.headers["location"] == "/login?error=1"
    r = c.post("/login", data={"username": "JUDGE", "password": "pw123"})
    assert r.status_code == 303 and r.headers["location"] == "/studio"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert c.get("/api/presets").status_code == 200
    assert c.get("/studio").status_code == 200
    assert c.get("/login").status_code == 303
    c.post("/logout")
    assert c.get("/api/presets").status_code == 401


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "pw123")
    from app import main
    main._FAILS.clear()
    c = TestClient(app, follow_redirects=False)
    for _ in range(10):
        c.post("/login", data={"username": "judge", "password": "x"})
    assert c.post("/login", data={"username": "judge", "password": "pw123"}).status_code == 429
    main._FAILS.clear()
