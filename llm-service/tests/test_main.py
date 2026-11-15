from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app, get_client


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_retry_then_success() -> None:
    fake = FakeClient(["this is not a JSON", '{"summary":"DB timeout","severity":"high","service":"payment"}'])
    app.dependency_overrides[get_client] = lambda: fake
    r = TestClient(app).post("/summarize", json={"log": "ERROR db timeout"})
    assert r.status_code == 200
    assert r.json()["severity"] == "high"
    assert fake.calls == 2

def test_excersise_ok_first() -> None:
    fake = FakeClient(['{"summary":"DB timeout","severity":"high","service":"payment"}'])
    app.dependency_overrides[get_client] = lambda: fake
    r = TestClient(app).post("/summarize", json={"log": "ERROR db timeout"})
    assert r.status_code == 200
    assert r.json()["severity"] == "high"
    assert fake.calls == 1


def test_excersise_bad_three_times() -> None:
    fake = FakeClient(["this is not a JSON","this is not a JSON","this is not a JSON"])
    app.dependency_overrides[get_client] = lambda: fake
    r = TestClient(app).post("/summarize", json={"log": "ERROR db timeout"})
    assert r.status_code == 502
    assert fake.calls == 3
