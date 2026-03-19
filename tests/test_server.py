"""Tests for the ORS Server against the specification endpoints."""

import asyncio
import json
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from pydantic import BaseModel

from ors import Environment, Server, tool, ToolOutput, TextBlock, Split


class SubmitInput(BaseModel):
    answer: str


class GSM8KEnv(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            return [
                {"id": "0", "question": "What is 2+2?", "answer": "4"},
                {"id": "1", "question": "What is 3*3?", "answer": "9"},
            ]
        elif split == "test":
            return [
                {"id": "0", "question": "What is 5+5?", "answer": "10"},
            ]
        return []

    def get_prompt(self):
        return [TextBlock(text=self.task_spec["question"])]

    @tool
    def submit(self, params: SubmitInput) -> ToolOutput:
        """Submit your answer to the math problem"""
        correct = params.answer.strip() == self.task_spec["answer"]
        return ToolOutput(
            blocks=[TextBlock(text="Correct!" if correct else f"Incorrect. Answer was {self.task_spec['answer']}")],
            reward=1.0 if correct else 0.0,
            finished=True,
        )


@pytest.fixture
def server():
    return Server([GSM8KEnv])


@pytest.fixture
async def client(server):
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as c:
        yield c


def _parse_sse(text: str) -> list[tuple[str, str]]:
    """Parse SSE text into (event, data) tuples."""
    events = []
    current_event = None
    data_lines = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line == "":
            if current_event:
                events.append((current_event, "\n".join(data_lines)))
            current_event = None
            data_lines = []
    if current_event:
        events.append((current_event, "\n".join(data_lines)))
    return events


def _extract_sid(resp_text: str) -> str:
    """Extract SID from SSE create_session response."""
    events = _parse_sse(resp_text)
    for event_type, data in events:
        if event_type == "task_id":
            return data.strip()
    raise ValueError(f"No task_id event found in SSE response: {resp_text!r}")


async def _create_session_sid(server) -> str:
    """Create a session ID directly through the server internals (bypasses SSE)."""
    sid = str(uuid.uuid4())
    return sid


# ── Discovery Endpoints ──

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_environments(client):
    resp = await client.get("/list_environments")
    assert resp.status_code == 200
    envs = resp.json()
    assert "gsm8kenv" in envs


@pytest.mark.asyncio
async def test_list_tools(client):
    resp = await client.get("/gsm8kenv/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert len(data["tools"]) == 1
    assert data["tools"][0]["name"] == "submit"
    assert data["tools"][0]["description"] == "Submit your answer to the math problem"
    assert data["tools"][0]["input_schema"] is not None


@pytest.mark.asyncio
async def test_list_splits(client):
    resp = await client.get("/gsm8kenv/splits")
    assert resp.status_code == 200
    splits = resp.json()
    assert len(splits) == 2
    names = {s["name"] for s in splits}
    assert "train" in names
    assert "test" in names


@pytest.mark.asyncio
async def test_list_tasks(client):
    resp = await client.post("/gsm8kenv/tasks", json={"split": "train"})
    assert resp.status_code == 200
    data = resp.json()
    assert "tasks" in data
    assert len(data["tasks"]) == 2
    assert data["env_name"] == "gsm8kenv"


@pytest.mark.asyncio
async def test_num_tasks(client):
    resp = await client.post("/gsm8kenv/num_tasks", json={"split": "train"})
    assert resp.status_code == 200
    assert resp.json()["num_tasks"] == 2


@pytest.mark.asyncio
async def test_get_task(client):
    resp = await client.post("/gsm8kenv/task", json={"split": "train", "index": 0})
    assert resp.status_code == 200
    task = resp.json()["task"]
    assert task["question"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_get_task_range(client):
    resp = await client.post("/gsm8kenv/task_range", json={"split": "train", "start": 0, "stop": 2})
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert len(tasks) == 2


@pytest.mark.asyncio
async def test_invalid_split(client):
    resp = await client.post("/gsm8kenv/tasks", json={"split": "nonexistent"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_unknown_environment(client):
    """Unknown env names get redirected to default (per redirect middleware), so we test by explicit name."""
    resp = await client.get("/gsm8kenv_DOESNT_EXIST/tools", follow_redirects=False)
    # Should get 308 redirect (middleware behavior) or 404
    # The redirect middleware sends unknown paths to the default env
    assert resp.status_code in (308, 404)


# ── Session / Episode Endpoints ──
# We bypass SSE-based /create_session (which has event loop issues in test)
# by creating SIDs directly and using /create


@pytest.mark.asyncio
async def test_full_episode(client, server):
    """Test a complete episode: create -> prompt -> call -> delete"""
    sid = str(uuid.uuid4())
    headers = {"X-Session-ID": sid}

    # 1. Create environment with task
    resp = await client.post("/create", headers=headers, json={
        "env_name": "gsm8kenv",
        "task_spec": {"id": "0", "question": "What is 2+2?", "answer": "4"},
    })
    assert resp.status_code == 200
    assert resp.json()["sid"] == sid

    # Wait for async setup to complete
    await asyncio.sleep(0.1)

    # 2. Get prompt
    resp = await client.get("/gsm8kenv/prompt", headers=headers)
    assert resp.status_code == 200
    prompt = resp.json()
    assert len(prompt) >= 1
    assert prompt[0]["text"] == "What is 2+2?"

    # 3. List tools (task_tools includes shared + task-specific)
    resp = await client.get("/gsm8kenv/task_tools", headers=headers)
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    assert len(tools) >= 1
    assert tools[0]["name"] == "submit"

    # 4. Call tool (correct answer) - SSE response
    resp = await client.post(
        "/gsm8kenv/call",
        headers={**headers, "Accept": "text/event-stream"},
        json={"name": "submit", "input": {"answer": "4"}},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    # Collect data from chunk and end events
    end_data = ""
    for event_type, data in events:
        if event_type in ("chunk", "end"):
            end_data += data

    result = json.loads(end_data)
    assert result["ok"] is True
    assert result["output"]["reward"] == 1.0
    assert result["output"]["finished"] is True
    assert result["output"]["blocks"][0]["text"] == "Correct!"

    # 5. Delete
    resp = await client.post("/delete", headers=headers)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_episode_incorrect_answer(client, server):
    """Test episode with incorrect answer."""
    sid = str(uuid.uuid4())
    headers = {"X-Session-ID": sid}

    await client.post("/create", headers=headers, json={
        "env_name": "gsm8kenv",
        "task_spec": {"id": "0", "question": "What is 2+2?", "answer": "4"},
    })
    await asyncio.sleep(0.1)

    resp = await client.post(
        "/gsm8kenv/call",
        headers={**headers, "Accept": "text/event-stream"},
        json={"name": "submit", "input": {"answer": "5"}},
    )
    events = _parse_sse(resp.text)
    end_data = ""
    for event_type, data in events:
        if event_type in ("chunk", "end"):
            end_data += data
    result = json.loads(end_data)
    assert result["ok"] is True
    assert result["output"]["reward"] == 0.0
    assert result["output"]["finished"] is True

    await client.post("/delete", headers=headers)


@pytest.mark.asyncio
async def test_episode_with_split_index(client, server):
    """Test creating an episode using split/index instead of task_spec."""
    sid = str(uuid.uuid4())
    headers = {"X-Session-ID": sid}

    resp = await client.post("/create", headers=headers, json={
        "env_name": "gsm8kenv",
        "split": "test",
        "index": 0,
    })
    assert resp.status_code == 200
    await asyncio.sleep(0.1)

    resp = await client.get("/gsm8kenv/prompt", headers=headers)
    assert resp.status_code == 200
    prompt = resp.json()
    assert prompt[0]["text"] == "What is 5+5?"

    await client.post("/delete", headers=headers)


@pytest.mark.asyncio
async def test_ping(client, server):
    """Test session keepalive."""
    sid = str(uuid.uuid4())
    headers = {"X-Session-ID": sid}

    await client.post("/create", headers=headers, json={
        "env_name": "gsm8kenv",
        "task_spec": {"id": "0", "question": "x", "answer": "y"},
    })
    await asyncio.sleep(0.1)

    resp = await client.post("/ping", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    await client.post("/delete", headers=headers)


@pytest.mark.asyncio
async def test_missing_session_header(client):
    """Endpoints requiring X-Session-ID should return 400 without it."""
    resp = await client.post("/ping")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_session(client):
    """Test delete_session endpoint."""
    sid = str(uuid.uuid4())
    resp = await client.post("/delete_session", headers={"X-Session-ID": sid})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_session_already_exists(client, server):
    """Creating a session with the same SID twice should fail."""
    sid = str(uuid.uuid4())
    headers = {"X-Session-ID": sid}

    resp = await client.post("/create", headers=headers, json={
        "env_name": "gsm8kenv",
        "task_spec": {"id": "0", "question": "x", "answer": "y"},
    })
    assert resp.status_code == 200

    # Try again with same SID
    resp = await client.post("/create", headers=headers, json={
        "env_name": "gsm8kenv",
        "task_spec": {"id": "0", "question": "x", "answer": "y"},
    })
    assert resp.status_code == 400

    await client.post("/delete", headers=headers)
