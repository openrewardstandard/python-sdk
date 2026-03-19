"""
Client test script — mirrors the "Your First Environment" tutorial from docs.openreward.ai
but uses `ors.client` instead of `openreward`.

Usage:
  1. In one terminal:  python examples/gsm8k_server.py
  2. In another:       python examples/test_client.py
"""
from ors.client import ORS

BASE_URL = "http://localhost:8080"

or_client = ORS(base_url=BASE_URL)
environment = or_client.environments.get(name="gsm8k")

# ── Discovery ──
print("=== Splits ===")
splits = environment.list_splits()
print(splits)

print("\n=== Tasks (train, first 2) ===")
tasks = environment.list_tasks("train")
for t in tasks[:2]:
    print(f"  {t.task_spec['id']}: {t.task_spec['question'][:60]}...")

print(f"\n=== Num tasks (train) ===")
n = environment.num_tasks("train")
print(f"  {n} tasks")

print("\n=== Task range (train, 0..2) ===")
task_range = environment.get_task_range("train", start=0, stop=2)
print(f"  Got {len(task_range)} tasks")

print("\n=== Tools ===")
tools = environment.list_tools()
for t in tools:
    print(f"  {t.name}: {t.description}")
    print(f"    schema: {t.input_schema}")

print("\n=== Tools (OpenAI format) ===")
openai_tools = environment.list_tools(format="openai")
for t in openai_tools:
    print(f"  {t}")

print("\n=== Tools (Anthropic format) ===")
anthropic_tools = environment.list_tools(format="anthropic")
for t in anthropic_tools:
    print(f"  {t}")

# ── Session: correct answer ──
print("\n=== Session: correct answer ===")
example_task = tasks[0]
print(f"  Task: {example_task.task_spec['question'][:60]}...")

with environment.session(task=example_task) as session:
    prompt = session.get_prompt()
    print(f"  Prompt: {prompt[0].text[:60]}...")

    session_tools = session.list_tools()
    print(f"  Session tools: {[t.name for t in session_tools]}")

    tool_result = session.call_tool("answer", {"answer": "72"})
    print(f"  Result: blocks={[b.text for b in tool_result.blocks]}, reward={tool_result.reward}, finished={tool_result.finished}")

# ── Session: incorrect answer ──
print("\n=== Session: incorrect answer ===")
with environment.session(task=example_task) as session:
    tool_result = session.call_tool("answer", {"answer": "999"})
    print(f"  Result: blocks={[b.text for b in tool_result.blocks]}, reward={tool_result.reward}, finished={tool_result.finished}")

# ── Session: using split + index ──
print("\n=== Session: split/index ===")
with environment.session(split="test", index=0) as session:
    prompt = session.get_prompt()
    print(f"  Prompt: {prompt[0].text}")
    tool_result = session.call_tool("answer", {"answer": "3"})
    print(f"  Result: blocks={[b.text for b in tool_result.blocks]}, reward={tool_result.reward}, finished={tool_result.finished}")

print("\nAll tests passed!")
or_client.close()
