# ORS SDK

[![Docs](https://img.shields.io/badge/docs-openrewardstandard.io-blue)](https://openrewardstandard.io)

Python SDK for the [Open Reward Standard](https://openrewardstandard.io) (ORS) — an HTTP-based protocol for connecting AI agents to reinforcement learning environments.

## Installation

```bash
pip install ors-sdk
```

## Quick Start

### Server (hosting an environment)

```python
from pydantic import BaseModel
from ors import Environment, Server, tool, ToolOutput, TextBlock, Split


class SubmitInput(BaseModel):
    answer: str


class MathEnv(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        return [{"question": "What is 2+2?", "answer": "4"}]

    def get_prompt(self):
        return [TextBlock(text=self.task_spec["question"])]

    @tool
    def submit(self, params: SubmitInput) -> ToolOutput:
        correct = params.answer.strip() == self.task_spec["answer"]
        return ToolOutput(
            blocks=[TextBlock(text="Correct!" if correct else "Incorrect")],
            reward=1.0 if correct else 0.0,
            finished=True,
        )


if __name__ == "__main__":
    Server([MathEnv]).run(port=8080)
```

### Client (connecting to an environment)

```python
from ors.client import ORS

client = ORS(base_url="http://localhost:8080")
env = client.environment("mathenv")
tasks = env.list_tasks(split="test")

with env.session(task=tasks[0]) as session:
    prompt = session.get_prompt()
    print(f"Question: {prompt[0].text}")

    result = session.call_tool("submit", {"answer": "4"})
    print(f"Reward: {result.reward}, Finished: {result.finished}")
```

## Links

- [ORS Specification](https://openrewardstandard.io)
- [ORS HTTP API](https://openrewardstandard.io/specification/http-api)

## License

Apache 2.0
