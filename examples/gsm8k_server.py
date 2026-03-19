"""
GSM8K Environment — adapted from the OpenReward "Your First Environment" tutorial.

Uses inline data instead of parquet files so it works standalone.
Run with:  python examples/gsm8k_server.py
"""
from pydantic import BaseModel

from ors import Environment, Server, Split, TextBlock, ToolOutput, tool

# ── Inline GSM8K data (subset) ──

train_tasks = [
    {"id": "0", "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
    {"id": "1", "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
    {"id": "2", "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?", "answer": "5"},
    {"id": "3", "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "answer": "42"},
    {"id": "4", "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": "624"},
]

test_tasks = [
    {"id": "0", "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
    {"id": "1", "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
]


class GSM8KTaskSpec(BaseModel):
    id: str
    question: str
    answer: str


class AnswerParams(BaseModel):
    answer: str


class GSM8K(Environment):
    """A GSM8K environment for math word problems."""

    def __init__(self, task_spec=None, secrets=None):
        super().__init__(task_spec or {})
        if task_spec:
            self.config = GSM8KTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            return train_tasks
        elif split == "test":
            return test_tasks
        raise ValueError(f"Unknown split: {split}")

    def get_prompt(self):
        return [TextBlock(text=self.config.question)]

    @tool
    def answer(self, params: AnswerParams) -> ToolOutput:
        """Submit your final answer to complete the task."""
        # Simple string comparison (the real GSM8K env uses math_verify)
        gold = self.config.answer.strip()
        submitted = params.answer.strip()
        is_correct = gold == submitted

        return ToolOutput(
            blocks=[TextBlock(text="Correct!" if is_correct else "Wrong!")],
            reward=1.0 if is_correct else 0.0,
            finished=True,
        )


if __name__ == "__main__":
    Server([GSM8K]).run()
