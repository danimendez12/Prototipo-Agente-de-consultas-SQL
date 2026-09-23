"""Centralized pricing metadata for evaluation scripts."""

MODEL_PRICING = {
    "claude-sonnet-5": {"input": 2 / 1_000_000, "output": 10 / 1_000_000},
    "groq/openai/gpt-oss-120b": {"input": 0.15 / 1_000_000, "output": 0.75 / 1_000_000},
    "groq/openai/gpt-oss-20b": {"input": 0.10 / 1_000_000, "output": 0.50 / 1_000_000},
}


def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Return the estimated USD cost for a model call based on token usage."""
    rates = MODEL_PRICING[model_key]
    return input_tokens * rates["input"] + output_tokens * rates["output"]
