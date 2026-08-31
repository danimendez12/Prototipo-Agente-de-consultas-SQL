import os
import re
import json

def is_rate_limit_error(exception):
    return "rate_limit" in str(exception).lower() or "429" in str(exception)

def invoke_with_backoff(llm, messages):
    return llm.invoke(messages)


def _try_salvage_from_error(error) -> dict | None:
    """
    When the model wraps the correct answer inside an invented tool name (for example, 'json'),
    Groq rejects the call with a 400 error but the generated JSON still appears in the error body.
    We recover it here instead of discarding a response that was fundamentally correct.
    """
    text = str(error)
    match = re.search(r"'failed_generation':\s*'(.*?)'\}$", text, re.DOTALL)
    if not match:
        return None
    try:
        raw = match.group(1).encode().decode("unicode_escape")
        payload = json.loads(raw)
        args = payload.get("arguments", {})
        if "tables" in args and "reasoning" in args:
            return args
    except Exception:
        return None
    return None


def build_llm(provider: str = "groq", model_name: str = "openai/gpt-oss-120b"):
    """
    Single change point for provider/model selection. Everything else in this file works the
    same regardless of the chosen backend because LangChain normalizes the tool-calling interface.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq
        resolved_key = os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Export the variable before running this script:\n"
                "  export GROQ_API_KEY='your_key_here'\n"
                "  python3 src/agent_explorador_langchain.py"
            )
        return ChatGroq(model=model_name, temperature=0, api_key=resolved_key)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)
    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")