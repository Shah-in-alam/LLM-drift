from openai import OpenAI

from drift.config import CHAT_MODEL, EMBEDDING_MODEL, load_openai_api_key

_SYSTEM_PROMPT = "You are a concise assistant. Answer briefly and directly."

_client_singleton: OpenAI | None = None


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI(api_key=load_openai_api_key())
    return _client_singleton


def chat(prompt: str) -> str:
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def embed(text: str) -> list[float]:
    resp = _client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding
