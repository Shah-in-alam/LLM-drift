from openai import OpenAI

from drift.config import load_openai_api_key

EMBEDDING_MODEL = "text-embedding-3-large"

_client: OpenAI | None = None


def _client_lazy() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_openai_api_key())
    return _client


def embed(text: str) -> list[float]:
    resp = _client_lazy().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding
