from openai import OpenAI

from drift.config import load_openai_api_key

_SYSTEM_PROMPT = "You are a concise assistant. Answer briefly and directly."


class OpenAIChatProvider:
    name = "openai"
    chat_model = "gpt-4o-mini"

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    def _client_lazy(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=load_openai_api_key())
        return self._client

    def chat(self, prompt: str, *, temperature: float) -> str:
        resp = self._client_lazy().chat.completions.create(
            model=self.chat_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""
