from anthropic import Anthropic

from drift.config import load_anthropic_api_key

_SYSTEM_PROMPT = "You are a concise assistant. Answer briefly and directly."


class AnthropicChatProvider:
    name = "anthropic"
    chat_model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self._client: Anthropic | None = None

    def _client_lazy(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=load_anthropic_api_key())
        return self._client

    def chat(self, prompt: str, *, temperature: float) -> str:
        resp = self._client_lazy().messages.create(
            model=self.chat_model,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=temperature,
        )
        if len(resp.content) != 1 or resp.content[0].type != "text":
            raise RuntimeError(
                f"Unexpected Anthropic response shape: {len(resp.content)} blocks, "
                f"types={[b.type for b in resp.content]}"
            )
        return resp.content[0].text
