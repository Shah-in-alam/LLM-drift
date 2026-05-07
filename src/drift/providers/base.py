from typing import Protocol


class ChatProvider(Protocol):
    name: str
    chat_model: str

    def chat(self, prompt: str, *, temperature: float) -> str: ...
