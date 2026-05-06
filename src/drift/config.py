import os
import sys

from dotenv import load_dotenv

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"


def load_openai_api_key() -> str:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key == "sk-replace-me":
        sys.stderr.write(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.\n"
        )
        sys.exit(1)
    return key


def load_anthropic_api_key() -> str:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or key == "sk-ant-replace-me":
        sys.stderr.write(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.\n"
        )
        sys.exit(1)
    return key
