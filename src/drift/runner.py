from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from drift.config import CHAT_MODEL, EMBEDDING_MODEL, load_openai_api_key
from drift.providers import openai as openai_provider
from drift.storage import connect, insert_response, insert_run


class Prompt(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def _load_prompts(prompts_path: Path) -> list[Prompt]:
    raw = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{prompts_path} must contain a YAML list of prompts")
    prompts = [Prompt.model_validate(item) for item in raw]
    seen = set()
    for p in prompts:
        if p.id in seen:
            raise ValueError(f"duplicate prompt id: {p.id}")
        seen.add(p.id)
    return prompts


def run_baseline(prompts_path: Path, db_path: Path) -> int:
    load_openai_api_key()  # fail fast before touching disk or hitting the API
    prompts = _load_prompts(prompts_path)
    conn = connect(db_path)
    run_id = insert_run(
        conn, model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL, kind="baseline"
    )

    failures = 0
    for p in prompts:
        try:
            response = openai_provider.chat(p.text)
            embedding = openai_provider.embed(response)
        except Exception as exc:
            failures += 1
            print(f"[fail] prompt_id={p.id}: {exc}")
            continue

        insert_response(
            conn,
            run_id=run_id,
            prompt_id=p.id,
            prompt_text=p.text,
            response_text=response,
            embedding=embedding,
        )
        print(f"[ok] prompt_id={p.id} ({len(response)} chars, {len(embedding)}-dim)")

    captured = len(prompts) - failures
    print(f"Baseline complete: {captured}/{len(prompts)} prompts captured. run_id={run_id}")
    return 1 if failures else 0
