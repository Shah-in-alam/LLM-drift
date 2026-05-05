# LLM Output Drift Detector

A monitoring tool that detects when your production LLM silently changes behavior. It runs a fixed evaluation suite on a schedule, embeds the responses, and alerts you when outputs drift semantically from a baseline — even when the model version string hasn't changed.

## The Problem

LLM providers update their models. Sometimes they announce it. Sometimes they don't. A "minor" backend change can quietly break your prompt chains, shift tone, alter formatting, or change refusal behavior. By the time users complain, you've been shipping degraded output for days.

This tool catches that drift before your users do.

## How It Works

1. **Define a golden prompt set** — 20–100 representative prompts covering your real use cases.
2. **Capture a baseline** — run the prompts once, store the responses and their embeddings.
3. **Schedule recurring runs** — same prompts, same parameters (temperature, system prompt, etc.), on a cron.
4. **Measure drift** — compare new embeddings against the baseline using cosine similarity, then aggregate with statistical drift metrics (PSI, KL divergence) over a rolling window.
5. **Alert and inspect** — when drift exceeds a threshold, surface the diffs in a dashboard so you can see exactly which prompts are behaving differently.

## Tech Stack

**Backend / core**
- Python 3.11+
- `pydantic` for schemas, `httpx` for provider calls
- LLM providers via `anthropic`, `openai`, or whichever you target (provider-agnostic interface)
- Embeddings: `voyage-3` or `text-embedding-3-large` (pick one and stay consistent — switching embedding models invalidates your baseline)

**Storage**
- SQLite for run metadata and prompt history (Postgres if you outgrow it)
- A vector store for embeddings — Chroma or Qdrant locally, Pinecone if you want hosted

**Scheduling**
- Plain cron + a CLI entrypoint for v1
- Prefect or Temporal if you want retries, observability, and parallel runs later

**Drift metrics**
- `numpy` and `scipy` for cosine similarity, PSI, and KL divergence
- `scikit-learn` for distribution comparisons if you want to get fancy

**Dashboard**
- Next.js + Tailwind + shadcn/ui
- Recharts for time-series drift plots
- A simple FastAPI backend exposing runs, prompts, and diffs

**Alerting**
- Slack webhook for v1
- PagerDuty / email as add-ons

## Suggested Project Structure

```
llm-drift-detector/
├── core/
│   ├── providers/        # Wrappers per LLM provider
│   ├── embeddings.py     # Embedding client
│   ├── metrics.py        # Cosine, PSI, KL divergence
│   └── runner.py         # Executes a prompt suite
├── storage/
│   ├── db.py             # SQLite + migrations
│   └── vectors.py        # Chroma/Qdrant client
├── api/                  # FastAPI app
├── dashboard/            # Next.js app
├── prompts/              # Your golden prompt YAML files
├── scripts/
│   ├── baseline.py       # Capture initial baseline
│   └── run_eval.py       # Cron entrypoint
└── README.md
```

## Roadmap

**v0.1 — Working prototype**
- CLI to capture baseline and run an eval
- SQLite storage, single provider, cosine similarity only
- Markdown report on drift

**v0.2 — Real monitoring**
- Scheduled runs, multi-provider support
- PSI and rolling-window drift detection
- Slack alerts

**v0.3 — Dashboard**
- Web UI showing per-prompt drift over time
- Side-by-side diffs of baseline vs current responses
- Threshold configuration per prompt

**v0.4 — Production polish**
- Multi-tenant support
- API for integrating into existing eval pipelines
- Statistical significance testing on drift signals (so you don't alert on noise)

## Design Decisions Worth Thinking About

- **Determinism vs reality**: do you fix `temperature=0` to make drift detection cleaner, or run with your actual production parameters and use multiple samples per prompt? The latter is more honest but needs more compute.
- **Embedding model lock-in**: once you pick an embedding model, you can't change it without re-baselining everything. Choose carefully.
- **What counts as drift?** Semantic similarity catches meaning changes but misses formatting drift (e.g., the model started using bullet points). Consider a second signal — token-level edit distance or structural checks — for catching format-only changes.
- **Baseline staleness**: should the baseline be a fixed snapshot, or a rolling average of the last N days? Fixed catches gradual drift, rolling avoids false alarms when you intentionally change your prompts.

## Why This Project Is Worth Building

- It teaches you embeddings, statistical drift detection, and LLM evaluation infrastructure in one project.
- It's genuinely useful — anyone running LLMs in production has been burned by silent provider updates.
- The scope is flexible: a usable v0.1 fits in a weekend, but you can keep extending it for months.
- It's portfolio-friendly — "I built monitoring infra for LLMs" is a clear, credible story.

## License

MIT (or your preference).
