"""Streamlit dashboard for the LLM Drift Detector.

Run: `uv run streamlit run dashboard.py`
Then open the URL Streamlit prints (usually http://localhost:8501).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from drift.metrics import cosine
from drift.storage import connect, latest_baseline_run, responses_for_run

st.set_page_config(page_title="LLM Drift Detector", page_icon="📈", layout="wide")
st.title("📈 LLM Drift Detector")
st.caption("Monitor your production LLM for silent behavior changes.")

# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    db_path_str = st.text_input("SQLite database", value="drift.db")
    reports_dir_str = st.text_input("Reports directory", value="reports")
    threshold = st.slider(
        "Drift threshold (cosine similarity)",
        min_value=0.50,
        max_value=1.00,
        value=0.95,
        step=0.01,
    )
    st.caption("Anything below this counts as drift.")

db_path = Path(db_path_str)
reports_dir = Path(reports_dir_str)

if not db_path.exists():
    st.warning(f"No database at `{db_path}`. Run `uv run drift baseline` first, then come back.")
    st.stop()

conn = connect(db_path)

# --- Runs table ------------------------------------------------------------

st.header("Runs")
runs_df = pd.read_sql_query(
    "SELECT id, started_at, kind, provider, model, embedding_model FROM runs ORDER BY id DESC",
    conn,
)
if runs_df.empty:
    st.info("No runs in the database yet.")
    st.stop()

st.dataframe(runs_df, use_container_width=True, hide_index=True)

baseline = latest_baseline_run(conn)
if baseline is None:
    st.warning("No baseline run captured yet — drift comparisons need a baseline.")
    st.stop()

st.success(
    f"Comparing against latest baseline: **run {baseline['id']}** "
    f"({baseline['provider']} / {baseline['model']}, captured {baseline['started_at']})"
)

# --- Drift over time chart -------------------------------------------------

st.header("Per-prompt similarity over time")

baseline_responses = responses_for_run(conn, baseline["id"])
baseline_by_pid = {r["prompt_id"]: r for r in baseline_responses}

eval_runs = runs_df[runs_df["kind"] == "eval"]

rows = []
for _, run in eval_runs.iterrows():
    eval_resps = responses_for_run(conn, int(run["id"]))
    for r in eval_resps:
        pid = r["prompt_id"]
        b = baseline_by_pid.get(pid)
        if b is None:
            continue
        sim = cosine(b["embedding"], r["embedding"])
        rows.append(
            {
                "run_id": int(run["id"]),
                "started_at": run["started_at"],
                "prompt_id": pid,
                "similarity": sim,
                "passed": sim >= threshold,
            }
        )

if not rows:
    st.info(
        "No eval runs yet (or none have prompts in common with the baseline). "
        "Run `uv run drift run` after baselining to populate this chart."
    )
else:
    sim_df = pd.DataFrame(rows)
    chart_df = sim_df.pivot(index="run_id", columns="prompt_id", values="similarity")
    st.line_chart(chart_df, height=320)
    st.caption(f"Threshold = {threshold:.2f}. Lines that dip below it indicate drift.")

    fail_count = (~sim_df["passed"]).sum()
    total_compared = len(sim_df)
    if fail_count:
        st.error(f"⚠️ {fail_count}/{total_compared} prompt-runs below threshold {threshold:.2f}.")
    else:
        st.success(f"✅ All {total_compared} prompt-runs at or above threshold.")

# --- Per-prompt explorer ---------------------------------------------------

st.header("Per-prompt explorer")

all_prompt_ids = sorted({r["prompt_id"] for r in baseline_responses})
if all_prompt_ids:
    chosen = st.selectbox("Pick a prompt", all_prompt_ids)
    b_resp = baseline_by_pid.get(chosen)
    if b_resp:
        st.subheader(f"Prompt: `{chosen}`")
        st.markdown(f"**Prompt text:** {b_resp['prompt_text']}")
        st.markdown(f"**Baseline response (run {baseline['id']}):**")
        st.code(b_resp["response_text"], language="text")

        # Show every eval response for this prompt
        history_rows = []
        for _, run in eval_runs.iterrows():
            for r in responses_for_run(conn, int(run["id"])):
                if r["prompt_id"] == chosen:
                    sim = cosine(b_resp["embedding"], r["embedding"])
                    history_rows.append(
                        {
                            "run_id": int(run["id"]),
                            "started_at": run["started_at"],
                            "similarity": round(sim, 4),
                            "response": r["response_text"],
                        }
                    )
        if history_rows:
            st.markdown("**Eval history for this prompt:**")
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No eval runs include this prompt yet.")

# --- Latest report viewer --------------------------------------------------

st.header("Latest drift report")

report_files = sorted(reports_dir.glob("run-*.md")) if reports_dir.exists() else []
if not report_files:
    st.info(f"No reports in `{reports_dir}/` yet. They appear after `drift run`.")
else:
    chosen_report = st.selectbox(
        "Report",
        options=report_files,
        format_func=lambda p: p.name,
        index=len(report_files) - 1,
    )
    st.markdown(chosen_report.read_text(encoding="utf-8"))
