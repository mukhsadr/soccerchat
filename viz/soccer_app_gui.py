#!/usr/bin/env python3
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import requests

DEFAULT_DATA_PATH = "score.jsonl"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_VIDEO_ROOT = ""


def _s(x) -> str:
    return "" if x is None else str(x)


def load_eval_table(data_path: str) -> pd.DataFrame:
    p = Path(_s(data_path).strip())
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Use .jsonl or .csv for data file.")

    if "video" in df.columns and "video_file" not in df.columns:
        df["video_file"] = df["video"].astype(str).apply(lambda x: Path(x).name)
    return df


def ollama_extract(request_text: str, model: str, ollama_url: str) -> dict:
    schema = {
        "query": "",
        "group": "",
        "judge_score_01": None,
        "sort_by": "token_f1",
        "ascending": True,
        "top_n": 20,
    }
    prompt = f"""
Return ONLY valid JSON.
Extract filters for soccer evaluation browsing.

Schema:
{json.dumps(schema, indent=2)}

Rules:
- "query" is a substring for video/case lookup.
- "group" empty if unspecified.
- "judge_score_01" can be 0, 1, or null.
- "sort_by" should be one of token_f1, rougeL_f1, judge_score_01, judge_score_cont, judge_score_llm.
- "ascending" true for "worst/lowest", false for "best/highest".
- "top_n" integer 1..200.

User request:
{request_text}
""".strip()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(ollama_url, json=payload, timeout=90)
    r.raise_for_status()
    response_text = r.json().get("response", "").strip()
    return json.loads(response_text)


def pick_display_cols(df: pd.DataFrame):
    preferred = [
        "video_file",
        "video",
        "group",
        "token_f1",
        "rougeL_f1",
        "judge_score_01",
        "judge_score_cont",
        "judge_score_llm",
        "query",
        "pred",
        "gt",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = list(df.columns[:12])
    return cols


def resolve_video_path(row: pd.Series, video_root: str) -> str | None:
    v = _s(row.get("video", "")).strip()
    if not v:
        return None
    if os.path.exists(v):
        return v
    root = _s(video_root).strip()
    if root:
        candidate = str(Path(root) / Path(v).name)
        if os.path.exists(candidate):
            return candidate
    return None


def build_metrics_plot(row: pd.Series):
    keys = [k for k in ["token_f1", "rougeL_f1", "judge_score_cont", "judge_score_llm"] if k in row.index]
    vals = []
    names = []
    for k in keys:
        try:
            vals.append(float(row[k]))
            names.append(k)
        except Exception:
            pass

    if not vals:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(names, vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"][: len(names)], edgecolor="#222")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Sample Metrics")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, min(0.98, v + 0.02), f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def run_soccer_query(
    data_path,
    request_text,
    manual_query,
    manual_group,
    manual_judge,
    manual_sort,
    manual_asc,
    top_n,
    video_root,
    model,
    ollama_url,
):
    data_path = _s(data_path).strip() or DEFAULT_DATA_PATH
    df = load_eval_table(data_path)

    parsed = {
        "query": _s(manual_query).strip(),
        "group": _s(manual_group).strip(),
        "judge_score_01": None if _s(manual_judge).strip() == "" else int(manual_judge),
        "sort_by": _s(manual_sort).strip() or "token_f1",
        "ascending": bool(manual_asc),
        "top_n": int(top_n),
    }

    request_text = _s(request_text).strip()
    if request_text and not parsed["query"] and not parsed["group"] and parsed["judge_score_01"] is None:
        parsed = ollama_extract(request_text, _s(model).strip() or DEFAULT_MODEL, _s(ollama_url).strip() or DEFAULT_OLLAMA_URL)

    query = _s(parsed.get("query", "")).strip().lower()
    group = _s(parsed.get("group", "")).strip()
    judge_val = parsed.get("judge_score_01", None)
    sort_by = _s(parsed.get("sort_by", "token_f1")).strip()
    asc = bool(parsed.get("ascending", True))
    n = max(1, min(200, int(parsed.get("top_n", 20))))

    f = df.copy()
    if query:
        search_cols = [c for c in ["video_file", "video", "query"] if c in f.columns]
        if search_cols:
            mask = False
            for c in search_cols:
                mask = mask | f[c].astype(str).str.lower().str.contains(query, na=False)
            f = f[mask]
    if group and "group" in f.columns:
        f = f[f["group"].astype(str) == group]
    if judge_val in (0, 1) and "judge_score_01" in f.columns:
        f = f[pd.to_numeric(f["judge_score_01"], errors="coerce") == int(judge_val)]
    if sort_by in f.columns:
        f[sort_by] = pd.to_numeric(f[sort_by], errors="coerce")
        f = f.sort_values(sort_by, ascending=asc, na_position="last")

    out = f.head(n).reset_index(drop=True)
    display_cols = pick_display_cols(out)

    status = f"Rows matched: {len(f)} | Showing top {len(out)} | sort_by={sort_by} asc={asc}"

    if len(out) == 0:
        return json.dumps(parsed, indent=2), status, pd.DataFrame(columns=display_cols), None, None, "", "", ""

    row0 = out.iloc[0]
    video_path = resolve_video_path(row0, video_root)
    fig = build_metrics_plot(row0)
    query_text = _s(row0.get("query", ""))
    pred_text = _s(row0.get("pred", ""))
    gt_text = _s(row0.get("gt", ""))

    return json.dumps(parsed, indent=2), status, out[display_cols], video_path, fig, query_text, pred_text, gt_text


def build_ui():
    with gr.Blocks(title="SoccerChat Eval Browser") as app:
        gr.Markdown("## SoccerChat Eval Browser\nRequest -> parsed filters -> ranked rows -> sample plot/video")
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                data_path = gr.Textbox(label="Data path (.jsonl or .csv)", value=DEFAULT_DATA_PATH)
                request_text = gr.Textbox(
                    label="Natural language request",
                    value="show worst 20 by token_f1 in group test",
                    lines=3,
                )
                run_btn = gr.Button("Run", variant="primary")

                with gr.Accordion("Manual Override", open=False):
                    manual_query = gr.Textbox(label="Query substring")
                    manual_group = gr.Textbox(label="Group")
                    manual_judge = gr.Dropdown(label="Judge score 0/1", choices=["", "0", "1"], value="")
                    manual_sort = gr.Dropdown(
                        label="Sort by",
                        choices=["token_f1", "rougeL_f1", "judge_score_01", "judge_score_cont", "judge_score_llm"],
                        value="token_f1",
                    )
                    manual_asc = gr.Checkbox(label="Ascending (worst first)", value=True)
                    top_n = gr.Slider(label="Top N", minimum=1, maximum=200, step=1, value=20)

                with gr.Accordion("Advanced", open=False):
                    video_root = gr.Textbox(label="Video root (optional)", value=DEFAULT_VIDEO_ROOT)
                    model = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL)
                    ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)

                status = gr.Textbox(label="Status", lines=2)
                parsed = gr.Code(label="Parsed Filters JSON", language="json")

            with gr.Column(scale=2, min_width=760):
                table = gr.Dataframe(label="Filtered Results", wrap=True)
                with gr.Row():
                    video = gr.Video(label="Selected Video (first row)")
                    plot = gr.Plot(label="Selected Row Metrics")
                sample_q = gr.Textbox(label="Sample Query", lines=3)
                sample_pred = gr.Textbox(label="Sample Prediction", lines=4)
                sample_gt = gr.Textbox(label="Sample Ground Truth", lines=4)

        run_btn.click(
            fn=run_soccer_query,
            inputs=[
                data_path,
                request_text,
                manual_query,
                manual_group,
                manual_judge,
                manual_sort,
                manual_asc,
                top_n,
                video_root,
                model,
                ollama_url,
            ],
            outputs=[parsed, status, table, video, plot, sample_q, sample_pred, sample_gt],
        )
    return app


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7862)
