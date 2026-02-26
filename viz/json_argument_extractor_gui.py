#!/usr/bin/env python3
import json
import re
import html
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import requests

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def default_data_path() -> str:
    root = Path(__file__).resolve().parent.parent
    base = Path(__file__).parent
    cands = [
        root / "SoccerChat_valid_xfoul_abs_preds_100_scored_v2.jsonl",
        root / "SoccerChat_valid_xfoul_abs_preds_100_scored.jsonl",
        base / "SoccerChat_valid_xfoul_abs_preds_100_scored_v2.jsonl",
        base / "SoccerChat_valid_xfoul_abs_preds_100_scored.jsonl",
        base / "sanity_pack_30.jsonl",
    ]
    for p in cands:
        if p.exists():
            return str(p)
    return str(cands[0])


def s(x) -> str:
    return "" if x is None else str(x)


def clean_question(q: str) -> str:
    t = s(q)
    t = re.sub(r"<\s*video\s*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_df(path_text: str) -> pd.DataFrame:
    p = Path(s(path_text).strip())
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("Use .jsonl or .csv")


def parse_json_strict(txt: str) -> dict:
    txt = s(txt).strip()
    if not txt:
        raise RuntimeError("Empty model response")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            raise RuntimeError(f"Non-JSON response: {txt[:240]}")
        return json.loads(m.group(0))


def ollama_extract(request_text: str, columns: list[str], model: str, url: str) -> dict:
    schema = {
        "query_contains": "",
        "filters": {"group": "", "decision_type_regex": "", "judge_score_01": None, "judge_score_regex": None, "judge_score_llm": None},
        "sort": {"by": "token_f1", "ascending": True},
        "top_n": 100,
    }
    prompt = f"""
Return ONLY valid JSON.
Extract filter/sort arguments from user request for a soccer evaluation table.

Available columns:
{columns}

Schema:
{json.dumps(schema, indent=2)}

Rules:
- Keep unknown filters as empty string or null.
- sort.by must be one of available columns; fallback token_f1.
- top_n should be 1..200.

User request:
{request_text}
""".strip()
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    return parse_json_strict(r.json().get("response", ""))


def local_extract(request_text: str, columns: list[str]) -> dict:
    t = s(request_text).lower()
    top_n = 100
    m = re.search(r"\b(top|worst|best)\s+(\d+)\b", t)
    if m:
        top_n = int(m.group(2))
    elif "100" in t:
        top_n = 100

    asc = True
    if "best" in t or "highest" in t:
        asc = False
    if "worst" in t or "lowest" in t:
        asc = True

    by = "judge_score_01" if "judge_score_01" in columns else "token_f1"
    for c in ["fail_score", "judge_score_llm", "judge_score_regex", "judge_score_01", "token_f1", "rougeL_f1", "decision_match"]:
        if c.lower() in t and c in columns:
            by = c
            break

    filt = {"group": "", "decision_type_regex": "", "judge_score_01": None, "judge_score_regex": None, "judge_score_llm": None}
    if "card" in t and "decision_type_regex" in columns:
        filt["decision_type_regex"] = "card"
    if "foul" in t and "decision_type_regex" in columns:
        filt["decision_type_regex"] = "foul"
    if "advantage" in t and "decision_type_regex" in columns:
        filt["decision_type_regex"] = "advantage"
    if "spa" in t or "dogso" in t:
        if "decision_type_regex" in columns:
            filt["decision_type_regex"] = "spa_dogso"
    if "group" in columns:
        m2 = re.search(r"group\s+([a-zA-Z0-9_\\-]+)", t)
        if m2:
            filt["group"] = m2.group(1)
    if "judge_score_01" in columns:
        if "judge 0" in t or "fail only" in t:
            filt["judge_score_01"] = 0
        if "judge 1" in t or "pass only" in t:
            filt["judge_score_01"] = 1

    return {
        "query_contains": "",
        "filters": filt,
        "sort": {"by": by, "ascending": asc},
        "top_n": max(1, min(200, top_n)),
    }


def enforce_request_constraints(request_text: str, args: dict, columns: list[str]) -> dict:
    """Hard-fix common phrases so request intent is always applied."""
    t = s(request_text).lower()
    out = args or {}
    out.setdefault("filters", {})
    out.setdefault("sort", {})

    m = re.search(r"\b(top|worst|best)\s+(\d+)\b", t)
    if m:
        out["top_n"] = max(1, min(200, int(m.group(2))))

    if "worst" in t or "lowest" in t:
        out["sort"]["ascending"] = True
    if "best" in t or "highest" in t:
        out["sort"]["ascending"] = False

    if "token_f1" in t and "token_f1" in columns:
        out["sort"]["by"] = "token_f1"
    elif "rougel_f1" in t and "rougeL_f1" in columns:
        out["sort"]["by"] = "rougeL_f1"
    elif "fail_score" in t and "fail_score" in columns:
        out["sort"]["by"] = "fail_score"

    if "decision_type_regex" in columns:
        if "is card" in t or "decision_type_regex is card" in t or "card only" in t:
            out["filters"]["decision_type_regex"] = "card"
        elif "is foul" in t:
            out["filters"]["decision_type_regex"] = "foul"
        elif "is advantage" in t:
            out["filters"]["decision_type_regex"] = "advantage"
        elif "is spa_dogso" in t or "spa_dogso" in t:
            out["filters"]["decision_type_regex"] = "spa_dogso"
    return out


def apply_args(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    out = df.copy()
    q = s(args.get("query_contains", "")).strip().lower()
    if q:
        cols = [c for c in ["query", "gt", "pred", "video"] if c in out.columns]
        if cols:
            mask = False
            for c in cols:
                mask = mask | out[c].astype(str).str.lower().str.contains(q, na=False)
            out = out[mask]

    f = args.get("filters", {}) or {}
    for k, v in f.items():
        if k not in out.columns:
            continue
        if v is None or s(v).strip() == "":
            continue
        if isinstance(v, (int, float)):
            out = out[pd.to_numeric(out[k], errors="coerce") == float(v)]
        else:
            out = out[out[k].astype(str) == s(v)]

    sort = args.get("sort", {}) or {}
    by = s(sort.get("by", "token_f1")).strip()
    asc = bool(sort.get("ascending", True))
    if by in out.columns:
        out[by] = pd.to_numeric(out[by], errors="coerce")
        out = out.sort_values(by, ascending=asc, na_position="last")

    n = max(1, min(200, int(args.get("top_n", 20))))
    return out.head(n).reset_index(drop=True)


def make_plot(df: pd.DataFrame, y_col: str, y2_col: str, title_prefix: str):
    if len(df) == 0 or y_col not in df.columns:
        return None
    d = df.copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if y2_col in d.columns:
        d[y2_col] = pd.to_numeric(d[y2_col], errors="coerce")
    need_cols = [y_col] + ([y2_col] if y2_col in d.columns else [])
    d = d.dropna(subset=need_cols).head(20).reset_index(drop=True)
    if len(d) == 0:
        return None
    x = [f"#{i+1}" for i in range(len(d))]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    vals = d[y_col].tolist()
    vals2 = d[y2_col].tolist() if y2_col in d.columns else None
    w = 0.30 if vals2 is not None else 0.45
    idx = list(range(len(vals)))
    c1 = "#ef4444"
    c2 = "#f59e0b"
    if vals2 is not None:
        bars = ax.bar([i - w / 2 for i in idx], vals, width=w, color=c1, edgecolor="white", linewidth=0.4, label=y_col)
        bars2 = ax.bar([i + w / 2 for i in idx], vals2, width=w, color=c2, edgecolor="white", linewidth=0.4, label=y2_col)
    else:
        bars = ax.bar(idx, vals, width=w, color=c1, edgecolor="white", linewidth=0.4, label=y_col)
        bars2 = []
    ymax = max(vals + (vals2 if vals2 is not None else [0]))
    ax.set_ylim(0, 1 if ymax <= 1 else max(1.0, ymax * 1.1))
    title = f"{title_prefix} | {y_col}" + (f" vs {y2_col}" if vals2 is not None else "")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_ylabel("Score")
    ax.set_xticks(idx)
    ax.set_xticklabels(x)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for b, v in zip(bars2, vals2 or []):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    if vals2 is not None:
        ax.legend(frameon=False)
    fig.tight_layout(pad=0.9)
    fig.subplots_adjust(top=0.84, bottom=0.2)
    return fig


def resolve_data_path(data_path):
    return data_path


def run(request_text, model, ollama_url, manual_json):
    actual_path = default_data_path()
    df = load_df(actual_path)
    cols = list(df.columns)
    if s(manual_json).strip():
        args = parse_json_strict(manual_json)
    else:
        try:
            args = ollama_extract(s(request_text).strip(), cols, s(model).strip() or DEFAULT_MODEL, s(ollama_url).strip() or DEFAULT_OLLAMA_URL)
        except Exception:
            args = local_extract(s(request_text).strip(), cols)
    args = enforce_request_constraints(s(request_text).strip(), args, cols)
    out = apply_args(df, args)
    fallback_used = False
    if len(out) == 0:
        # Safety fallback: keep sort/top_n but drop restrictive filters so user always sees rows.
        args_fallback = {
            "query_contains": "",
            "filters": {},
            "sort": (args.get("sort") or {}),
            "top_n": args.get("top_n", 100),
        }
        out = apply_args(df, args_fallback)
        fallback_used = True
    out = out.copy()
    if "query" in out.columns:
        out["question"] = out["query"].astype(str).apply(clean_question)

    # Compact table by default.
    show_cols = [c for c in ["question", "gt", "pred", "judge_score_llm", "judge_score_regex", "fail_score", "decision_type_regex"] if c in out.columns]
    if not show_cols:
        show_cols = list(out.columns[:12])
    llm_non_null = None
    if "judge_score_llm" in df.columns:
        llm_non_null = int(pd.to_numeric(df["judge_score_llm"], errors="coerce").notna().sum())
    top_n = int(args.get("top_n", len(out)))
    chosen_sort = s((args.get("sort") or {}).get("by", "")).strip()
    status = f"Matched {len(out)} / Total {len(df)} | top_n={top_n} | sort_by={chosen_sort or 'auto'}"
    if fallback_used:
        status += " | note: original filters returned 0; showing fallback top rows"
    if chosen_sort and chosen_sort not in cols:
        status += f" | note: '{chosen_sort}' not in columns"
    if llm_non_null is not None:
        status += f" | judge_score_llm_non_null={llm_non_null}"

    # Human-friendly column names.
    table_df = out[show_cols].copy()
    # Short text view for compact columns.
    for src, dst in [("question", "question_short"), ("gt", "gt_short"), ("pred", "pred_short")]:
        if src in table_df.columns:
            table_df[dst] = table_df[src].astype(str).apply(lambda x: (x[:220] + "...") if len(x) > 220 else x)
    compact_cols = [c for c in ["question_short", "gt_short", "pred_short", "judge_score_llm", "judge_score_regex", "fail_score"] if c in table_df.columns]
    table_df = table_df[compact_cols]
    rename_map = {
        "question_short": "Q",
        "gt_short": "GT",
        "pred_short": "Pred",
        "decision_type_regex": "DecisionType",
        "judge_score_llm": "Score_LLM",
        "judge_score_regex": "Score_Regex",
        "decision_match": "DecisionMatch",
        "fail_score": "FailScore",
        "token_f1": "TokenF1",
        "rougeL_f1": "RougeLF1",
    }
    table_df = table_df.rename(columns=rename_map)
    for c in ["FailScore", "TokenF1", "RougeLF1"]:
        if c in table_df.columns:
            table_df[c] = pd.to_numeric(table_df[c], errors="coerce").round(4)
    for c in ["Score_LLM", "Score_Regex", "DecisionMatch"]:
        if c in table_df.columns:
            table_df[c] = pd.to_numeric(table_df[c], errors="coerce")

    def _fmt(v, dec=4):
        try:
            fv = float(v)
            if fv.is_integer():
                return str(int(fv))
            return f"{fv:.{dec}f}"
        except Exception:
            return str(v)

    # Top-3 readable preview block.
    preview_lines = ["### Top 3 Rows (Readable)"]
    for i, row in table_df.head(3).iterrows():
        q = str(row.get("Q", ""))
        gtv = str(row.get("GT", ""))
        pv = str(row.get("Pred", ""))
        score_line = (
            f"LLM={_fmt(row.get('Score_LLM', ''))} | Regex={_fmt(row.get('Score_Regex', ''))} | "
            f"Fail={_fmt(row.get('FailScore', ''))} | TokenF1={_fmt(row.get('TokenF1', ''))} | RougeLF1={_fmt(row.get('RougeLF1', ''))}"
        )
        preview_lines.append(f"#### Row {i+1}")
        preview_lines.append(f"**Question**  \n{q}")
        preview_lines.append(f"**Ground Truth**  \n{gtv}")
        preview_lines.append(f"**Prediction**  \n{pv}")
        preview_lines.append(f"**Scores**  \n{score_line}")
        preview_lines.append("---")
    preview_md = "\n".join(preview_lines)

    heat_html = build_fail_heat_table_html(table_df)
    caption_md = f"### Filtered Cases: {len(table_df)} / {len(df)}"
    return json.dumps(args, indent=2), caption_md, heat_html, preview_md, table_df


def run_ui(*args):
    try:
        return run(*args)
    except Exception as e:
        return "{}", "### Filtered Cases: 0 / 0", "<div>Error</div>", f"### Error\n{type(e).__name__}: {e}", pd.DataFrame()


def initial_run(model, ollama_url, manual_json):
    default_request = "show top 100 by judge_score_01"
    return run_ui(default_request, model, ollama_url, manual_json)


def build_fail_heat_table_html(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "<div>No rows.</div>"
    if "FailScore" not in df.columns:
        return "<div>FailScore column not found.</div>"

    d = df.copy()
    d["FailScore"] = pd.to_numeric(d["FailScore"], errors="coerce")
    fs = d["FailScore"].dropna()
    if len(fs) == 0:
        return "<div>FailScore has no numeric values.</div>"

    mn = float(fs.min())
    mx = float(fs.max())

    cols = [c for c in ["Q", "GT", "Pred", "Score_LLM", "Score_Regex", "FailScore"] if c in d.columns]
    rows_html = []
    for _, row in d.iterrows():
        v = row.get("FailScore")
        try:
            v = float(v)
        except Exception:
            v = mn
        # absolute bands so low fail score is always light
        if v <= 1.0:
            bg = "rgba(34,197,94,0.10)"     # green (good)
        elif v <= 2.0:
            bg = "rgba(250,204,21,0.10)"    # yellow (medium)
        elif v <= 3.0:
            bg = "rgba(249,115,22,0.11)"    # orange (bad)
        else:
            bg = "rgba(239,68,68,0.12)"     # red (worst)
        cells = []
        for c in cols:
            txt = html.escape(str(row.get(c, "")))
            if c in ("Q", "GT", "Pred") and len(txt) > 220:
                txt = txt[:220] + "..."
            cells.append(f"<td style='padding:6px;border:1px solid #e5e7eb;vertical-align:top'>{txt}</td>")
        rows_html.append(f"<tr style='background:{bg}'>" + "".join(cells) + "</tr>")

    header = "".join([f"<th style='padding:6px;border:1px solid #e5e7eb;text-align:left;background:#f8fafc'>{html.escape(c)}</th>" for c in cols])
    table = (
        f"<div style='font-size:12px;margin-bottom:6px'>FailScore min={mn:.4f}, max={mx:.4f} "
        f"(darker red = worse)</div>"
        "<div style='max-height:460px;overflow:auto;border:1px solid #e5e7eb'>"
        "<table style='border-collapse:collapse;width:100%;font-size:12px'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    return table


def build_ui():
    with gr.Blocks(
        title="Soccer JSON Argument Extractor",
        theme=gr.themes.Default(),
        css=":root { color-scheme: light; }",
    ) as app:
        gr.Markdown("## Soccer JSON Argument Extractor")
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                request_text = gr.Textbox(
                    label="Request",
                    value="show top 100 by judge_score_01",
                    lines=3,
                )
                run_btn = gr.Button("Extract + Apply", variant="primary")
                with gr.Accordion("Advanced (optional)", open=False):
                    model = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL)
                    ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
                    manual_json = gr.Textbox(label="Manual JSON override", lines=8)
            with gr.Column(scale=3, min_width=860):
                caption = gr.Markdown("### Filtered Cases: 0 / 0")
                table = gr.Dataframe(label="Filtered Table (Plain)", wrap=True)
                with gr.Accordion("Optional Heat View", open=False):
                    heat_table = gr.HTML(label="Row color by FailScore")
                with gr.Accordion("Top 3 Rows (Readable)", open=False):
                    preview = gr.Markdown("No data yet.")
                with gr.Accordion("Parsed Arguments JSON", open=False):
                    parsed = gr.Code(label="", language="json")
        run_btn.click(
            fn=run_ui,
            inputs=[request_text, model, ollama_url, manual_json],
            outputs=[parsed, caption, heat_table, preview, table],
        )
        app.load(
            fn=initial_run,
            inputs=[model, ollama_url, manual_json],
            outputs=[parsed, caption, heat_table, preview, table],
        )
    return app


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7864)
