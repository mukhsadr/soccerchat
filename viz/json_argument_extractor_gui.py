#!/usr/bin/env python3
import json
import re
import html
from pathlib import Path
from collections import Counter

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def default_data_path() -> str:
    root = Path(__file__).resolve().parent.parent
    base = Path(__file__).parent
    cands = [
        root / "SoccerChat_valid_xfoul_abs_preds_100_scored_v4.jsonl",
        root / "SoccerChat_valid_xfoul_abs_preds_100_scored_v2.jsonl",
        root / "SoccerChat_valid_xfoul_abs_preds_100_scored.jsonl",
        base / "SoccerChat_valid_xfoul_abs_preds_100_scored_v4.jsonl",
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


def build_plot(df: pd.DataFrame, plot_choice: str):
    if len(df) == 0:
        return None
    c = s(plot_choice).strip()
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if c == "1":
        if "FailScore" not in df.columns:
            plt.close(fig)
            return None
        d = df.copy()
        d["FailScore"] = pd.to_numeric(d["FailScore"], errors="coerce")
        d = d.dropna(subset=["FailScore"]).reset_index(drop=True)
        if len(d) == 0:
            plt.close(fig)
            return None
        vals = d["FailScore"].tolist()
        mn, mx = min(vals), max(vals)
        denom = (mx - mn) if mx > mn else 1.0
        # low fail -> green, high fail -> red
        colors = [(v - mn) / denom for v in vals]
        ax.bar(range(len(d)), vals, color=plt.cm.RdYlGn_r(colors), edgecolor="white", linewidth=0.3, alpha=0.92)
        ax.set_title(f"SoccerChat VLM Evaluation on {len(d)} Video Data Points", fontsize=13, pad=14)
        ax.set_xlabel(
            "Each bar = one case. decision_match: exact decision correct (0/1);\n"
            "token_f1: token overlap between prediction and ground truth;\n"
            "rougeL_f1: sequence-level overlap (longest common subsequence) F1.",
            fontsize=10,
            labelpad=10,
        )
        ax.set_ylabel(
            "Fail Score (higher = worse)\n"
            "(1-decision_match)*2 + (1-token_f1) + (1-rougeL_f1)",
            fontsize=10,
        )
        xt = list(range(0, len(d), 10))
        ax.set_xticks(xt)
        ax.set_xticklabels([str(i) for i in xt], fontsize=9)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r), ax=ax, pad=0.01)
        cbar.set_label("Low fail (green)  ->  High fail (red)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    elif c == "2":
        if ("Score_LLM" not in df.columns) or ("Score_Regex" not in df.columns):
            plt.close(fig)
            return None
        d = df.copy()
        d["Score_LLM"] = pd.to_numeric(d["Score_LLM"], errors="coerce")
        d["Score_Regex"] = pd.to_numeric(d["Score_Regex"], errors="coerce")
        if "FailScore" in d.columns:
            d["FailScore"] = pd.to_numeric(d["FailScore"], errors="coerce")
        d = d.dropna(subset=["Score_LLM", "Score_Regex"])
        if len(d) == 0:
            plt.close(fig)
            return None
        if "FailScore" in d.columns and d["FailScore"].notna().any():
            mn = float(d["FailScore"].min())
            mx = float(d["FailScore"].max())
            denom = (mx - mn) if mx > mn else 1.0
            norm_vals = (d["FailScore"] - mn) / denom
            sc = ax.scatter(
                d["Score_Regex"], d["Score_LLM"],
                c=norm_vals, cmap=plt.cm.RdYlGn_r, s=36, alpha=0.85
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label("Low fail  →  High fail", fontsize=8)
            cbar.ax.tick_params(labelsize=8)
            ax.set_title("Option 2: LLM vs Regex (color by FailScore)")
        else:
            ax.scatter(d["Score_Regex"], d["Score_LLM"], s=36, alpha=0.8, color="#2563eb")
            ax.set_title("Option 2: LLM vs Regex")
        ax.set_xlabel("Score_Regex")
        ax.set_ylabel("Score_LLM")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    else:
        # Option 3: grouped fail-score plot by top error-tag combinations (robust style).
        if "FailScore" not in df.columns:
            plt.close(fig)
            return None
        tag_col = "ErrorTags" if "ErrorTags" in df.columns else ("error_tags" if "error_tags" in df.columns else None)
        if tag_col is None:
            plt.close(fig)
            return None

        d = df.copy()
        d["FailScore"] = pd.to_numeric(d["FailScore"], errors="coerce")
        if "DecisionMatch" in d.columns:
            d["decision_match_num"] = pd.to_numeric(d["DecisionMatch"], errors="coerce")
        else:
            d["decision_match_num"] = np.nan
        if "Score_Regex" in d.columns:
            d["decision_match_num"] = d["decision_match_num"].fillna(pd.to_numeric(d["Score_Regex"], errors="coerce"))
        d = d.dropna(subset=["FailScore"]).reset_index(drop=True)
        if len(d) == 0:
            plt.close(fig)
            return None

        def parse_tags(v):
            if isinstance(v, list):
                tags = v
            else:
                s = str(v).strip()
                if not s:
                    tags = []
                elif s.startswith("[") and s.endswith("]"):
                    tags = [x.strip().strip("'\"") for x in s[1:-1].split(",") if x.strip()]
                else:
                    for sep in [",", ";", "|", "+"]:
                        if sep in s:
                            tags = [t.strip() for t in s.split(sep) if t.strip()]
                            break
                    else:
                        tags = [s]
            return [str(t).strip() for t in tags if str(t).strip()]

        def two_line_label(group_name):
            if group_name == "other":
                return "other\n(mixed)"
            if group_name == "no_tag":
                return "no_tag\n(n/a)"
            parts = [x.strip() for x in str(group_name).split("+") if x.strip()]
            if len(parts) == 0:
                return "no_tag\n(n/a)"
            if len(parts) == 1:
                return f"{parts[0]}\n(n/a)"
            return f"{parts[0]}\n{parts[1]}"

        combo_list = []
        for v in d[tag_col].tolist():
            tags = sorted(set(parse_tags(v)))
            combo_list.append(" + ".join(tags) if tags else "no_tag")
        d["error_combo"] = combo_list

        # Keep this style stable with notebook version.
        top_groups = 3
        legend_loc = "upper right"
        xtick_every = 10
        only_decision_failed = False
        min_fail_score = None

        if only_decision_failed:
            d = d[d["decision_match_num"] == 0].copy()
        if min_fail_score is not None:
            d = d[d["FailScore"] >= float(min_fail_score)].copy()
        if len(d) == 0:
            plt.close(fig)
            return None

        keep = [k for k, _ in Counter(d["error_combo"].tolist()).most_common(top_groups)]
        d["combo_group"] = d["error_combo"].apply(lambda x: x if x in keep else "other")
        group_order = keep + (["other"] if (d["combo_group"] == "other").any() else [])

        parts = []
        for g in group_order:
            sub = d[d["combo_group"] == g].sort_values("FailScore", ascending=False).copy()
            sub["group"] = g
            parts.append(sub)
        plot_df = pd.concat(parts, axis=0).reset_index(drop=True)
        vals = plot_df["FailScore"].to_numpy()
        if len(vals) == 0:
            plt.close(fig)
            return None

        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11.6, 4.8))
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlGn_r
        bar_colors = cmap(norm(vals))
        n = len(vals)

        ax.bar(np.arange(n), vals, color=bar_colors, edgecolor="white", linewidth=0.25)
        ax.set_ylim(0, max(vmax * 1.14, 2.5))

        start = 0
        legend_rows = []
        for i, g in enumerate(group_order):
            m = int((plot_df["group"] == g).sum())
            if m == 0:
                continue
            end = start + m
            mid = (start + end - 1) / 2
            y = ax.get_ylim()[1] * (0.98 if i % 2 == 0 else 0.93)
            ax.text(
                mid,
                y,
                f"G{i+1}\n(n={m})",
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )
            if end < n:
                ax.axvline(end - 0.5, color="#475569", linestyle="--", linewidth=1.0, alpha=0.85)
            legend_rows.append((f"G{i+1}", two_line_label(g), m))
            start = end

        ax.set_title("Grouped Fail-Score Distribution (top error-tag combos)", fontsize=12, pad=10)
        ax.set_xlabel("Case index (sorted within group by fail_score)", fontsize=10)
        ax.set_ylabel("Fail score (higher = worse)", fontsize=10)
        xt = np.arange(0, n, xtick_every)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(i) for i in xt], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Fail score (low=green -> high=red)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        handles = []
        for gid, gtext, m in legend_rows:
            label = f"{gid}: {gtext.replace(chr(10), ' / ')} (n={m})"
            handles.append(Patch(facecolor="none", edgecolor="none", label=label))
        ax.legend(handles=handles, loc=legend_loc, frameon=True, fontsize=8, title="Group mapping")

    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def run(request_text, model, ollama_url, manual_json, plot_choice):
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
    show_cols = [
        c
        for c in [
            "question",
            "gt",
            "pred",
            "judge_score_llm",
            "judge_score_regex",
            "fail_score",
            "decision_type_regex",
            "error_tags",
            "error_summary",
        ]
        if c in out.columns
    ]
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
    compact_cols = [
        c
        for c in [
            "question_short",
            "gt_short",
            "pred_short",
            "judge_score_llm",
            "judge_score_regex",
            "fail_score",
            "error_tags",
            "error_summary",
            "decision_match",
        ]
        if c in table_df.columns
    ]
    table_df = table_df[compact_cols]
    rename_map = {
        "question_short": "Q",
        "gt_short": "GT",
        "pred_short": "Pred",
        "decision_type_regex": "DecisionType",
        "error_tags": "ErrorTags",
        "error_summary": "ErrorSummary",
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

    plot_fig = build_plot(table_df, plot_choice)
    heat_html = build_fail_heat_table_html(table_df)
    caption_md = f"### Filtered Cases: {len(table_df)} / {len(df)}"
    return json.dumps(args, indent=2), caption_md, plot_fig, heat_html, preview_md, table_df


def run_ui(*args):
    try:
        return run(*args)
    except Exception as e:
        return "{}", "### Filtered Cases: 0 / 0", None, "<div>Error</div>", f"### Error\n{type(e).__name__}: {e}", pd.DataFrame()


def initial_run(model, ollama_url, manual_json, plot_choice):
    default_request = "show top 100 by fail_score"
    return run_ui(default_request, model, ollama_url, manual_json, plot_choice)


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
                    value="show top 100 by fail_score",
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
                preview = gr.Markdown("No data yet.")
                with gr.Accordion("Optional Heat View", open=False):
                    heat_table = gr.HTML(label="Row color by FailScore")
                with gr.Accordion("Parsed Arguments JSON", open=False):
                    parsed = gr.Code(label="", language="json")
                with gr.Accordion("Plots (Last)", open=False):
                    plot_choice = gr.Dropdown(
                        label="Plot Type",
                        choices=["1", "2", "3"],
                        value="3",
                        info="1=FailScore, 2=LLM vs Regex, 3=Grouped fail by top error-tag combos",
                    )
                    plot_view = gr.Plot(label="Plot")
        run_btn.click(
            fn=run_ui,
            inputs=[request_text, model, ollama_url, manual_json, plot_choice],
            outputs=[parsed, caption, plot_view, heat_table, preview, table],
        )
        app.load(
            fn=initial_run,
            inputs=[model, ollama_url, manual_json, plot_choice],
            outputs=[parsed, caption, plot_view, heat_table, preview, table],
        )
    return app


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7864)
