#!/usr/bin/env python3
"""
run_plot.py

Terminal runner for grouped fail-score plot.

Just run:
    python run_plot.py

Edit CONFIG section below if needed.
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors


# ==========================
# CONFIG (EDIT HERE)
# ==========================
DATA_PATH = Path(r"C:/Users/adams/Documents/Projects/soccerchat/SoccerChat_valid_xfoul_abs_preds_100_scored_v4.jsonl")

TOP_GROUPS = 3
LEGEND_LOC = "upper right"
XTICK_EVERY = 10

ONLY_DECISION_FAILED = False
MIN_FAIL_SCORE = None

SAVE_PNG = None   # e.g. Path("fail_groups.png")  or None
# ==========================


def tags_from_value(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        for sep in [",", ";", "|", "+"]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s]
    return []


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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    print(f"Loading: {DATA_PATH}")

    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    df["fail_score"] = pd.to_numeric(df.get("fail_score"), errors="coerce")

    if "decision_match" in df.columns:
        df["decision_match_num"] = pd.to_numeric(df["decision_match"], errors="coerce")
    else:
        df["decision_match_num"] = np.nan

    if "judge_score_regex" in df.columns:
        rx = pd.to_numeric(df["judge_score_regex"], errors="coerce")
        df["decision_match_num"] = df["decision_match_num"].fillna(rx)

    df = df.dropna(subset=["fail_score"]).reset_index(drop=True)

    if ONLY_DECISION_FAILED:
        df = df[df["decision_match_num"] == 0]

    if MIN_FAIL_SCORE is not None:
        df = df[df["fail_score"] >= float(MIN_FAIL_SCORE)]

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering.")

    # Build tag combos
    combo_list = []
    for v in df.get("error_tags", []):
        tags = sorted(set(tags_from_value(v)))
        combo_list.append(" + ".join(tags) if tags else "no_tag")

    df["error_combo"] = combo_list

    combo_counts = Counter(df["error_combo"].tolist())
    keep = [k for k, _ in combo_counts.most_common(TOP_GROUPS)]
    df["combo_group"] = df["error_combo"].apply(lambda x: x if x in keep else "other")

    group_order = keep + (["other"] if (df["combo_group"] == "other").any() else [])

    parts = []
    for g in group_order:
        sub = df[df["combo_group"] == g].sort_values("fail_score", ascending=False).copy()
        sub["group"] = g
        parts.append(sub)

    plot_df = pd.concat(parts, axis=0).reset_index(drop=True)

    vals = plot_df["fail_score"].to_numpy()
    n = len(vals)

    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r
    bar_colors = cmap(norm(vals))

    fig, ax = plt.subplots(figsize=(17, 6.2))
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
            mid, y, f"G{i+1}\n(n={m})",
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        )

        if end < n:
            ax.axvline(end - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)

        legend_rows.append((f"G{i+1}", two_line_label(g), m))
        start = end

    ax.set_title("Grouped Fail-Score Distribution (top error-tag combos)")
    ax.set_xlabel("Case index (sorted within group)")
    ax.set_ylabel("Fail score (higher = worse)")

    xt = np.arange(0, n, XTICK_EVERY)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(i) for i in xt], fontsize=8)

    ax.grid(axis="y", linestyle="--", alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Fail score (green → red)")

    handles = []
    for gid, gtext, m in legend_rows:
        label = f"{gid}: {gtext.replace(chr(10), ' / ')} (n={m})"
        handles.append(Patch(facecolor="none", edgecolor="none", label=label))

    ax.legend(handles=handles, loc=LEGEND_LOC, frameon=True, fontsize=8)

    plt.tight_layout()

    if SAVE_PNG:
        fig.savefig(SAVE_PNG, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {SAVE_PNG}")

    print(f"Loaded rows: {len(df)}")
    print("Top groups:", keep)

    plt.show()


if __name__ == "__main__":
    main()