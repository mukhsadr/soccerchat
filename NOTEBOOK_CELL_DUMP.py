"""
Notebook Cell Dump
Use this file as a copy source for Jupyter cells.

Workflow:
1) Open this file in VS Code.
2) Copy the CELL block you need.
3) Paste into a new Jupyter cell and run.
"""

# ==========================
# CELL: grouped_failscore_plot_v1
# ==========================
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

# CONFIG
DATA_PATH = Path(r"C:/Users/adams/Documents/Projects/soccerchat/SoccerChat_valid_xfoul_abs_preds_100_scored_v4.jsonl")
TOP_GROUPS = 3
LEGEND_LOC = "upper right"
XTICK_EVERY = 10
ONLY_DECISION_FAILED = False
MIN_FAIL_SCORE = None

# LOAD
rows = []
with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
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
    df = df[df["decision_match_num"] == 0].copy()
if MIN_FAIL_SCORE is not None:
    df = df[df["fail_score"] >= float(MIN_FAIL_SCORE)].copy()

df = df.reset_index(drop=True)


def tags_from_value(v):
    if isinstance(v, list):
        out = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        for sep in [",", ";", "|", "+"]:
            if sep in s:
                parts = [p.strip() for p in s.split(sep)]
                return [p for p in parts if p]
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


combo_list = []
for v in df.get("error_tags", pd.Series([None] * len(df))):
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

if n == 0:
    raise RuntimeError("No rows left after filtering. Relax ONLY_DECISION_FAILED / MIN_FAIL_SCORE.")

vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
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
    short_txt = f"G{i+1}\n(n={m})"
    ax.text(
        mid,
        y,
        short_txt,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
    )

    if end < n:
        ax.axvline(end - 0.5, color="#475569", linestyle="--", linewidth=1.0, alpha=0.85)

    legend_rows.append((f"G{i+1}", two_line_label(g), m))
    start = end

ax.set_title("Grouped Fail-Score Distribution (top error-tag combos)", fontsize=13, pad=12)
ax.set_xlabel("Case index (sorted within group by fail_score)", fontsize=10)
ax.set_ylabel("Fail score (higher = worse)", fontsize=10)

xt = np.arange(0, n, XTICK_EVERY)
ax.set_xticks(xt)
ax.set_xticklabels([str(i) for i in xt], fontsize=8)

ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("Fail score (low=green -> high=red)")

handles = []
for gid, gtext, m in legend_rows:
    label = f"{gid}: {gtext.replace(chr(10), ' / ')} (n={m})"
    handles.append(Patch(facecolor="none", edgecolor="none", label=label))

ax.legend(handles=handles, loc=LEGEND_LOC, frameon=True, fontsize=8, title="Group mapping")

plt.tight_layout()
plt.show()

print(f"Loaded rows: {len(df)} | plotted: {n}")
print("Top groups:", keep)
