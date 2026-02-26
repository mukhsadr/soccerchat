#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Change path if needed
JSONL_PATH = Path(r"C:/Users/adams/Documents/Projects/soccerchat/SoccerChat_valid_xfoul_abs_preds_100_scored_v2.jsonl")
SCORE_COL = "fail_score"
TOP_N = 100

rows = []
with JSONL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise RuntimeError("No rows found in JSONL")

df = pd.DataFrame(rows)
if SCORE_COL not in df.columns:
    raise ValueError(f"Column '{SCORE_COL}' not found. Columns: {list(df.columns)}")

df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
d = df.dropna(subset=[SCORE_COL]).head(TOP_N).reset_index(drop=True)
if len(d) == 0:
    raise RuntimeError(f"No numeric values in '{SCORE_COL}'")

vals = d[SCORE_COL].tolist()
mn, mx = min(vals), max(vals)
denom = (mx - mn) if mx > mn else 1.0
norm = [(v - mn) / denom for v in vals]
colors = plt.cm.RdYlGn_r(norm)  # low fail -> green, high fail -> red

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.bar(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.3)
ax.set_title(f"Fail Score for {len(vals)} Cases")
ax.set_xlabel("Case Index")
ax.set_ylabel("Fail Score")
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("Low fail  ->  High fail")

plt.tight_layout()
plt.show()
