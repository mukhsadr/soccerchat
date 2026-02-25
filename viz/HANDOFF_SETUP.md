# SoccerChat Handoff + Build Instructions

This file is for another LLM assistant or teammate to continue work quickly.

## Project Goal

Build a practical SoccerChat evaluation app that:

1. Loads scored evaluation data (`.jsonl` or `.csv`)
2. Filters/ranks examples
3. Shows prediction quality (metrics + text + video)
4. Exports worst/failing cases for error analysis

---

## Current Files

- `README.md`: project summary, risks, checklist
- `soccer_app_gui.py`: Gradio app for eval browsing
- `test va.png`, `testrun.png`: report images

---

## Environment Setup

Use Python 3.10+ (3.11/3.12 OK).

```bash
cd /mnt/c/Users/adams/Documents/Projects/soccerchat
python -m venv .venv
source .venv/bin/activate
pip install gradio pandas matplotlib requests
```

If using Ollama parsing:

```bash
ollama list
ollama pull llama3.2:3b
```

---

## Run the Soccer Eval GUI

```bash
cd /mnt/c/Users/adams/Documents/Projects/soccerchat
source .venv/bin/activate
python viz/soccer_app_gui.py
```

Open:

- `http://127.0.0.1:7862`

---

## Data Expectations

Supported input:

- `.jsonl` (one JSON object per line)
- `.csv`

Useful columns (if present):

- `video`
- `video_file`
- `group`
- `token_f1`
- `rougeL_f1`
- `judge_score_01`
- `judge_score_cont` or `judge_score_llm`
- `query`
- `pred`
- `gt`

---

## Next Build Target (Important)

We want a **soccer test pre-extract failure workflow**:

1. Run on test split
2. Auto-rank worst examples (low F1/ROUGE, judge mismatch)
3. Export failing cases to CSV/JSON for review slides
4. Add error tags (missed foul, wrong card, hallucination, timing, vague)

Suggested output files:

- `reports/fail_cases_top100.csv`
- `reports/fail_cases_tag_template.csv`
- `reports/fail_summary.md`

---

## Implementation Plan for Fail-Case Extractor

Create file: `soccer_fail_extract.py`

Inputs:

- `--data score.jsonl`
- `--group test`
- `--top_n 100`
- `--sort_by token_f1`
- `--ascending true`

Logic:

1. Load dataframe
2. Keep `group == test` if column exists
3. Build `fail_score` (example):
   - `1 - token_f1`
   - `1 - rougeL_f1`
   - `(judge_score_01 == 0)` penalty
4. Sort by `fail_score` desc
5. Save top N with key fields:
   - `video`, `query`, `pred`, `gt`, `token_f1`, `rougeL_f1`, `judge_score_01`, `fail_score`

---

## Suggested UI Upgrade

For `soccer_app_gui.py`:

1. Add button: `Export current filtered rows`
2. Add button: `Generate fail-case report`
3. Add chart tab: histogram of `token_f1`
4. Add row selector to preview any row (not only first row)

---

## ACCRE Notes

Typical run:

```bash
python app5_dynamic.py --port 7860 --allowed_video_root /nobackup/user/sadridm/Soccer/Data/SoccerChat/videos/xfoul-valid
```

Tunnel:

```bash
ssh -N -L 7860:localhost:7860 sadridm@login.accre.vu
```

---

## Git Workflow

```bash
git add .
git commit -m "Add soccer eval GUI and handoff build plan"
git push
```

