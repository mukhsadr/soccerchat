# SoccerChat Project Map

## Run This First (100 + LLM Scoring)

```powershell
cd C:\Users\adams\Documents\Projects\soccerchat
python .\run_score_100_llm.py
```

Input:
- `SoccerChat_valid_xfoul_abs_preds_100.jsonl`

Output:
- `viz/SoccerChat_valid_xfoul_abs_preds_100_scored_v2.jsonl`

---

## GUI 1: Main Table (Q / GT / Pred)

```powershell
cd C:\Users\adams\Documents\Projects\soccerchat
python .\run_soccer_table_gui.py
```

Default port: `7863`

---

## GUI 2: JSON Argument Extractor + Plot

```powershell
cd C:\Users\adams\Documents\Projects\soccerchat
python .\run_json_extractor_gui.py
```

Default port: `7864`

---

## Core Code Locations

- Scoring logic (token/rouge + regex + llm + fail score):
  - `viz/soccer/score_preds_jsonl.py`
- Main evaluation table GUI:
  - `viz/soccer_app_gui.py`
- JSON argument extractor GUI:
  - `viz/json_argument_extractor_gui.py`

