# SoccerChat

## Description
SoccerChat lets users ask questions about uploaded soccer game video clips.

Target audience: soccer fans and broadcasters who want extra info or description.

It uses a Visual Language Model fine-tuned for soccer data.

High-risk item/feature: assessing model performance by computing agreement scores.

## Audience & Value

### Sports Media & Broadcast Companies
- Reduce manual annotation cost.
- Accelerate highlight publishing with a fine-tuned, domain-specific VLM trained on soccer events.

### Soccer Analytics Companies
- Bridge structured tracking data with explainable multimodal reasoning.

### AI Research Community (Academic)
- Demonstrate domain adaptation of a 7B multimodal foundation model to a specialized soccer reasoning task using about 85k examples and LoRA.

### Primary Audience (High Value)
- Sports media and broadcast companies.

## Test Run Images

Test run first: we ran `test va`, which predicts the next token.

![Test validation next-token plot](test%20va.png)

## Error Analysis (Start Here)

- Review the 20 worst examples (lowest F1/ROUGE-L or judge score)
- Tag each error type (missed action, wrong actor, hallucinated event, too vague/too long, timing error)
- Count the most frequent error types
- Summarize 2–3 findings for improvement

![Test run output](testrun.png)

## De-Risking Checklist (Dr. Landman Style)

- [x] Model runs end-to-end on a small sample
- [x] Inference on test split produces outputs without errors
- [x] Basic evaluation metrics computed (e.g., token F1, ROUGE-L)
- [x] Sanity check: F1 scoring = semantic/token overlap/matching pred vs GT
- [x] Likert scaling: 1 = Strongly Disagree ... 5 = Strongly Agree
- [x] GUI loads and accepts video input
- [ ] Full test split evaluation completed
- [ ] Error analysis on worst cases
- [ ] Performance/latency profiling

## Risk Management Table

| Feature / Capability | Difficulty (1–5) | Risk (1–5) | Status |
|---|---:|---:|---|
| Run model end-to-end | 2 | 3 | ✅ De-risked |
| Test split inference + evaluation | 3 | 4 | ✅ De-risked |
| Agreement scores for evaluation | 4 | 5 | ⚠️ High risk |
| GUI for video + chat | 3 | 3 | ✅ De-risked |
| Large-scale evaluation | 4 | 4 | ⏳ Pending |
| Error analysis & mitigation | 3 | 4 | ⏳ Pending |

## Eval GUI (New)

Run:

```bash
python run_soccer_table_gui.py
```

Open:

- `http://127.0.0.1:7862`

Handoff/build notes for another assistant:

- `viz/HANDOFF_SETUP.md`

## Score 100-Pred File

If you start from raw prediction file (`response` + `pred`), create scored JSONL first:

```bash
python viz/soccer/score_preds_jsonl.py \
  --input SoccerChat_valid_xfoul_abs_preds_100.jsonl \
  --output viz/SoccerChat_valid_xfoul_abs_preds_100_scored.jsonl \
  --group valid_preds_100
```

One-command run (100 + LLM judge + v2 output):

```bash
python run_score_100_llm.py
```

Then open GUI and use:

- `viz/SoccerChat_valid_xfoul_abs_preds_100_scored.jsonl`
- This is the default file expected by `viz/soccer_app_gui.py`

## JSON Argument Extractor GUI

```bash
python run_json_extractor_gui.py
```

Project run order and file map:

- `PROJECT_MAP.md`
