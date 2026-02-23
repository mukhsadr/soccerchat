# SoccerChat

## Description
User can ask uploaded-soccer-game-video-clip.

Description from Visual Language Model fine-tuned for soccer data.

High risk item/feature: Assessing the model's performance by computing agreement scores.

## Test Run Images

Test run first: we ran test va, which predicts the next token.

![Test validation next-token plot](test%20va.png)

![Test run output](testrun.png)

## De-Risking Checklist (Dr. Landman Style)

- [x] Model runs end-to-end on a small sample
- [x] Inference on test split produces outputs without errors
- [x] Basic evaluation metrics computed (e.g., token F1, ROUGE-L)
- [x] Sanity check: F1 scoring = sementic/token overlapping/matching pred vs gt
- [x] Likert Scaling: 1=StrDsgr .. 5=StrAgr
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
