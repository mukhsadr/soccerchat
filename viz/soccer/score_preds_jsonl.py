#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import requests


def tok(s: str):
    return re.findall(r"\w+", (s or "").lower())


def token_f1(pred: str, gt: str) -> float:
    p, g = tok(pred), tok(gt)
    if not p or not g:
        return 0.0
    ps, gs = set(p), set(g)
    inter = len(ps & gs)
    if inter == 0:
        return 0.0
    prec = inter / len(ps)
    rec = inter / len(gs)
    return 2 * prec * rec / (prec + rec)


def lcs(a, b):
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def rougeL_f1(pred: str, gt: str) -> float:
    p, g = tok(pred), tok(gt)
    if not p or not g:
        return 0.0
    l = lcs(p, g)
    if l == 0:
        return 0.0
    prec = l / len(p)
    rec = l / len(g)
    return 2 * prec * rec / (prec + rec)


def detect_decision_type(question: str) -> str:
    q = (question or "").lower()
    if "foul" in q:
        return "foul"
    if "advantage" in q:
        return "advantage"
    if "card" in q:
        return "card"
    if "spa" in q or "dogso" in q:
        return "spa_dogso"
    return "other"


def extract_decision(text: str, decision_type: str) -> str:
    t = (text or "").lower()
    if decision_type in ("foul", "advantage"):
        no_pats = ["no foul", "not a foul", "no, ", "no.", "cannot", "could not", "would not", "no advantage"]
        yes_pats = ["yes", "is a foul", "foul", "advantage", "could", "can give advantage"]
        if any(p in t for p in no_pats):
            return "no"
        if any(p in t for p in yes_pats):
            return "yes"
        return "unknown"
    if decision_type == "card":
        if "red card" in t or "red" in t:
            return "red"
        if "yellow card" in t or "yellow" in t:
            return "yellow"
        if "no card" in t or "without card" in t:
            return "no_card"
        return "unknown"
    if decision_type == "spa_dogso":
        has_spa = "spa" in t or "promising attack" in t
        has_dogso = "dogso" in t or "deny" in t and "goal" in t
        if has_dogso:
            return "dogso"
        if has_spa:
            return "spa"
        if "neither" in t or "no spa" in t or "no dogso" in t:
            return "neither"
        return "unknown"
    return "unknown"


def regex_judge(question: str, gt: str, pred: str):
    d_type = detect_decision_type(question)
    gt_d = extract_decision(gt, d_type)
    pr_d = extract_decision(pred, d_type)
    match = 1 if (gt_d != "unknown" and pr_d != "unknown" and gt_d == pr_d) else 0
    return d_type, gt_d, pr_d, match


def _ollama_call_json(prompt: str, model: str, url: str, timeout_s: int, retries: int):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    last_err = None
    txt = ""
    for _ in range(max(1, retries)):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            txt = r.json().get("response", "").strip()
            break
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    if not txt and last_err is not None:
        return None, f"request_error:{type(last_err).__name__}"
    if not txt:
        return None, "empty_response"
    try:
        return json.loads(txt), None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None, f"invalid_json:{txt[:120]}"
        return json.loads(m.group(0)), None


def ollama_judge(question: str, pred: str, gt: str, model: str, url: str, timeout_s: int, retries: int):
    schema = {
        "decision_type": "foul|advantage|card|spa_dogso|other",
        "gt_decision": "yes/no/yellow/red/no_card/spa/dogso/neither/unknown",
        "pred_decision": "yes/no/yellow/red/no_card/spa/dogso/neither/unknown",
        "decision_match": 0,
        "explanation": "",
    }
    prompt = f"""
You are a strict evaluator for SoccerChat referee QA.

You will be given:
- QUESTION (the referee question)
- GROUND_TRUTH (reference answer)
- PREDICTION (model answer)

Task:
1) Identify the DECISION TYPE from the question:
   - foul (yes/no)
   - advantage (yes/no)
   - card (no card / yellow / red)
   - spa_dogso (SPA / DOGSO / neither)
   - other
2) Extract the ground-truth decision and the predicted decision.
3) Score whether the predicted decision matches the ground-truth decision.

Output ONLY valid JSON with this schema:
{json.dumps(schema, indent=2)}

Rules:
- Ignore extra explanation details unless they change the decision.
- If the decision cannot be determined, use "unknown" and decision_match = 0.
- Be conservative: only output decision_match=1 if the decisions clearly match.

QUESTION:
{question}

GROUND_TRUTH:
{gt}

PREDICTION:
{pred}
""".strip()
    obj, err = _ollama_call_json(prompt, model, url, timeout_s, retries)
    if err:
        return None, None, None, None, err
    decision_type = str(obj.get("decision_type", "")).strip() or None
    gt_decision = str(obj.get("gt_decision", "")).strip() or None
    pred_decision = str(obj.get("pred_decision", "")).strip() or None
    try:
        decision_match = int(obj.get("decision_match"))
    except Exception:
        decision_match = None
    explanation = str(obj.get("explanation", "")).strip()
    return decision_type, gt_decision, pred_decision, decision_match, explanation


def ollama_error_tag(
    question: str,
    gt: str,
    pred: str,
    decision_type,
    gt_decision,
    pred_decision,
    decision_match,
    model: str,
    url: str,
    timeout_s: int,
    retries: int,
):
    allowed_tags = [
        "wrong_decision",
        "unknown_or_vague",
        "hallucinated_details",
        "misread_contact",
        "misread_severity",
        "misread_context",
        "irrelevant_reasoning",
        "format_issue",
        "other",
    ]
    schema = {"error_tags": allowed_tags[:2], "error_summary": ""}

    prompt = f"""
You are labeling failure modes for SoccerChat referee QA.

Choose 1-3 tags from this allowed list ONLY:
{json.dumps(allowed_tags, indent=2)}

Then write ONE short sentence (<=20 words) describing the main error.

Use these signals if provided:
- decision_type={decision_type}
- gt_decision={gt_decision}
- pred_decision={pred_decision}
- decision_match={decision_match}

Output ONLY valid JSON with schema:
{json.dumps(schema, indent=2)}

QUESTION:
{question}

GROUND_TRUTH:
{gt}

PREDICTION:
{pred}
""".strip()

    obj, err = _ollama_call_json(prompt, model, url, timeout_s, retries)
    if err:
        return None, None, err

    tags = obj.get("error_tags", None)
    if not isinstance(tags, list):
        tags = None
    else:
        tags = [t for t in tags if t in allowed_tags][:3]
        if not tags:
            tags = None

    summary = obj.get("error_summary", "")
    summary = ("" if summary is None else str(summary)).strip()
    if len(summary.split()) > 30:
        summary = " ".join(summary.split()[:30])
    return tags, summary, None


def main():
    ap = argparse.ArgumentParser(description="Score SoccerChat preds JSONL with token_f1 and rougeL_f1")
    ap.add_argument("--input", default="SoccerChat_valid_xfoul_abs_preds_100.jsonl", help="Input preds jsonl (query,response,pred,video)")
    ap.add_argument("--output", default="viz/SoccerChat_valid_xfoul_abs_preds_100_scored.jsonl", help="Output scored jsonl")
    ap.add_argument("--group", default="valid_preds_100", help="group label for output rows")
    ap.add_argument("--ollama_judge", action="store_true", help="Add judge columns using Ollama per row")
    ap.add_argument("--judge_model", default="llama3.2:3b", help="Ollama model for judging")
    ap.add_argument("--judge_url", default="http://localhost:11434/api/generate", help="Ollama generate URL")
    ap.add_argument("--judge_timeout_s", type=int, default=45, help="Timeout seconds per Ollama request")
    ap.add_argument("--judge_retries", type=int, default=2, help="Retries per Ollama request")
    ap.add_argument("--ollama_error_tag", action="store_true", help="Add error_tags + error_summary per row (Ollama)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    # Pre-count lines for progress display.
    total = 0
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1

    rows = 0
    with inp.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            gt = r.get("gt") or r.get("response") or ""
            pred = r.get("pred") or ""
            video = r.get("video") or r.get("resolved_video") or (r.get("videos") or [""])[0]

            out_row = {
                "video": video,
                "query": r.get("query", ""),
                "gt": gt,
                "pred": pred,
                "group": args.group,
                "token_f1": token_f1(pred, gt),
                "rougeL_f1": rougeL_f1(pred, gt),
            }
            q = r.get("query", "")
            d_type_rx, gt_rx, pred_rx, m_rx = regex_judge(question=q, gt=gt, pred=pred)
            out_row["decision_type_regex"] = d_type_rx
            out_row["gt_decision_regex"] = gt_rx
            out_row["pred_decision_regex"] = pred_rx
            out_row["judge_score_regex"] = m_rx
            out_row["judge_score_llm"] = None
            out_row["judge_score_01"] = None
            out_row["judge_reason"] = None
            decision_type = None
            gt_decision = None
            pred_decision = None
            decision_match = None
            explanation = None
            if args.ollama_judge:
                print(f"[{rows+1}/{total}] judging...", flush=True)
                decision_type, gt_decision, pred_decision, decision_match, explanation = ollama_judge(
                    question=q,
                    pred=pred,
                    gt=gt,
                    model=args.judge_model,
                    url=args.judge_url,
                    timeout_s=args.judge_timeout_s,
                    retries=args.judge_retries,
                )
                out_row["decision_type"] = decision_type
                out_row["gt_decision"] = gt_decision
                out_row["pred_decision"] = pred_decision
                out_row["decision_match"] = decision_match
                out_row["explanation"] = explanation
                out_row["judge_score_llm"] = decision_match
                # Backward-compatible columns for existing GUI sorting.
                out_row["judge_score_01"] = decision_match
                out_row["judge_reason"] = explanation

            out_row["error_tags"] = None
            out_row["error_summary"] = None
            if args.ollama_error_tag:
                print(f"[{rows+1}/{total}] tagging...", flush=True)
                tags, summ, err = ollama_error_tag(
                    question=q,
                    gt=gt,
                    pred=pred,
                    decision_type=decision_type or out_row.get("decision_type_regex"),
                    gt_decision=gt_decision or out_row.get("gt_decision_regex"),
                    pred_decision=pred_decision or out_row.get("pred_decision_regex"),
                    decision_match=decision_match if decision_match is not None else out_row.get("judge_score_regex", 0),
                    model=args.judge_model,
                    url=args.judge_url,
                    timeout_s=args.judge_timeout_s,
                    retries=args.judge_retries,
                )
                if err:
                    out_row["error_summary"] = f"tagger_error:{err}"
                else:
                    out_row["error_tags"] = tags
                    out_row["error_summary"] = summ
            # Combined failure score: decision mismatch dominates.
            decision_for_fail = out_row.get("judge_score_llm")
            if decision_for_fail is None:
                decision_for_fail = out_row.get("judge_score_regex", 0)
            try:
                d = float(decision_for_fail)
            except Exception:
                d = 0.0
            tf1 = float(out_row.get("token_f1", 0.0) or 0.0)
            rf1 = float(out_row.get("rougeL_f1", 0.0) or 0.0)
            out_row["fail_score"] = (1.0 - d) * 2.0 + (1.0 - tf1) + (1.0 - rf1)
            w.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            w.flush()
            rows += 1
            if rows % 5 == 0 or rows == total:
                print(f"[{rows}/{total}] processed", flush=True)

    print(f"Saved {rows} rows -> {out}")


if __name__ == "__main__":
    main()
