#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_DATA = Path("SoccerChat_valid_xfoul_abs_preds_100_scored_v3.jsonl")


def clean_question(q: str) -> str:
    q = re.sub(r"<\s*video\s*>", "", str(q or ""), flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


@st.cache_data
def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "query" in df.columns:
        df["question"] = df["query"].astype(str).apply(clean_question)
    return df


def row_color(row: pd.Series):
    v = pd.to_numeric(row.get("fail_score", None), errors="coerce")
    if pd.isna(v):
        return [""] * len(row)
    if v <= 1.0:
        c = "#ecfdf5"
    elif v <= 2.0:
        c = "#fefce8"
    elif v <= 3.0:
        c = "#fff7ed"
    else:
        c = "#fef2f2"
    return [f"background-color: {c}"] * len(row)


def main():
    st.set_page_config(page_title="SoccerChat Viewer", layout="wide")
    st.title("SoccerChat Viewer")

    col1, col2 = st.columns([3, 2])
    with col1:
        data_path = st.text_input("Data file", value=str(DEFAULT_DATA))
    with col2:
        top_n = st.slider("Top N", 1, 200, 100)

    df = load_jsonl(data_path)

    sort_choices = [c for c in ["fail_score", "judge_score_01", "judge_score_llm", "judge_score_regex", "token_f1", "rougeL_f1"] if c in df.columns]
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        sort_by = st.selectbox("Sort by", options=sort_choices, index=0 if sort_choices else None)
    with c2:
        ascending = st.checkbox("Ascending", value=False)
    with c3:
        decision_types = ["(all)"] + sorted(df["decision_type_regex"].dropna().astype(str).unique().tolist()) if "decision_type_regex" in df.columns else ["(all)"]
        decision_filter = st.selectbox("Decision type", decision_types, index=0)
    with c4:
        q_filter = st.text_input("Contains text", value="")

    out = df.copy()
    if q_filter:
        q = q_filter.lower()
        cols = [c for c in ["question", "gt", "pred"] if c in out.columns]
        mask = False
        for c in cols:
            mask = mask | out[c].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]

    if decision_filter != "(all)" and "decision_type_regex" in out.columns:
        out = out[out["decision_type_regex"].astype(str) == decision_filter]

    if sort_by in out.columns:
        out[sort_by] = pd.to_numeric(out[sort_by], errors="coerce")
        out = out.sort_values(sort_by, ascending=ascending, na_position="last")

    out = out.head(top_n).reset_index(drop=True)
    st.markdown(f"### Filtered Cases: {len(out)} / {len(df)}")

    show_cols = [c for c in ["question", "gt", "pred", "judge_score_llm", "judge_score_regex", "judge_score_01", "fail_score", "token_f1", "rougeL_f1"] if c in out.columns]
    view = out[show_cols].copy()
    for c in ["fail_score", "token_f1", "rougeL_f1"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(4)

    st.dataframe(view.style.apply(row_color, axis=1), use_container_width=True, height=520)

    with st.expander("Top 3 Readable", expanded=False):
        for i, r in view.head(3).iterrows():
            st.markdown(f"**Row {i+1}**")
            st.write("Question:", r.get("question", ""))
            st.write("GT:", r.get("gt", ""))
            st.write("Pred:", r.get("pred", ""))
            st.write("Scores:", {k: r.get(k, None) for k in ["judge_score_llm", "judge_score_regex", "fail_score", "token_f1", "rougeL_f1"]})


if __name__ == "__main__":
    main()

