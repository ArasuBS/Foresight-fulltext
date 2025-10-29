# stage4_summarize.py
# Foresight (Full-Text) — Stage 4: Summarize & Actionables
# Input: sections.csv (from Stage 3 ZIP)  |  Output: paper_summaries.csv, report.md, stage4_bundle.zip

import re, io, zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta

APP = "Foresight Full-Text — Stage 4 (Summarize)"
IST = timezone(timedelta(hours=5, minutes=30))

# -------------------------- tiny utils --------------------------
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

SEC_PRIOR = [
    r"\babstract\b", r"\bintroduction\b", r"\bbackground\b",
    r"\bresults?\b", r"\bdiscussion\b", r"\bconclusion(s)?\b",
    r"\bmaterials?\s+and\s+methods?\b", r"\bmethods?\b"
]
SEC_PRIOR_RE = [re.compile(p, re.I) for p in SEC_PRIOR]

METHOD_HINTS = re.compile(
    r"(conjugation|site[-\s]?specific|thiol|amine|maleimide|val[-\s]?cit|vc[-\s]?pabc|spaac|dbco|tetrazine|tco|"
    r"transglutaminase|sortase|glycan|hydrazone|disulfide|non[-\s]?cleavable|qc|hplc|sec[-\s]?mals|msd|bli|dar|hic|tff|"
    r"scale[-\s]?up|gmp|cmc|qbd|pat|hpapi|containment)", re.I)

SIGNALS = re.compile(
    r"(her2|trop[-\s]?2|nectin[-\s]?4|cldn18\.?2|egfr|her3|5t4|frα|fr[ -]?alpha|"
    r"mmae|dm1|dm4|sn[-\s]?38|pbd|duocarmycin|auristatin)", re.I)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def pick_sections(df_paper):
    # prefer specific sections by title; fallback to longest text
    sec_texts = []
    for _, r in df_paper.iterrows():
        title = str(r.get("section_title",""))
        text  = str(r.get("text",""))
        score = 0
        for w, rx in enumerate(SEC_PRIOR_RE):
            if rx.search(title):
                score = max(score, 100 - 5*w)
        sec_texts.append((score, len(text), title, text))
    if not sec_texts:
        return ""
    sec_texts.sort(reverse=True)  # by score then length
    # take top 4 chunks
    return " ".join([t[3] for t in sec_texts[:4]])

def bulletize(text, n=7):
    sents = SENT_SPLIT.split(clean(text))
    # quick de-dup and filter
    uniq = []
    seen = set()
    for s in sents:
        ss = clean(s)
        if len(ss) < 40: continue
        key = ss.lower()[:120]
        if key in seen: continue
        seen.add(key)
        uniq.append(ss)
        if len(uniq) >= n: break
    return ["• " + u for u in uniq]

def extract_methods(text):
    lines = []
    for sent in SENT_SPLIT.split(clean(text)):
        if METHOD_HINTS.search(sent):
            lines.append("– " + clean(sent))
        if len(lines) >= 8: break
    return lines

def extract_signals(text):
    hits = sorted(set(m.group(0).upper() for m in SIGNALS.finditer(text)))
    return ", ".join(hits) if hits else ""

def now_ist_stamp():
    return datetime.now(IST).strftime("%Y-%m-%d_%H%M%S_IST")

# ------------------------------ UI ------------------------------
st.set_page_config(page_title=APP, layout="wide")
st.title(APP)
st.caption("Reads sections.csv from Stage 3 and produces a human-friendly summary report + a single ZIP bundle.")

# Previous run fast downloads
prev = st.session_state.get("stage4_store")
if prev:
    with st.expander("⬇️ Previous run — quick downloads", expanded=False):
        st.download_button("Report (Markdown)", data=prev["report_md"], file_name=prev["report_name"], mime="text/markdown")
        st.download_button("Paper summaries (CSV)", data=prev["summ_csv"], file_name="paper_summaries.csv", mime="text/csv")
        st.download_button("Download EVERYTHING (ZIP)", data=prev["bundle_zip"], file_name=prev["bundle_name"], mime="application/zip")

col1, col2 = st.columns([2,1])
with col1:
    sec_csv = st.file_uploader("Upload sections.csv (from Stage 3)", type=["csv"])
    path_text = st.text_input("Or path to sections.csv (optional)", value="")
with col2:
    max_papers = st.number_input("Max papers to summarize (K)", 1, 500, 30, 1)
    include_methods = st.checkbox("Include Methods bullets", value=True)

run = st.button("Summarize")

if not run:
    st.info("Upload your Stage-3 sections.csv and click **Summarize**.")
    st.stop()

# --------------------------- load ---------------------------
try:
    if sec_csv is not None:
        df = pd.read_csv(sec_csv)
    elif path_text.strip():
        df = pd.read_csv(path_text.strip())
    else:
        st.error("Please provide sections.csv."); st.stop()
except Exception as e:
    st.error(f"Failed to read sections.csv: {e}"); st.stop()

need_cols = {"PMCID","PMID","DOI","Title","section_title","text"}
miss = [c for c in need_cols if c not in df.columns]
if miss:
    st.error(f"sections.csv missing columns: {miss}"); st.stop()

# ------------------------ group & summarize ------------------------
# keep a stable paper key
df["paper_key"] = df["PMCID"].fillna("").astype(str)
empty_mask = df["paper_key"].eq("") & df["DOI"].notna()
df.loc[empty_mask, "paper_key"] = "DOI:" + df.loc[empty_mask, "DOI"].astype(str)

orders = list(dict.fromkeys(df["paper_key"].tolist()))
summ_rows = []
stamp = now_ist_stamp()

for pk in orders[:int(max_papers)]:
    sub = df[df["paper_key"] == pk]
    if sub.empty: continue
    title = sub["Title"].dropna().astype(str).unique()
    title = title[0] if len(title) else ""

    merged = pick_sections(sub)
    bullets = bulletize(merged, n=7)
    meth = extract_methods(merged) if include_methods else []
    sigs = extract_signals(merged)

    summ_rows.append({
        "paper_key": pk,
        "Title": title,
        "Signals": sigs,
        "SummaryBullets": "\n".join(bullets),
        "MethodsBullets": "\n".join(meth)
    })

summ_df = pd.DataFrame(summ_rows)

# ------------------------ compose report.md ------------------------
lines = []
lines.append(f"# ADC Full-Text Summary Report ({stamp})")
lines.append("")
lines.append(f"Total papers summarized: {len(summ_df)}")
lines.append("")
for i, r in enumerate(summ_df.itertuples(), start=1):
    lines.append(f"## {i}. {r.Title or r.paper_key}")
    if r.Signals:
        lines.append(f"**Signals/Entities:** {r.Signals}")
    if r.SummaryBullets:
        lines.append("**What’s new / Key points**")
        lines.extend(r.SummaryBullets.splitlines())
    if include_methods and r.MethodsBullets:
        lines.append("**Methods & Process notes**")
        lines.extend(r.MethodsBullets.splitlines())
    lines.append("")

report_md = ("\n".join(lines)).encode("utf-8")
summ_csv = summ_df.to_csv(index=False).encode("utf-8")

# ------------------------ bundle + sticky downloads ------------------------
bundle_name = f"stage4_summary_bundle_{stamp}.zip"
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("report.md", report_md)
    zf.writestr("paper_summaries.csv", summ_df.to_csv(index=False))

# persist in session so downloads don’t vanish on rerun
st.session_state["stage4_store"] = {
    "report_md": report_md,
    "report_name": "report.md",
    "summ_csv": summ_csv,
    "bundle_zip": buf.getvalue(),
    "bundle_name": bundle_name,
}

st.success("Summaries generated.")
st.download_button("Report (Markdown)", data=report_md, file_name="report.md", mime="text/markdown")
st.download_button("Paper summaries (CSV)", data=summ_csv, file_name="paper_summaries.csv", mime="text/csv")
st.download_button("Download EVERYTHING (ZIP)", data=st.session_state["stage4_store"]["bundle_zip"],
                   file_name=bundle_name, mime="application/zip")
