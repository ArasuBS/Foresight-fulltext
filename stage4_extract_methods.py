# stage4_extract_methods.py
# Foresight (Full-Text) — Stage 4: Methods extraction
# Input: sections.csv (+ optional tables.csv, figures.csv) OR Stage-3 ZIP
# Output: methods_sections.csv, methods_tables.csv, methods_figures.csv + one ZIP bundle
#
# Notes:
# - No 'manifest' anywhere (fixes your NameError)
# - Robust uploads: CSVs or Stage-3 ZIP; ZIP auto-fills any missing CSVs
# - Keeps results in st.session_state to avoid vanishing after downloads
# - One-click "Download EVERYTHING (ZIP)" with IST timestamped filename

import io, re, zipfile
from pathlib import Path
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py3.9+
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None

import pandas as pd
import streamlit as st

APP_TITLE = "Foresight Full-Text — Stage 4 (Extract Methods)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Loads Stage-3 outputs → finds Methods-like sections/tables/fig captions → exports CSVs + one ZIP bundle")

# ---------------------- Previous run quick download ----------------------
prev = st.session_state.get("stage4_store")
if prev:
    with st.expander("⬇️ Previous run — quick downloads", expanded=False):
        st.download_button("Download EVERYTHING (ZIP)",
                           data=prev["bundle_zip"],
                           file_name=prev["bundle_name"],
                           mime="application/zip",
                           key="prev_bundle")
        st.download_button("methods_sections.csv",
                           data=prev["sec_csv"],
                           file_name="methods_sections.csv",
                           mime="text/csv",
                           key="prev_sec")
        if prev.get("tbl_csv"):
            st.download_button("methods_tables.csv",
                               data=prev["tbl_csv"],
                               file_name="methods_tables.csv",
                               mime="text/csv",
                               key="prev_tbl")
        if prev.get("fig_csv"):
            st.download_button("methods_figures.csv",
                               data=prev["fig_csv"],
                               file_name="methods_figures.csv",
                               mime="text/csv",
                               key="prev_fig")

# ----------------------------- Controls -----------------------------
col1, col2 = st.columns([2,1])
with col1:
    sec_file = st.file_uploader("Upload sections.csv", type=["csv"])
    tbl_file = st.file_uploader("Upload tables.csv (optional)", type=["csv"])
    fig_file = st.file_uploader("Upload figures.csv (optional)", type=["csv"])
    zip_stage3 = st.file_uploader("Or upload Stage-3 bundle ZIP", type=["zip"])
with col2:
    max_rows = st.number_input("Max rows to scan (per file)", min_value=100, max_value=200000, value=50000, step=1000)
    include_tables = st.checkbox("Include tables", value=True)
    include_figs = st.checkbox("Include figure captions", value=True)

run = st.button("Extract Methods")

if not run:
    st.info("Upload sections.csv (and optionally tables/figures CSVs or a Stage-3 ZIP), then click **Extract Methods**.")
    st.stop()

# ----------------------------- Load inputs -----------------------------
def _read_csv_upload(f):
    if f is None: return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None

sections_df = _read_csv_upload(sec_file)
tables_df   = _read_csv_upload(tbl_file)
figures_df  = _read_csv_upload(fig_file)

# If ZIP is provided, fill missing ones from inside the ZIP
if zip_stage3 is not None:
    try:
        with zipfile.ZipFile(zip_stage3) as zf:
            if sections_df is None and "sections.csv" in zf.namelist():
                sections_df = pd.read_csv(io.BytesIO(zf.read("sections.csv")))
            if include_tables and tables_df is None and "tables.csv" in zf.namelist():
                tables_df = pd.read_csv(io.BytesIO(zf.read("tables.csv")))
            if include_figs and figures_df is None and "figures.csv" in zf.namelist():
                figures_df = pd.read_csv(io.BytesIO(zf.read("figures.csv")))
    except Exception as e:
        st.error(f"Stage-3 ZIP read error: {e}")
        st.stop()

if sections_df is None:
    st.error("sections.csv is required (direct upload or inside Stage-3 ZIP).")
    st.stop()

# Truncate for safety
if len(sections_df) > max_rows: sections_df = sections_df.head(max_rows)
if include_tables and tables_df is not None and len(tables_df) > max_rows: tables_df = tables_df.head(max_rows)
if include_figs and figures_df is not None and len(figures_df) > max_rows: figures_df = figures_df.head(max_rows)

# ------------------------ Methods heuristics ------------------------
# 1) Headings likely to mark Methods
METHOD_HEAD_PAT = re.compile(
    r"""
    \b(
       materials?\s*(and|&)\s*methods? |
       methods? |
       methodology |
       experimental(\s+procedures?)? |
       reagents? |
       (cell|cellular)\s+culture |
       statistical\s+analysis |
       animals?\s*(and|&)\s*(ethics|welfare)? |
       sample\s+preparation |
       data\s+analysis |
       protocol(s)? |
       instrumentation |
       chromatography |
       purification |
       conjugation |
       synthesis |
       analytics? |
       assay\s+procedures? |
       (mass|liquid)\s+spectrometry |
       hplc|uplc|sec[-\s]?mals? | hic | tff
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)

# 2) Verb/keyword cues inside text
METHOD_TEXT_CUES = re.compile(
    r"(conjugat|synthes|prepare|incubat|purif|analy[sz]e|measure|assay|"
    r"buffer|reagent|centrifug|chromatograph|gradient|flow\s*rate|"
    r"instrument|parameter|protocol|step|mix|dilut|pH\s*\d|"
    r"transglutaminase|sortase|maleimide|val[-\s]?cit|vc[-\s]?pabc|dar\b|"
    r"msd|bli|elisa|hplc|uplc|sec[-\s]?mals?|hic|tff|hpapi|containment)",
    re.IGNORECASE
)

# 3) Table caption cues
TABLE_CUES = re.compile(
    r"(reagent|buffer|composition|parameters?|settings?|antibody|linker|payload|"
    r"conjugation|dar\b|chromatograph|hplc|uplc|sec|hic|tff|condition|concentration)",
    re.IGNORECASE
)

# 4) Figure caption cues (optional)
FIG_CUES = re.compile(
    r"(workflow|scheme|synthe|setup|apparatus|protocol|conjugation|"
    r"chromatograph|hplc|dar\b)",
    re.IGNORECASE
)

def _clean(s):
    return re.sub(r"\s+", " ", str(s or "").strip())

# ------------------------ Filter sections ------------------------
sec_cols = [c for c in ["PMCID","PMID","DOI","Title","source","section_id","section_title","text","level"] if c in sections_df.columns]
sdf = sections_df[sec_cols].copy()

# mark by heading and/or text cues
sdf["__by_heading"] = sdf.get("section_title","").astype(str).str.contains(METHOD_HEAD_PAT)
sdf["__by_text"]    = sdf.get("text","").astype(str).str.contains(METHOD_TEXT_CUES)

methods_sections = sdf[(sdf["__by_heading"]) | (sdf["__by_text"])].copy()
for c in ["__by_heading","__by_text"]:
    if c in methods_sections.columns: methods_sections.drop(columns=[c], inplace=True)

# ------------------------ Filter tables (optional) ------------------------
methods_tables = pd.DataFrame()
if include_tables and tables_df is not None and not tables_df.empty:
    tcols = [c for c in ["PMCID","PMID","DOI","Title","source","table_id","label","caption","html"] if c in tables_df.columns]
    tdf = tables_df[tcols].copy()
    tdf["__hit"] = tdf.get("caption","").astype(str).str.contains(TABLE_CUES) | tdf.get("label","").astype(str).str.contains(TABLE_CUES)
    methods_tables = tdf[tdf["__hit"]].drop(columns=["__hit"])

# ------------------------ Filter figures (optional) ------------------------
methods_figures = pd.DataFrame()
if include_figs and figures_df is not None and not figures_df.empty:
    fcols = [c for c in ["PMCID","PMID","DOI","Title","source","figure_id","label","caption"] if c in figures_df.columns]
    fdf = figures_df[fcols].copy()
    fdf["__hit"] = fdf.get("caption","").astype(str).str.contains(FIG_CUES) | fdf.get("label","").astype(str).str.contains(FIG_CUES)
    methods_figures = fdf[fdf["__hit"]].drop(columns=["__hit"])

# ------------------------ Previews ------------------------
st.subheader("Methods sections (preview)")
st.dataframe(methods_sections.head(20), use_container_width=True, height=300)

if not methods_tables.empty:
    st.subheader("Methods tables (preview)")
    st.dataframe(methods_tables.head(10), use_container_width=True, height=260)

if not methods_figures.empty:
    st.subheader("Methods figure captions (preview)")
    st.dataframe(methods_figures.head(10), use_container_width=True, height=260)

# ------------------------ Exports + single ZIP ------------------------
sec_csv = methods_sections.to_csv(index=False).encode("utf-8")
tbl_csv = methods_tables.to_csv(index=False).encode("utf-8") if not methods_tables.empty else None
fig_csv = methods_figures.to_csv(index=False).encode("utf-8") if not methods_figures.empty else None

# timestamped bundle name (IST if available)
dt = datetime.now(tz=IST) if IST else datetime.now()
stamp = dt.strftime("%Y%m%d_%H%M%S_IST") if IST else dt.strftime("%Y%m%d_%H%M%S")
bundle_name = f"stage4_methods_bundle_{stamp}.zip"

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("methods_sections.csv", sec_csv)
    if tbl_csv is not None: zf.writestr("methods_tables.csv", tbl_csv)
    if fig_csv is not None: zf.writestr("methods_figures.csv", fig_csv)
bundle_zip = buf.getvalue()

# persist across reruns
st.session_state.stage4_store = {
    "bundle_zip": bundle_zip,
    "bundle_name": bundle_name,
    "sec_csv": sec_csv,
    "tbl_csv": tbl_csv,
    "fig_csv": fig_csv,
}

# buttons
st.download_button("Download EVERYTHING (ZIP)", data=bundle_zip, file_name=bundle_name, mime="application/zip", key="dl_all_zip")
st.download_button("methods_sections.csv", data=sec_csv, file_name="methods_sections.csv", mime="text/csv", key="dl_sec")
if tbl_csv is not None:
    st.download_button("methods_tables.csv", data=tbl_csv, file_name="methods_tables.csv", mime="text/csv", key="dl_tbl")
if fig_csv is not None:
    st.download_button("methods_figures.csv", data=fig_csv, file_name="methods_figures.csv", mime="text/csv", key="dl_fig")

st.markdown("---")
st.caption("Heuristics: title regex + method verbs for sections; caption cues for tables/figures. Adjust patterns in METHOD_HEAD_PAT / METHOD_TEXT_CUES / TABLE_CUES / FIG_CUES as needed.")
