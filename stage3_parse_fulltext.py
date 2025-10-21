# stage3_parse_fulltext.py
# Foresight (Full-Text) — Stage 3: Parse PMC XML (preferred) + Publisher HTML (fallback)
# Input: stage2_manifest.csv
# Output: sections.csv, tables.csv, figures.csv (+ zipped bundle)
#
# Notes:
# - Prefers PMC JATS XML (pmc_xml/PMCID.xml)
# - Falls back to publisher HTML (publisher_oa/*.html) if no XML
# - Skips PDFs for now (OCR planned for Stage 4)
# - Lightweight deps only; BeautifulSoup is optional

import os, re, io, json, zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

from xml.etree import ElementTree as ET

# Optional HTML parsing
try:
    from bs4 import BeautifulSoup  # type: ignore
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

APP_TITLE = "Foresight Full-Text — Stage 3 (Parse & Extract)"
XML_DIR = Path("pmc_xml")
PUB_DIR = Path("publisher_oa")
CACHE_DIR = Path(".cache/parsed"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- utils --------------------------
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "")

def _ns_map(root: ET.Element) -> Dict[str, str]:
    # Rough namespace gatherer
    ns = {}
    for k, v in root.attrib.items():
        if k.startswith("xmlns:"):
            ns[k.split(":",1)[1]] = v
    # Common fallbacks
    ns.setdefault("xlink", "http://www.w3.org/1999/xlink")
    ns.setdefault("mml", "http://www.w3.org/1998/Math/MathML")
    ns.setdefault("xmlns", root.tag.split("}")[0].strip("{") if "}" in root.tag else "")
    return ns

def _serialize_element(el: ET.Element) -> str:
    try:
        return ET.tostring(el, encoding="unicode")
    except Exception:
        return ""

# -------------------- PMC XML parsing -----------------------
def parse_pmc_xml(xml_text: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Returns: (sections, tables, figures)
    sections: {section_id, section_title, text, level}
    tables:   {table_id, label, caption, html}
    figures:  {figure_id, label, caption}
    """
    sections, tables, figures = [], [], []
    if not xml_text.strip().startswith("<"):
        return sections, tables, figures

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return sections, tables, figures

    ns = _ns_map(root)

    # Title can be elsewhere; we capture section structure under <body>
    body = None
    for tag in ("body", "{*}body"):
        body = root.find(".//" + tag)
        if body is not None:
            break

    # -------- sections (JATS: <sec>) --------
    def walk_secs(node: ET.Element, level: int = 1, counter=[0]):
        for sec in list(node.findall("./sec")) + list(node.findall("./{*}sec")):
            counter[0] += 1
            sid = sec.get("id") or f"sec{counter[0]}"
            title_el = sec.find("./title") or sec.find("./{*}title")
            title = _clean_ws("".join(title_el.itertext())) if title_el is not None else ""
            # Gather text paragraphs (<p>, nested text)
            paras = []
            for p in list(sec.findall(".//p")) + list(sec.findall(".//{*}p")):
                paras.append(_clean_ws("".join(p.itertext())))
            text = _clean_ws(" ".join(paras))
            if title or text:
                sections.append({
                    "section_id": sid,
                    "section_title": title,
                    "text": text,
                    "level": level
                })
            # recurse
            walk_secs(sec, level+1)
    if body is not None:
        walk_secs(body, 1)

    # -------- tables (JATS: <table-wrap>) --------
    for tw in list(root.findall(".//table-wrap")) + list(root.findall(".//{*}table-wrap")):
        tid = tw.get("id") or ""
        label_el = tw.find("./label") or tw.find("./{*}label")
        cap_el = tw.find("./caption") or tw.find("./{*}caption")
        label = _clean_ws("".join(label_el.itertext())) if label_el is not None else ""
        caption = _clean_ws("".join(cap_el.itertext())) if cap_el is not None else ""
        # inner table element (often XHTML namespaced)
        table_el = None
        # prefer <table> in any ns
        for cand in list(tw.findall(".//table")) + list(tw.findall(".//{*}table")):
            table_el = cand; break
        table_html = _serialize_element(table_el) if table_el is not None else ""
        tables.append({
            "table_id": tid or "",
            "label": label,
            "caption": caption,
            "html": table_html
        })

    # -------- figures (JATS: <fig>) --------
    for fg in list(root.findall(".//fig")) + list(root.findall(".//{*}fig")):
        fid = fg.get("id") or ""
        label_el = fg.find("./label") or fg.find("./{*}label")
        cap_el = fg.find("./caption") or fg.find("./{*}caption")
        label = _clean_ws("".join(label_el.itertext())) if label_el is not None else ""
        caption = _clean_ws("".join(cap_el.itertext())) if cap_el is not None else ""
        if label or caption:
            figures.append({
                "figure_id": fid or "",
                "label": label,
                "caption": caption
            })
    return sections, tables, figures

# ----------------- Publisher HTML parsing --------------------
def parse_publisher_html(html_text: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Fallback, structure is heuristic. We try to build sections from H1/H2/H3 and paragraphs.
    Returns (sections, tables, figures) similar to XML path.
    """
    sections, tables, figures = [], [], []
    if not html_text:
        return sections, tables, figures

    if HAVE_BS4:
        soup = BeautifulSoup(html_text, "html.parser")
        # Sections: h1/h2/h3 + following <p> until next header
        headers = soup.find_all(["h1","h2","h3"])
        sid = 0
        for h in headers:
            sid += 1
            title = _clean_ws(h.get_text(" "))
            level = {"h1":1,"h2":2,"h3":3}.get(h.name.lower(), 2)
            # collect sibling paragraphs until next header
            text_parts = []
            for sib in h.find_all_next():
                if sib.name in ["h1","h2","h3"]:
                    break
                if sib.name == "p":
                    text_parts.append(_clean_ws(sib.get_text(" ")))
            if title or text_parts:
                sections.append({
                    "section_id": f"sec{sid}",
                    "section_title": title,
                    "text": _clean_ws(" ".join(text_parts)),
                    "level": level
                })
        # Tables
        for i, tbl in enumerate(soup.find_all("table"), start=1):
            # caption may be <caption> or neighboring sibling with 'caption' class
            cap = ""
            cap_el = tbl.find("caption")
            if cap_el: cap = _clean_ws(cap_el.get_text(" "))
            tables.append({
                "table_id": f"html_table_{i}",
                "label": "",
                "caption": cap,
                "html": str(tbl)
            })
        # Figures: we can only pull alt text / nearby captions heuristically
        for i, fig in enumerate(soup.find_all(["figure"]), start=1):
            lab = fig.find(["span","div"], class_=re.compile("label|num|number", re.I))
            cap = fig.find(["figcaption","div","p"], class_=re.compile("caption", re.I)) or fig.find("figcaption")
            figures.append({
                "figure_id": f"html_fig_{i}",
                "label": _clean_ws(lab.get_text(" ")) if lab else "",
                "caption": _clean_ws(cap.get_text(" ")) if cap else ""
            })
    else:
        # Minimal fallback: strip tags, no structure
        text = _clean_ws(_strip_tags(html_text))
        if text:
            sections.append({"section_id": "sec1", "section_title": "Full text (HTML, unstructured)", "text": text, "level": 1})
    return sections, tables, figures

# --------------------------- UI ------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Parses PMC XML (preferred) and Publisher HTML (fallback) → sections / tables / figure captions")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload stage2_manifest.csv", type=["csv"])
    csv_path_text = st.text_input("Or path to manifest (optional)", value="")
with col2:
    max_k = st.number_input("Max papers to parse (K)", min_value=1, max_value=500, value=30, step=1)
    parse_tables = st.checkbox("Extract tables", value=True)
    parse_figcaps = st.checkbox("Extract figure captions", value=True)

run = st.button("Parse now")

if not run:
    st.info("Upload (or select) your Stage-2 manifest and press **Parse now**.")
    st.stop()

# ---------------------- Load manifest ------------------------
try:
    if uploaded is not None:
        manifest = pd.read_csv(uploaded)
    elif csv_path_text.strip():
        manifest = pd.read_csv(csv_path_text.strip())
    else:
        st.error("Please upload or specify a manifest CSV path.")
        st.stop()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

need_cols = {"PMCID","PMID","DOI","Title","xml_path","publisher_html","publisher_pdf","status"}
missing = [c for c in need_cols if c not in manifest.columns]
if missing:
    st.warning(f"Manifest missing expected columns: {missing} — continuing with available fields.")

# ---- Inspect manifest quickly ----
st.write("Manifest columns:", list(manifest.columns))
st.write("Sample rows:", manifest.head(3))
st.write("Status value counts:", manifest.get("status", pd.Series()).value_counts())

# ---- Normalize path fields & prefer XML then HTML; skip PDFs for now ----
def _norm_path(x: object) -> str:
    s = str(x).strip() if pd.notna(x) else ""
    return "" if s.lower() in ("", "nan", "none") else s

rows = []
xml_ok = html_ok = pdf_only = 0

for _, r in manifest.iterrows():
    xmlp = _norm_path(r.get("xml_path", ""))
    htmlp = _norm_path(r.get("publisher_html", ""))
    pdfp = _norm_path(r.get("publisher_pdf", ""))

    if xmlp and Path(xmlp).exists():
        rows.append((r, "xml", xmlp)); xml_ok += 1
    elif htmlp and Path(htmlp).exists():
        rows.append((r, "html", htmlp)); html_ok += 1
    elif pdfp:
        pdf_only += 1  # present but we skip PDFs in Stage 3

st.write(f"Found XML files: {xml_ok} | HTML files: {html_ok} | PDF-only (skipped): {pdf_only}")

if not rows:
    st.warning("No parsable files found. Likely your rows are PDF-only or the paths are missing. Enable Unpaywall HTML or ensure PMC XMLs were saved in Stage 2.")
    st.stop()

# ---------------------- Parse loop ---------------------------
sec_records, tbl_records, fig_records = [], [], []
progress = st.progress(0)
status_area = st.empty()

for i, (r, kind, path_str) in enumerate(rows, start=1):
    pmcid = r.get("PMCID","")
    pmid = r.get("PMID","")
    doi = r.get("DOI","")
    title = r.get("Title","")
    src = "pmc_xml" if kind == "xml" else "publisher_html"

    p = Path(path_str)
    text = read_text(p)
    if not text:
        status_area.write(f"Empty file: {path_str}")
        progress.progress(int(i/len(rows)*100))
        continue

    if kind == "xml":
        secs, tbls, figs = parse_pmc_xml(text)
    else:
        secs, tbls, figs = parse_publisher_html(text)

    # attach ids
    for s in secs:
        s.update({"PMCID": pmcid, "PMID": pmid, "DOI": doi, "Title": title, "source": src})
    for t in tbls:
        t.update({"PMCID": pmcid, "PMID": pmid, "DOI": doi, "Title": title, "source": src})
    for f in figs:
        f.update({"PMCID": pmcid, "PMID": pmid, "DOI": doi, "Title": title, "source": src})

    sec_records.extend(secs)
    if parse_tables:
        tbl_records.extend(tbls)
    if parse_figcaps:
        fig_records.extend(figs)

    status_area.write(f"Parsed {pmcid or doi} ({i}/{len(rows)}): "
                      f"{len(secs)} sections, {len(tbls)} tables, {len(figs)} figures.")
    progress.progress(int(i/len(rows)*100))

# ---------------------- DataFrames + downloads ----------------
sec_df = pd.DataFrame(sec_records) if sec_records else pd.DataFrame(columns=["PMCID","PMID","DOI","Title","source","section_id","section_title","text","level"])
tbl_df = pd.DataFrame(tbl_records) if tbl_records else pd.DataFrame(columns=["PMCID","PMID","DOI","Title","source","table_id","label","caption","html"])
fig_df = pd.DataFrame(fig_records) if fig_records else pd.DataFrame(columns=["PMCID","PMID","DOI","Title","source","figure_id","label","caption"])

st.subheader("Sections (preview)")
st.dataframe(sec_df.head(20), use_container_width=True, height=300)

if parse_tables:
    st.subheader("Tables (preview)")
    st.dataframe(tbl_df.head(10), use_container_width=True, height=250)

if parse_figcaps:
    st.subheader("Figure captions (preview)")
    st.dataframe(fig_df.head(10), use_container_width=True, height=250)

# Save & offer downloads
try:
    sec_csv = sec_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download sections (CSV)", data=sec_csv, file_name="sections.csv", mime="text/csv")
    if parse_tables:
        tbl_csv = tbl_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download tables (CSV)", data=tbl_csv, file_name="tables.csv", mime="text/csv")
    if parse_figcaps:
        fig_csv = fig_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download figure captions (CSV)", data=fig_csv, file_name="figures.csv", mime="text/csv")
except Exception as e:
    st.warning(f"CSV write error: {e}")

# ZIP bundle
try:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sections.csv", sec_df.to_csv(index=False))
        if parse_tables:
            zf.writestr("tables.csv", tbl_df.to_csv(index=False))
        if parse_figcaps:
            zf.writestr("figures.csv", fig_df.to_csv(index=False))
    st.download_button("Download all parsed outputs (ZIP)", data=buf.getvalue(),
                       file_name="stage3_parsed_bundle.zip", mime="application/zip")
except Exception as e:
    st.warning(f"ZIP write error: {e}")

st.markdown("---")
st.caption("Stage 3 extracts structured text from PMC XML (JATS) and basic sections from publisher HTML.")
