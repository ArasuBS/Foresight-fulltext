# stage2_fetch_pmc_xml.py
# Foresight (Full-Text) — Stage 2: Download & cache PMC XML (from PMCID)
# Input: stage1_shortlist.csv  |  Output: pmc_xml/*.xml + stage2_manifest.csv

import os, re, io, time, tarfile, zipfile, requests
from pathlib import Path
import pandas as pd
import streamlit as st

# --------------------------- Config ---------------------------
UA = "Foresight-FullText/1.0 (contact: research@syngeneintl.com)"
OA_FCGI = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
XML_DIR = Path("pmc_xml")
XML_DIR.mkdir(exist_ok=True)
TIMEOUT = 30
RETRIES = 3
PAUSE = 0.5  # seconds between retries

# --------------------------- Helpers --------------------------
def normalize_pmcid(pmcid: str) -> str:
    m = re.search(r"(PMC\d+)", str(pmcid))
    return m.group(1) if m else ""

def get_oa_manifest_xml(pmcid: str):
    """Call oa.fcgi and return its XML text (or None)."""
    try:
        r = requests.get(OA_FCGI, params={"id": pmcid}, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None

def parse_oa_links(oa_xml_text: str):
    """
    Parse oa.fcgi response and return a list of dicts:
      [{'format': 'xml'|'tgz'|..., 'href': 'https://...'}, ...]
    """
    if not oa_xml_text:
        return []
    # very light parsing via regex to avoid heavy XML deps
    # <link format="xml" href="..."/>
    links = []
    for fmt, href in re.findall(r'<link[^>]*format="([^"]+)"[^>]*href="([^"]+)"', oa_xml_text):
        links.append({"format": fmt.lower(), "href": href})
    return links

def download_bytes(url: str):
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        time.sleep(PAUSE * (attempt + 1))
    return None

def save_xml_from_tgz(buf: bytes, pmcid: str) -> tuple[str, str]:
    """
    Extract first *.nxml or *.xml from a PMC tgz and save to XML_DIR/PMCID.xml.
    Returns (xml_path, note)
    """
    try:
        fobj = io.BytesIO(buf)
        with tarfile.open(fileobj=fobj, mode="r:gz") as tf:
            # Prefer *.nxml
            candidates = [m for m in tf.getmembers() if m.name.lower().endswith((".nxml", ".xml"))]
            if not candidates:
                return "", "no_xml_in_tgz"
            member = candidates[0]
            xml_bytes = tf.extractfile(member).read()
            out_path = XML_DIR / f"{pmcid}.xml"
            with open(out_path, "wb") as f:
                f.write(xml_bytes)
            return str(out_path), "ok_tgz"
    except Exception as e:
        return "", f"tgz_extract_error:{e}"

def save_xml_bytes(buf: bytes, pmcid: str) -> tuple[str, str]:
    try:
        out_path = XML_DIR / f"{pmcid}.xml"
        with open(out_path, "wb") as f:
            f.write(buf)
        return str(out_path), "ok_xml"
    except Exception as e:
        return "", f"xml_write_error:{e}"

def fetch_and_cache_xml(pmcid: str) -> tuple[str, str]:
    """
    For a PMCID:
      - If cached, return cached path.
      - Else hit oa.fcgi, prefer direct XML link, else TGZ (extract XML),
        else report no_xml.
    Returns (xml_path, note/status)
    """
    pmcid = normalize_pmcid(pmcid)
    if not pmcid:
        return "", "invalid_pmcid"

    cached = XML_DIR / f"{pmcid}.xml"
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached), "cached"

    # Query OA manifest
    oa_xml = get_oa_manifest_xml(pmcid)
    if oa_xml is None:
        return "", "oa_manifest_error"

    # embargo detection (simple)
    if re.search(r"embargo-date", oa_xml, re.IGNORECASE):
        # We still might get XML, but warn in note.
        embargo_note = "embargo_present"
    else:
        embargo_note = ""

    links = parse_oa_links(oa_xml)
    if not links:
        return "", "no_links"

    # Prefer XML direct, else TGZ
    xml_link = next((l["href"] for l in links if l["format"] == "xml"), None)
    tgz_link = next((l["href"] for l in links if l["format"] in ("tgz", "tar.gz")), None)

    if xml_link:
        data = download_bytes(xml_link)
        if data:
            path, note = save_xml_bytes(data, pmcid)
            if note.startswith("ok"):
                return path, f"{note}{';'+embargo_note if embargo_note else ''}"
        # fall through to tgz if xml failed
    if tgz_link:
        data = download_bytes(tgz_link)
        if data:
            path, note = save_xml_from_tgz(data, pmcid)
            if note.startswith("ok"):
                return path, f"{note}{';'+embargo_note if embargo_note else ''}"
        return "", "tgz_download_or_extract_failed"

    return "", "no_xml_format_found"

def zip_all_xmls() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in XML_DIR.glob("PMC*.xml"):
            zf.write(p.as_posix(), arcname=p.name)
    return buf.getvalue()

# --------------------------- UI -------------------------------
st.set_page_config(page_title="Foresight Full-Text — Stage 2 (PMC XML downloader)", layout="wide")
st.title("Foresight Full-Text — Stage 2")
st.caption("Input: Stage-1 shortlist CSV → Download PMC XML via oa.fcgi → Cache → Manifest + ZIP")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload stage1_shortlist.csv", type=["csv"])
    csv_path_text = st.text_input("Or path to stage1_shortlist.csv (optional)", value="")
with col2:
    topk = st.slider("Max OA papers to download (K)", 10, 200, 60, 5)
    skip_existing = st.checkbox("Skip if XML already cached", value=True)

run = st.button("Fetch PMC XML")

if not run:
    st.info("Upload (or point to) your Stage-1 CSV and press **Fetch PMC XML**.")
    st.stop()

# ----------------------- Load input CSV -----------------------
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif csv_path_text.strip():
        df = pd.read_csv(csv_path_text.strip())
    else:
        st.error("Please upload or specify a CSV path.")
        st.stop()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

required_cols = {"PMCID","PMID","Title"}
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"CSV missing columns: {missing_cols}")
    st.stop()

# Filter OA rows and normalize PMCID
df["PMCID"] = df["PMCID"].fillna("").astype(str).map(normalize_pmcid)
df = df[df["PMCID"].str.startswith("PMC")].copy()
if df.empty:
    st.warning("No OA rows (PMCID empty). Re-run Stage 1 or widen your query/time window.")
    st.stop()

# Choose top K OA (keep original order; or sort by Citations if present)
if "Citations" in df.columns:
    df = df.sort_values(["Citations"], ascending=False).reset_index(drop=True)

df = df.drop_duplicates(subset=["PMCID"]).head(topk).reset_index(drop=True)

st.write(f"**OA rows selected:** {len(df)}")
st.dataframe(df[["Title","PMID","PMCID"]].head(20), use_container_width=True, height=300)

# ----------------------- Download loop ------------------------
results = []
progress = st.progress(0)
status_area = st.empty()

for i, row in df.iterrows():
    pmcid = row["PMCID"]
    pmid = row.get("PMID", "")
    title = row.get("Title", "")
    doi = row.get("DOI", "")

    status_area.write(f"Fetching {pmcid} ({i+1}/{len(df)})…")
    if skip_existing and (XML_DIR / f"{pmcid}.xml").exists():
        xml_path = str(XML_DIR / f"{pmcid}.xml")
        note = "cached"
    else:
        xml_path, note = fetch_and_cache_xml(pmcid)

    results.append({
        "PMCID": pmcid,
        "PMID": pmid,
        "DOI": doi,
        "Title": title,
        "xml_path": xml_path,
        "status": note
    })
    progress.progress(int((i+1) / len(df) * 100))

status_area.write("Done.")

manifest = pd.DataFrame(results)
st.subheader("Results (Stage 2 Manifest)")
st.dataframe(manifest[["PMCID","PMID","status","xml_path","Title"]], use_container_width=True, height=400)

ok = (manifest["status"].str.startswith("ok")) | (manifest["status"].eq("cached"))
st.write(f"**Downloaded/ cached XMLs:** {int(ok.sum())} / {len(manifest)}")

# Save manifest + ZIP download
try:
    manifest_path = Path("stage2_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    st.download_button(
        "Download manifest (CSV)",
        data=manifest.to_csv(index=False).encode("utf-8"),
        file_name="stage2_manifest.csv",
        mime="text/csv"
    )
except Exception as e:
    st.warning(f"Could not write manifest: {e}")

try:
    zip_bytes = zip_all_xmls()
    st.download_button(
        "Download all XMLs (ZIP)",
        data=zip_bytes,
        file_name="pmc_xml_bundle.zip",
        mime="application/zip"
    )
except Exception as e:
    st.warning(f"Could not create ZIP: {e}")

st.markdown("---")
st.caption("Notes: Uses PMC oa.fcgi. Prefers direct XML, falls back to TGZ. Handles simple embargo flags and caching.")

