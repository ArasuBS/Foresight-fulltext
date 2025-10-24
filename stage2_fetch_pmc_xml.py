# stage2_fetch_pmc_xml.py
# Foresight (Full-Text) — Stage 2: Download & cache PMC XML (from PMCID) + Europe PMC + Unpaywall OA fallback
# Input: stage1_shortlist.csv  |  Output: pmc_xml/*.xml + stage2_manifest.csv (+ optional publisher OA files)

import os, re, io, time, tarfile, zipfile, requests
from pathlib import Path
import pandas as pd
import streamlit as st

# --------------------------- Config ---------------------------
UA = "Foresight-FullText/1.0 (contact: research@syngeneintl.com)"
OA_FCGI = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EPMC_XML_FMT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML?download=1"
UNPAYWALL = "https://api.unpaywall.org/v2/"  # /{DOI}?email=you@example.com

XML_DIR = Path("pmc_xml"); XML_DIR.mkdir(exist_ok=True)
PUB_DIR = Path("publisher_oa"); PUB_DIR.mkdir(exist_ok=True)
TIMEOUT = 30
RETRIES = 3
PAUSE = 0.5  # seconds between retries

# dump odd oa.fcgi responses for debugging
OA_DUMP_DIR = Path(".cache/oa_manifests")
OA_DUMP_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------- Helpers --------------------------
def normalize_pmcid(pmcid: str) -> str:
    m = re.search(r"(PMC\d+)", str(pmcid))
    return m.group(1) if m else ""

def normalize_link_url(url: str) -> str:
    # many PMC OA links are ftp://... -> use HTTPS mirror
    if url.startswith("ftp://"):
        return url.replace("ftp://", "https://", 1)
    return url

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s[:200] if len(s) > 200 else s

def _dump_oa_manifest(pmcid: str, text: str, note: str):
    try:
        p = OA_DUMP_DIR / f"{normalize_pmcid(pmcid)}_{note}.xml"
        with open(p, "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        pass

def get_oa_manifest_xml(pmcid: str):
    try:
        r = requests.get(OA_FCGI, params={"id": pmcid}, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None

def parse_oa_links(oa_xml_text: str):
    if not oa_xml_text:
        return []
    links = []
    for fmt, href in re.findall(
        r'<link[^>]*format\s*=\s*[\'"]([^\'"]+)[\'"][^>]*href\s*=\s*[\'"]([^\'"]+)[\'"]',
        oa_xml_text, flags=re.IGNORECASE
    ):
        links.append({"format": fmt.strip().lower(), "href": href.strip()})
    return links

def download_bytes(url: str, accept: str | None = None):
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                return r.content, r.headers.get("Content-Type","")
        except Exception:
            pass
        time.sleep(PAUSE * (attempt + 1))
    return None, ""

def save_xml_from_tgz(buf: bytes, pmcid: str) -> tuple[str, str]:
    try:
        fobj = io.BytesIO(buf)
        with tarfile.open(fileobj=fobj, mode="r:gz") as tf:
            candidates = [m for m in tf.getmembers() if m.name.lower().endswith((".nxml", ".xml"))]
            if not candidates:
                return "", "no_xml_in_tgz"
            member = candidates[0]
            extracted = tf.extractfile(member)
            if extracted is None:
                return "", "tgz_member_read_error"
            xml_bytes = extracted.read()
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
    pmcid = normalize_pmcid(pmcid)
    if not pmcid:
        return "", "invalid_pmcid"

    cached = XML_DIR / f"{pmcid}.xml"
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached), "cached"

    oa_xml = get_oa_manifest_xml(pmcid)
    if oa_xml is None:
        return "", "oa_manifest_error"

    if "<error" in oa_xml.lower():
        _dump_oa_manifest(pmcid, oa_xml, "error")

    embargo_note = "embargo_present" if re.search(r"embargo-date", oa_xml, re.IGNORECASE) else ""

    links = parse_oa_links(oa_xml)
    if not links:
        _dump_oa_manifest(pmcid, oa_xml, "no_links")
        return "", "no_links"

    xml_link = next((l["href"] for l in links if l["format"] == "xml"), None)
    tgz_link = next((l["href"] for l in links if l["format"] in ("tgz", "tar.gz")), None)

    if xml_link:
        data, _ct = download_bytes(normalize_link_url(xml_link))
        if data:
            path, note = save_xml_bytes(data, pmcid)
            if note.startswith("ok"):
                return path, f"{note}{';'+embargo_note if embargo_note else ''}"

    if tgz_link:
        data, _ct = download_bytes(normalize_link_url(tgz_link))
        if data:
            path, note = save_xml_from_tgz(data, pmcid)
            if note.startswith("ok"):
                return path, f"{note}{';'+embargo_note if embargo_note else ''}"
            return "", note
        return "", "tgz_download_failed"

    return "", "no_xml_format_found"

# ---------------- Europe PMC fallback -------------------------
def fetch_epmc_xml(pmcid: str) -> tuple[str, str]:
    pmcid = normalize_pmcid(pmcid)
    if not pmcid:
        return "", "invalid_pmcid"
    url = EPMC_XML_FMT.format(pmcid=pmcid)
    data, ct = download_bytes(url)
    if data and data.strip().startswith(b"<"):
        path, note = save_xml_bytes(data, pmcid)
        if note.startswith("ok"):
            return path, "ok_xml;epmc_fallback"
    return "", "epmc_no_xml"

# ---------------- Unpaywall (publisher OA) --------------------
def normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi

def fetch_unpaywall_record(doi: str, email: str) -> dict | None:
    doi = normalize_doi(doi)
    if not doi or not email:
        return None
    try:
        r = requests.get(f"{UNPAYWALL}{doi}", params={"email": email}, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def pick_best_oa_location(record: dict) -> tuple[str | None, str | None, str | None]:
    """
    Returns (pdf_url, html_url, license_str)
    """
    if not record:
        return None, None, None
    loc = record.get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf")
    html = loc.get("url")
    lic = loc.get("license") or record.get("license")
    return pdf, html, lic

def download_publisher_pdf_or_html(doi: str, email: str, pmcid: str | None = None) -> tuple[str, str, str]:
    """
    Try Unpaywall -> download PDF (preferred) or HTML. Save under publisher_oa/.
    Returns (pdf_path, html_path, note)
    """
    rec = fetch_unpaywall_record(doi, email)
    if not rec or not rec.get("is_oa"):
        return "", "", "unpaywall_not_oa_or_error"

    pdf_url, html_url, lic = pick_best_oa_location(rec)

    if pdf_url:
        data, ct = download_bytes(pdf_url, accept="application/pdf")
        if data and (("pdf" in ct.lower()) or pdf_url.lower().endswith(".pdf")):
            name = sanitize_filename((pmcid or normalize_doi(doi)).replace("/", "_")) + ".pdf"
            out = PUB_DIR / name
            with open(out, "wb") as f:
                f.write(data)
            return str(out), "", f"ok_pdf;license={lic or ''};source=unpaywall"

    if html_url:
        data, ct = download_bytes(html_url, accept="text/html, application/xhtml+xml")
        if data:
            name = sanitize_filename((pmcid or normalize_doi(doi)).replace("/", "_")) + ".html"
            out = PUB_DIR / name
            with open(out, "wb") as f:
                f.write(data)
            return "", str(out), f"ok_html;license={lic or ''};source=unpaywall"

    return "", "", "unpaywall_no_download"
def build_full_bundle_zip(manifest: pd.DataFrame) -> bytes:
    """
    Build a single ZIP with:
      - stage2_manifest.csv
      - pmc_xml/*.xml
      - publisher_oa/* (html/pdf if present)
      - .cache/oa_manifests/*.xml (debug, if present)
      - README.txt (summary + timestamp)
    """
    import io, zipfile, datetime
    buf = io.BytesIO()
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # manifest
        manifest_csv = manifest.to_csv(index=False).encode("utf-8")
        zf.writestr("stage2_manifest.csv", manifest_csv)

        # pmc xmls
        if XML_DIR.exists():
            for p in XML_DIR.glob("PMC*.xml"):
                zf.write(p.as_posix(), arcname=f"pmc_xml/{p.name}")

        # publisher OA (html/pdf)
        if PUB_DIR.exists():
            for p in PUB_DIR.glob("*"):
                if p.suffix.lower() in (".html", ".htm", ".pdf"):
                    zf.write(p.as_posix(), arcname=f"publisher_oa/{p.name}")

        # optional: oa.fcgi dumps (useful for debugging)
        if OA_DUMP_DIR.exists():
            for p in OA_DUMP_DIR.glob("*.xml"):
                zf.write(p.as_posix(), arcname=f"debug_oa_manifests/{p.name}")

        # readme
        ok_xml = (manifest["status"].str.contains("ok_xml|ok_tgz|cached", case=False, na=False)).sum()
        ok_pub = (manifest["status"].str.contains("ok_pdf|ok_html", case=False, na=False)).sum()
        readme = (
            "Foresight Full-Text — Stage 2 Bundle\n"
            f"Generated (UTC): {ts}\n\n"
            f"Rows: {len(manifest)}\n"
            f"XML ready: {ok_xml}\n"
            f"Publisher OA files: {ok_pub}\n"
            "\nFolders:\n"
            "  pmc_xml/         — PMC XML files\n"
            "  publisher_oa/    — Publisher HTML/PDF (OA)\n"
            "  debug_oa_manifests/ — Raw oa.fcgi responses for debugging (optional)\n"
            "\nFiles:\n"
            "  stage2_manifest.csv — Final manifest of this run\n"
        )
        zf.writestr("README.txt", readme)

    return buf.getvalue()

# --------------------------- UI -------------------------------
st.set_page_config(page_title="Foresight Full-Text — Stage 2 (PMC XML + OA fallback)", layout="wide")
st.title("Foresight Full-Text — Stage 2")
st.caption("PMC XML via oa.fcgi → Europe PMC XML fallback → Unpaywall (publisher OA) optional → Cache → Manifest + ZIP")

# --- Previous run downloads (optional but recommended) ---
prev = st.session_state.get("stage2_store", {})
if prev.get("full_bundle_zip"):
    with st.expander("Download from previous run", expanded=True):
        st.download_button(
            "Download EVERYTHING (ZIP)",
            data=prev["full_bundle_zip"],
            file_name="stage2_full_bundle.zip",
            mime="application/zip",
            key="prev_dl_full_bundle",
        )
        if st.button("Clear saved downloads"):
            st.session_state.stage2_store = {}
            st.experimental_rerun()

col1, col2, col3 = st.columns([2,1,1])
with col1:
    uploaded = st.file_uploader("Upload stage1_shortlist.csv", type=["csv"])
    csv_path_text = st.text_input("Or path to stage1_shortlist.csv (optional)", value="")
with col2:
    k_slider = st.slider("Max OA papers to process (K)", min_value=1, max_value=300, value=60, step=1)
    topk = st.number_input("Or type exact K", min_value=1, max_value=300, value=int(k_slider), step=1)
    skip_existing = st.checkbox("Skip if XML already cached", value=True)
with col3:
    enable_epmc = st.checkbox("Enable Europe PMC fallback", value=True)
    enable_unpaywall = st.checkbox("Enable Unpaywall OA fallback", value=False)
    unpaywall_email = st.text_input("Unpaywall email", value="", help="Required by Unpaywall if enabled")

run = st.button("Fetch Full Text")

if not run:
    st.info("Upload (or point to) your Stage-1 CSV and press **Fetch Full Text**.")
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
df = df[df["PMCID"].str.startswith("PMC") | df["DOI"].notna()].copy()  # keep DOI rows for Unpaywall fallback
if df.empty:
    st.warning("No usable rows (need PMCID or DOI). Re-run Stage 1 or widen your window.")
    st.stop()

# sort by citations (if present) and take top K
if "Citations" in df.columns:
    df = df.sort_values(["Citations"], ascending=False).reset_index(drop=True)
df = df.drop_duplicates(subset=["PMCID","DOI"]).head(topk).reset_index(drop=True)

st.write(f"**Rows selected:** {len(df)}")
st.dataframe(df[["Title","PMID","PMCID","DOI"]].head(20), use_container_width=True, height=300)

# ----------------------- Download loop ------------------------
results = []
progress = st.progress(0)
status_area = st.empty()

for i, row in df.iterrows():
    pmcid = row.get("PMCID", "")
    pmid = row.get("PMID", "")
    title = row.get("Title", "")
    doi = row.get("DOI", "")

    status_area.write(f"Fetching {pmcid or doi} ({i+1}/{len(df)})…")

    xml_path = ""
    note = ""
    pub_pdf = ""
    pub_html = ""
    lic_info = ""

    # 1) Try PMC (if PMCID present)
    if pmcid and (not (skip_existing and (XML_DIR / f"{pmcid}.xml").exists())):
        xml_path, note = fetch_and_cache_xml(pmcid)

    # 2) Europe PMC fallback (if enabled and no XML yet)
    if (not xml_path) and enable_epmc and pmcid:
        ep_path, ep_note = fetch_epmc_xml(pmcid)
        if ep_path:
            xml_path, note = ep_path, ep_note
        elif not note:
            note = ep_note

    # 3) Unpaywall (publisher OA) fallback (if enabled and we still lack XML)
    if (not xml_path) and enable_unpaywall and unpaywall_email and doi:
        pdf_path, html_path, up_note = download_publisher_pdf_or_html(doi, unpaywall_email, pmcid or normalize_doi(doi))
        pub_pdf, pub_html = pdf_path, html_path
        note = note + (";" if note else "") + up_note
        if "license=" in up_note:
            lic_info = up_note.split("license=", 1)[-1].split(";")[0]

    # if cached existed and was skipped
    if (not xml_path) and pmcid and skip_existing and (XML_DIR / f"{pmcid}.xml").exists():
        xml_path = str(XML_DIR / f"{pmcid}.xml")
        note = note + (";" if note else "") + "cached"

    results.append({
        "PMCID": pmcid,
        "PMID": pmid,
        "DOI": doi,
        "Title": title,
        "xml_path": xml_path,
        "publisher_pdf": pub_pdf,
        "publisher_html": pub_html,
        "license": lic_info,
        "status": note or "no_action"
    })
    progress.progress(int((i+1) / len(df) * 100))

status_area.write("Done.")

manifest = pd.DataFrame(results)
st.subheader("Results (Stage 2 Manifest)")
st.dataframe(manifest[["PMCID","PMID","status","xml_path","publisher_pdf","publisher_html","license","Title"]],
             use_container_width=True, height=420)

st.write("Status counts:", manifest["status"].value_counts(dropna=False))
ok_xml = (manifest["status"].str.contains("ok_xml|ok_tgz|cached", case=False, na=False))
ok_pub = (manifest["status"].str.contains("ok_pdf|ok_html", case=False, na=False))
st.write(f"**XML ready:** {int(ok_xml.sum())} / {len(manifest)}  |  **Publisher OA files:** {int(ok_pub.sum())}")

# ---- Build & persist ONE full bundle ZIP ----
full_zip = build_full_bundle_zip(manifest)

# Keep across reruns and also write to disk (optional)
if "stage2_store" not in st.session_state:
    st.session_state.stage2_store = {}
st.session_state.stage2_store["full_bundle_zip"] = full_zip
st.session_state.stage2_store["manifest_csv"] = manifest.to_csv(index=False).encode("utf-8")

# Optional disk copies (handy after refresh)
Path("stage2_full_bundle.zip").write_bytes(full_zip)
Path("stage2_manifest.csv").write_bytes(st.session_state.stage2_store["manifest_csv"])

# ---- Single download button ----
st.download_button(
    "Download EVERYTHING (ZIP)",
    data=st.session_state.stage2_store["full_bundle_zip"],
    file_name="stage2_full_bundle.zip",
    mime="application/zip",
    key="dl_full_bundle"
)

# -------- Persist outputs so downloads survive reruns --------
if "stage2_store" not in st.session_state:
    st.session_state.stage2_store = {}

# CSV bytes
manifest_csv_bytes = manifest.to_csv(index=False).encode("utf-8")
st.session_state.stage2_store["manifest_csv"] = manifest_csv_bytes

# ZIP of PMC XMLs (from files currently on disk)
buf_xml = io.BytesIO()
with zipfile.ZipFile(buf_xml, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in XML_DIR.glob("PMC*.xml"):
        zf.write(p.as_posix(), arcname=p.name)
xml_zip_bytes = buf_xml.getvalue()
st.session_state.stage2_store["xml_zip"] = xml_zip_bytes

# ZIP of publisher OA (if any)
pub_files = list(PUB_DIR.glob("*"))
if pub_files:
    buf_pub = io.BytesIO()
    with zipfile.ZipFile(buf_pub, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pub_files:
            zf.write(p.as_posix(), arcname=p.name)
    st.session_state.stage2_store["html_zip"] = buf_pub.getvalue()
else:
    st.session_state.stage2_store["html_zip"] = None

# Optional: also write to disk so you can grab them even after a full refresh
Path("stage2_manifest.csv").write_bytes(manifest_csv_bytes)
Path("pmc_xml_bundle.zip").write_bytes(xml_zip_bytes)
if st.session_state.stage2_store["html_zip"]:
    Path("publisher_oa_bundle.zip").write_bytes(st.session_state.stage2_store["html_zip"])

st.markdown("---")
st.caption("Order: PMC → Europe PMC (XML) → Unpaywall (publisher OA). All downloads respect OA availability & licenses.")
