# ---- Normalize path fields & prefer XML then HTML; skip PDFs for now ----
def _norm_path(x: object) -> str:
    s = str(x).strip() if pd.notna(x) else ""
    return "" if s.lower() in ("", "nan", "none") else s

def _best_existing_path(raw: str, kind: str) -> str:
    """
    Try a few common path variants so we tolerate small differences across machines.
    kind: "xml" or "html"
    """
    if not raw:
        return ""
    p = Path(raw)
    # exact
    if p.exists():
        return str(p)
    # handle bare filename → add folder
    name = p.name
    if kind == "xml":
        cand = Path("pmc_xml") / name
        if cand.exists():
            return str(cand)
    if kind == "html":
        cand = Path("publisher_oa") / name
        if cand.exists():
            return str(cand)
    # if manifest had only PMCID, synthesize pmc_xml/PMCID.xml
    m = re.search(r"(PMC\d+)", raw)
    if kind == "xml" and m:
        cand = Path("pmc_xml") / f"{m.group(1)}.xml"
        if cand.exists():
            return str(cand)
    return ""

rows = []
xml_ok = html_ok = pdf_only = 0

for _, r in manifest.iterrows():
    xmlp = _norm_path(r.get("xml_path", ""))
    htmlp = _norm_path(r.get("publisher_html", ""))
    pdfp = _norm_path(r.get("publisher_pdf", ""))

    xml_found = _best_existing_path(xmlp, "xml")
    html_found = _best_existing_path(htmlp, "html")

    if xml_found:
        rows.append((r, "xml", xml_found)); xml_ok += 1
    elif html_found:
        rows.append((r, "html", html_found)); html_ok += 1
    elif pdfp:
        pdf_only += 1  # present but we skip PDFs in Stage 3

st.write(f"Found XML files: {xml_ok} | HTML files: {html_ok} | PDF-only (skipped): {pdf_only}")

if not rows:
    st.warning("No parsable files found. Upload the Stage-2 full bundle ZIP here (so pmc_xml/ & publisher_oa/ exist), or re-run Stage 2 and then Stage 3 in the same working folder.")
    st.stop()
