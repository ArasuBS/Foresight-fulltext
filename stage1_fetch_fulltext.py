# stage1_fetch_fulltext.py
# Foresight (Full-Text) — Stage 1: Shortlist + OA detection + Citations + Ranking
# PubMed (E-Utils) · Crossref cache · PMCID resolution · TF-IDF (no sklearn)

import os, re, time, random, string, math, json, requests
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------- Config ---------------------------
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF = "https://api.crossref.org/works"
UA = "Foresight-FullText/1.0 (contact: research@syngeneintl.com)"

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
CIT_CACHE_PATH = CACHE_DIR / "crossref_citations.csv"

# --------------------------- Stopwords ------------------------
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during each
few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just ll m me more most my myself no nor not of
off on once only or other our ours ourselves out over own s same she should so
some such t than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while
who whom why will with you your yours yourself yourselves
""".split())
STOPWORDS |= {
    "antibody","drug","conjugate","conjugation","adc","linker","payload","study",
    "patient","patients","cancer","tumor","using","use","based","via","new","novel",
    "method","methods","results","data","analysis","effect","effects"
}

# --------------------------- Queries --------------------------
DEFAULT_QUERY = (
    '("antibody-drug conjugate"[TIAB] OR "antibody drug conjugate"[TIAB] OR "Antibody-Drug Conjugates"[MeSH]) '
    'AND (conjugation OR "site-specific" OR "site specific" OR linker OR "val-cit" OR "vc-PABC" OR PABC '
    'OR glucuronide OR "β-glucuronidase" OR hydrazone OR disulfide OR noncleavable OR "non-cleavable" '
    'OR maleimide OR SMCC OR "sulfo-SMCC" OR MCC OR "click chemistry" OR SPAAC OR DBCO OR azide OR tetrazine OR TCO '
    'OR "NHS-ester" OR "NHS ester" OR TCEP OR DTT OR NEM OR iodoacetamide '
    'OR "transglutaminase" OR "sortase" OR "glycan engineering" OR "bioorthogonal" OR "rebridging" '
    'OR "process development" OR "scale-up" OR manufacturing OR "GMP" OR "CMC" OR "QbD" OR "PAT" '
    'OR MMAE OR DM1 OR DM4 OR SN-38 OR PBD OR duocarmycin OR auristatin '
    'OR "pharmacokinetics" OR "toxicology" OR "in vivo oncology" OR "in vivo pharmacology" OR "in vitro functional assays" '
    'OR "discovery-grade" OR "conjugation characterization" OR "linker-payload synthesis" OR "linker-payload optimization" '
    'OR "antibody discovery" OR "antibody engineering" OR cloning OR "mammalian" OR "manufacturability assessment" '
    'OR "affinity determination" OR "KD determination" OR "sequence optimization" OR "payload library synthesis" '
    'OR "payload optimization" OR "linker modality" OR "linker payload scale-up" OR "chemical conjugation" '
    'OR "thiol conjugation" OR "amine conjugation" OR "site-specific conjugation" OR "enzymatic conjugation" '
    'OR "dual drug conjugation" OR "fluorophore conjugation" OR "oligo-conjugation" '
    'OR "scale-up for in vivo" OR "receptor expression" OR "receptor density" OR "internalization assay" '
    'OR "flow cytometry" OR Incucyte OR "cytotoxicity assay" OR "immune cell activation" OR "bystander effect assay" '
    'OR "2D cytotoxicity" OR "3D cytotoxicity" OR "cytokine release assay" OR "co-culture assay" '
    'OR "ADCP assay" OR "FcγR activation" OR immunogenicity OR "anti-drug antibody" OR ADA '
    'OR "species cross-reactivity" OR "tissue cross-reactivity" OR "accelerated stability" OR "analytical method development" '
    'OR "preclinical ADC" OR "proof-of-concept ADC" OR "immune stimulating ADC" OR ISAC '
    ') '
    'NOT (MRI OR "magnetic resonance" OR diffusion OR "apparent diffusion coefficient" OR DWI OR "ADC map")'
)

# --------------------- E-Utils helper -------------------------
def eutils_request(endpoint, params, max_retries=5, pause_base=0.25):
    headers = {"User-Agent": UA}
    # optional API key via Streamlit secrets
    try:
        api_key = st.secrets.get("NCBI_API_KEY", None)
    except Exception:
        api_key = None
    if api_key:
        params = {**params, "api_key": api_key}

    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{EUTILS}/{endpoint}", params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                sleep_s = min(10, (2 ** attempt) * pause_base) + random.uniform(0, 0.5)
                time.sleep(sleep_s); continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(min(10, (2 ** attempt) * pause_base) + random.uniform(0, 0.5))
    if last_exc: raise last_exc
    raise RuntimeError("Unknown E-Utils error")

@st.cache_data(show_spinner=False, ttl=3600)
def pm_esearch(term, start_dt, end_dt, retmax=400):
    p = {
        "db": "pubmed", "term": term, "retmode": "json", "retmax": str(retmax),
        "sort": "pubdate", "datetype": "pdat",
        "mindate": start_dt.strftime("%Y/%m/%d"),
        "maxdate": end_dt.strftime("%Y/%m/%d"),
    }
    r = eutils_request("esearch.fcgi", p)
    return r.json().get("esearchresult", {}).get("idlist", [])

@st.cache_data(show_spinner=False, ttl=3600)
def pm_esummary(ids, chunk=180, pause=0.25):
    if not ids:
        return []
    out = []
    for i in range(0, len(ids), chunk):
        batch = ids[i:i+chunk]
        r = eutils_request("esummary.fcgi", {"db": "pubmed", "id": ",".join(batch), "retmode": "json"})
        data = r.json().get("result", {})
        for pmid in batch:
            rec = data.get(pmid, {})
            if not rec:
                continue
            doi, pmcid = "", ""
            for aid in rec.get("articleids", []):
                t = (aid.get("idtype","") or "").lower()
                if t == "doi":
                    doi = aid.get("value","")
                elif t == "pmcid":
                    pmcid = aid.get("value","")  # e.g., "PMC1234567"
            out.append({
                "PMID": pmid,
                "Title": rec.get("title",""),
                "Journal": rec.get("source",""),
                "PubDate": rec.get("pubdate",""),
                "Authors": ", ".join([a.get("name","") for a in rec.get("authors", [])][:5]),
                "DOI": doi,
                "PMCID": pmcid,  # <-- new
            })
        time.sleep(pause)
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def pm_efetch_abs(ids, pause=0.15):
    absd={}
    for pmid in ids:
        try:
            r = eutils_request("efetch.fcgi", {"db":"pubmed","id":pmid,"retmode":"text","rettype":"abstract"})
            absd[pmid] = r.text.strip()
        except Exception:
            absd[pmid] = ""
        time.sleep(pause)
    return absd

def resolve_pmcid(pmid):
    """Return PMCID (e.g., 'PMC1234567') if available, else ''."""
    try:
        r = eutils_request("elink.fcgi", {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json",
            "linkname": "pubmed_pmc"  # <-- important
        })
        js = r.json()
        for ls in js.get("linksets", []):
            for db in ls.get("linksetdbs", []):
                if db.get("dbto") == "pmc":
                    links = db.get("links", [])
                    if links:
                        return "PMC" + links[0]["id"]
    except Exception:
        pass
    return ""

# -------------------- Crossref citations (cached) --------------
def _load_cit_cache():
    if CIT_CACHE_PATH.exists():
        try:
            df = pd.read_csv(CIT_CACHE_PATH)
            return {str(k): int(v) for k, v in zip(df["DOI"].astype(str), df["Citations"].fillna(-1))}
        except Exception:
            return {}
    return {}

def _save_cit_cache(cache_dict):
    try:
        df = pd.DataFrame([{"DOI": k, "Citations": v} for k, v in cache_dict.items()])
        df.to_csv(CIT_CACHE_PATH, index=False)
    except Exception:
        pass

CIT_CACHE = _load_cit_cache()

def crossref_cites_cached(doi):
    if not doi: return -1
    key = str(doi).lower().strip()
    if key in CIT_CACHE:
        return CIT_CACHE[key]
    try:
        r = requests.get(f"{CROSSREF}/{doi}", headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200:
            CIT_CACHE[key] = -1; _save_cit_cache(CIT_CACHE); return -1
        cnt = r.json().get("message",{}).get("is-referenced-by-count", -1)
        if cnt is None: cnt = -1
        CIT_CACHE[key] = int(cnt)
        _save_cit_cache(CIT_CACHE)
        return CIT_CACHE[key]
    except Exception:
        CIT_CACHE[key] = -1; _save_cit_cache(CIT_CACHE); return -1

# ------------------ TF-IDF (no sklearn) -----------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"[\n\r\t]", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if len(w) > 1 and w not in STOPWORDS)

def build_vocab(texts, max_terms=6000, min_len=3):
    counts={}
    for t in texts:
        for w in clean_text(t).split():
            if len(w) < min_len: continue
            counts[w] = counts.get(w,0) + 1
    vocab = [w for w,_ in sorted(counts.items(), key=lambda x:x[1], reverse=True)[:max_terms]]
    index = {w:i for i,w in enumerate(vocab)}
    return vocab, index

def tfidf_matrix(texts, index):
    mat = np.zeros((len(texts), len(index)), dtype=np.float32)
    for i,t in enumerate(texts):
        for w in clean_text(t).split():
            j = index.get(w)
            if j is not None:
                mat[i,j] += 1.0
    df = (mat > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
    mat = mat * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms

def embed_texts(texts):
    vocab, index = build_vocab(texts)
    return tfidf_matrix(texts, index), (vocab, index)

def embed_query(q, meta):
    vocab, index = meta
    return tfidf_matrix([q], index)[0]

def cosine_topk(qv, M):
    return (M @ qv)

# ----------------- Simple relevance filter --------------------
IMAGING_BAD = ["magnetic resonance","mri","diffusion","apparent diffusion coefficient","dwi","adc map","mr imaging","diffusion-weighted"]
BIOCONJ_MUST = [
    "antibody-drug conjugate","antibody drug conjugate","bioconjugate","adc ",
    "conjugation","site-specific","linker","payload","dar","transglutaminase","sortase","maleimide","val-cit","vc-pabc","dbco","tetrazine","tco","spaac"
]

def method_focused(title, abstract):
    txt = (title or "") + " " + (abstract or "")
    tl = txt.lower()
    if any(b in tl for b in IMAGING_BAD) and not any(k in tl for k in ["antibody-drug conjugate","antibody drug conjugate","bioconjugate","adc "]):
        return False
    return any(k in tl for k in BIOCONJ_MUST)

# -------------------------- UI -------------------------------
st.set_page_config(page_title="Foresight Full-Text — Stage 1", layout="wide")
st.title("Foresight Full-Text — Stage 1 (Shortlist & OA detection)")
st.caption("PubMed search → Abstract filter → Citations (cached) → PMCID (OA) → Rank → Top K export")

with st.sidebar:
    query = st.text_area("PubMed query", value=DEFAULT_QUERY, height=220)
    months_back = st.number_input("Time window (months)", 1, 48, 24, 1)
    retmax = st.slider("Max PubMed items to fetch", 50, 1000, 400, 50)
    topk = st.slider("Top K (keep for Stage 2)", 10, 120, 60, 5)
    oa_only = st.checkbox("Show OA-only (has PMCID)", value=False)
    debug = st.checkbox("Debug logs", value=False)
    if "run" not in st.session_state: st.session_state.run = False
    if st.button("Scan"):
        st.session_state.run = True
    if st.button("Reset"):
        st.session_state.run = False

if not st.session_state.run:
    st.info("Set your query and press **Scan** to run.")
    st.stop()

# ------------------------- Pipeline ---------------------------
end_dt = datetime.utcnow()
start_dt = end_dt - relativedelta(months=int(months_back))

with st.spinner("Searching PubMed…"):
    ids = pm_esearch(query, start_dt, end_dt, retmax=retmax)
    if not ids:
        st.warning("No PubMed IDs found. Adjust query or timeline.")
        st.stop()

with st.spinner("Fetching summaries & abstracts…"):
    meta = pm_esummary(ids)
    pmids = [m["PMID"] for m in meta]
    abstracts = pm_efetch_abs(pmids)

df = pd.DataFrame(meta)
df["Abstract"] = df["PMID"].map(abstracts).fillna("")

# Filter by method focus
df["MethodFocused"] = df.apply(lambda r: method_focused(r.get("Title",""), r.get("Abstract","")), axis=1)
df = df[df["MethodFocused"]].copy()
if df.empty:
    st.warning("No method-focused papers found after filtering.")
    st.stop()

# Citations (cached) and PMCID
with st.spinner("Resolving citations & PMCID…"):
    df["Citations"] = df["DOI"].fillna("").apply(crossref_cites_cached)
    def _year(pdstr):
        m = re.search(r"(\d{4})", str(pdstr) or "")
        return int(m.group(1)) if m else 0
    df["Year"] = df["PubDate"].apply(_year)
    df["PMCID"] = df["PMID"].apply(resolve_pmcid)
    df["OA"] = df["PMCID"].apply(lambda x: bool(x))
    df["PMCID_Link"] = df["PMCID"].apply(lambda x: f"https://www.ncbi.nlm.nih.gov/pmc/articles/{x}/" if x else "")

# Semantic score (titles + abstracts)
seed = ("ADC conjugation method development; site-specific; linkers; payload; DAR; HIC; TFF; GMP; "
        "CMC; QbD; PAT; scale-up; solvent handling; HPAPI containment")
texts = (df["Title"].fillna("") + ". " + df["Abstract"].fillna("")).tolist()
M, meta_embed = embed_texts(texts)
qv = embed_query(seed, meta_embed)
df["SemanticScore"] = cosine_topk(qv, M)

# Ranking: Citations DESC → Year DESC → Semantic DESC
df = df.sort_values(by=["Citations","Year","SemanticScore"], ascending=[False, False, False]).reset_index(drop=True)

# OA filter (view)
view = df[df["OA"]] if oa_only else df
st.subheader(f"Shortlist (showing {'OA-only' if oa_only else 'all'})")
st.dataframe(
    view.head(topk)[["Title","Journal","PubDate","Authors","Citations","Year","PMID","DOI","PMCID","PMCID_Link","SemanticScore"]],
    use_container_width=True, height=500
)

# Summary stats
tot = len(df); tot_oa = int(df["OA"].sum())
st.write(f"**Total method-focused:** {tot} | **Open Access (PMCID):** {tot_oa} | **Exporting top K = {topk}**")

# Export CSV (top K, preserve OA info + links)
export_cols = ["Title","Journal","PubDate","Authors","Citations","Year","PMID","DOI","PMCID","PMCID_Link","OA","SemanticScore","Abstract"]
topK_df = df.head(topk)[export_cols].copy()
st.download_button(
    "Download Shortlist (CSV)",
    data=topK_df.to_csv(index=False).encode("utf-8"),
    file_name="stage1_shortlist.csv",
    mime="text/csv"
)

# Notes & next steps
st.markdown("---")
st.caption("Stage 1 complete: shortlist + OA detection + citation ranking. Stage 2 will fetch & parse full text (PMC XML).")
