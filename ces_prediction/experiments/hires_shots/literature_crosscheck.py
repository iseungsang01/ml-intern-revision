# -*- coding: utf-8 -*-
"""Which of our 641 shots have actually been analysed in the published literature?

"Well-studied" is not a property we can infer from the CSVs -- it has to be looked up.
This script does that reproducibly: it pulls every KSTAR paper since the 2022 campaign
from arXiv and OpenAlex, downloads whatever is open access, extracts the full text, and
reports every five-digit number in our dataset's range (30801-32751) together with the
sentence around it, so a false positive (a DOI fragment, a page number, a reference id)
can be rejected by eye rather than silently counted.

Usage (repo root):
    py ces_prediction/experiments/hires_shots/literature_crosscheck.py            # full sweep
    py ces_prediction/experiments/hires_shots/literature_crosscheck.py --report   # cached only

Writes: literature_hits.json (+ a PDF cache under <scratch>/hires_lit_pdfs).

Known limits, stated so the result is not oversold:
  * Only open-access full text is searchable. Of ~145 OA links found, ~58 PDFs actually
    downloaded -- several publishers (IOP in particular) bot-block direct fetches, so a
    shot absent from this report is *not* proven absent from the literature.
  * Paywalled papers, conference proceedings, and theses are invisible here.
  * The two FIRE-mode papers that give us #31921 and #31923 were read through their
    abstract/HTML pages, not this sweep; they are recorded in CONFIRMED below.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("CES_DATA_DIR", REPO_ROOT / "data"))
HERE = Path(__file__).resolve().parent
CACHE = Path(tempfile.gettempdir()) / "hires_lit_pdfs"
MAIL = "lss010330@snu.ac.kr"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/122.0 Safari/537.36")
FROM_DATE = "2022-06-01"
BARE_RE = re.compile(r"\b(30[89]\d{2}|3[12]\d{3})\b")

# Hand-verified hits, including the two papers whose full text is paywalled but whose
# figure captions name the shot explicitly.
CONFIRMED = {
    31921: [("On FIRE mode in KSTAR", "10.1088/1741-4326/ae332f", "fig.10: edge CES profiles, "
             "H-mode vs FIRE mode with I-mode pedestal"),
            ("Experimental identification of I-mode characteristics at the edge of FIRE mode "
             "in KSTAR", "10.1088/1741-4326/adacfc",
             "fig.3/7-9: FIRE 5.40 s vs H-mode 8.05 s, BES bispectral analysis, "
             "WCM-zonal density phase coupling")],
    31923: [("Experimental identification of I-mode characteristics at the edge of FIRE mode "
             "in KSTAR", "10.1088/1741-4326/adacfc",
             "fig.2: L-mode -> FIRE transition, WCM ~50 kHz on BES_0206 at r/a = 0.95, 3-6 s")],
    31873: [("Highest fusion performance without harmful edge energy bursts in tokamak",
             "10.1038/s41467-024-48415-w",
             "fig.5: fully automated ELM suppression, ML-integrated RMP, Ip = 0.51 MA, "
             "q95 ~ 5.1, optimizer triggered at 4.5 s")],
    31276: [("Tailoring tokamak error fields to control plasma instabilities and transport",
             "10.1038/s41467-024-45454-1",
             "fig.3: optimized path to edge stability; BES density-fluctuation bicoherence")],
    31357: [("Tailoring tokamak error fields to control plasma instabilities and transport",
             "10.1038/s41467-024-45454-1",
             "fig.6: WITH n=1 ERMP -- H-mode transition avoided; BES bicoherence at rho_N ~ 0.92")],
    31359: [("Tailoring tokamak error fields to control plasma instabilities and transport",
             "10.1038/s41467-024-45454-1",
             "fig.6: WITHOUT n=1 ERMP -- density ETB forms at 5.5 s, ELMs appear, core Ti "
             "drops to ~5 keV (paired control of #31357)")],
    32027: [("PanoMHD: multimodal modelling of plasma dynamics towards tokamak control",
             "arXiv:2603.02672",
             "fig.7: clear L/H transition, 100-300 kHz cross-power / cross-phase spectrograms")],
    31888: [("Enhancing disruption prediction through Bayesian neural network in KSTAR",
             "10.48550/arXiv.2312.12979",
             "fig.15 / appendix B: continuous disruption-prediction example shot")],
}
# Five-digit numbers that fall in our range but are not shot numbers.
FALSE_POSITIVES = {
    30907: "page range in a biochemistry reference",
    32017: "DOI fragment 10.1038/s41467-022-32017-5 (an unrelated paper named 'KSTAR')",
    31589: "DOI fragment 10.1038/s41467-022-31589-6",
}


def dataset_shots():
    return {int(os.path.basename(f)[1:-4]) for f in glob.glob(str(DATA / "s*.csv"))}


def arxiv_list():
    out, seen = [], set()
    for q in ("all:KSTAR", 'all:"Korea Superconducting Tokamak"'):
        for start in (0, 100, 200):
            url = (f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}"
                   f"&start={start}&max_results=100&sortBy=submittedDate&sortOrder=descending")
            try:
                raw = urllib.request.urlopen(url, timeout=90).read().decode()
            except Exception as exc:
                print(f"  arxiv list error: {type(exc).__name__}")
                break
            entries = ET.fromstring(raw).findall(
                "a:entry", {"a": "http://www.w3.org/2005/Atom"})
            if not entries:
                break
            for e in entries:
                ns = {"a": "http://www.w3.org/2005/Atom"}
                aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
                pub = e.find("a:published", ns).text[:10]
                if aid in seen or pub < FROM_DATE:
                    continue
                seen.add(aid)
                out.append({"key": aid, "year": pub[:4], "doi": f"arXiv:{aid}",
                            "title": " ".join(e.find("a:title", ns).text.split()),
                            "pdf": f"https://arxiv.org/pdf/{aid}"})
            time.sleep(3)
    return out


def openalex_list():
    out, cursor = [], "*"
    while cursor:
        url = ("https://api.openalex.org/works?filter=title_and_abstract.search:KSTAR,"
               f"from_publication_date:{FROM_DATE}&per-page=200&cursor={cursor}&mailto={MAIL}")
        try:
            r = json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as exc:
            print(f"  openalex error: {type(exc).__name__}")
            break
        for w in r["results"]:
            pdf = None
            for key in ("best_oa_location", "primary_location"):
                loc = w.get(key) or {}
                if loc.get("pdf_url"):
                    pdf = loc["pdf_url"]
                    break
            out.append({"key": w["id"].rsplit("/", 1)[-1], "year": w.get("publication_year"),
                        "doi": w.get("doi") or "", "title": (w.get("title") or "")[:140],
                        "pdf": pdf})
        cursor = r["meta"].get("next_cursor")
        time.sleep(1)
    return out


def fetch_pdfs(papers):
    CACHE.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, p in enumerate(papers, 1):
        if not p.get("pdf"):
            continue
        dst = CACHE / f"{p['key']}.pdf"
        if dst.exists() and dst.stat().st_size > 20000:
            p["file"] = str(dst)
            n += 1
            continue
        try:
            req = urllib.request.Request(p["pdf"], headers={"User-Agent": UA,
                                                            "Accept": "application/pdf,*/*"})
            data = urllib.request.urlopen(req, timeout=90).read()
            if len(data) > 20000 and data[:5].startswith(b"%PDF"):
                dst.write_bytes(data)
                p["file"] = str(dst)
                n += 1
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  pdf {i}/{len(papers)} (ok {n})")
        time.sleep(0.6)
    return n


def scan(papers, shots):
    try:
        import fitz
    except ImportError:
        raise SystemExit("PyMuPDF required: py -m pip install pymupdf")
    hits = []
    for p in papers:
        if not p.get("file"):
            continue
        try:
            doc = fitz.open(p["file"])
            text = "\n".join(pg.get_text() for pg in doc)
            doc.close()
        except Exception:
            continue
        for s in sorted({int(m) for m in BARE_RE.findall(text)} & shots):
            ctxs = []
            for m in re.finditer(rf"\b{s}\b", text):
                a, b = max(0, m.start() - 280), min(len(text), m.end() + 280)
                ctxs.append(" ".join(text[a:b].split()))
            hits.append({"shot": s, "title": p["title"], "doi": p["doi"], "year": p["year"],
                         "n_mentions": len(ctxs), "context": ctxs[:3],
                         "false_positive": FALSE_POSITIVES.get(s)})
    return hits


def report(shots):
    print("=" * 100)
    print("HAND-VERIFIED LITERATURE HITS (shot present in our dataset)\n")
    for s, refs in sorted(CONFIRMED.items()):
        mark = "in dataset" if s in shots else "NOT in dataset"
        print(f"#{s}  [{mark}]")
        for title, doi, role in refs:
            print(f"    {title}\n      {doi}\n      {role}")
    print("\nknown false positives:", FALSE_POSITIVES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the verified table only")
    args = ap.parse_args()
    shots = dataset_shots()
    if args.report:
        report(shots)
        return
    papers = arxiv_list() + openalex_list()
    print(f"papers since {FROM_DATE}: {len(papers)} "
          f"(with a pdf link: {sum(1 for p in papers if p.get('pdf'))})")
    n = fetch_pdfs(papers)
    print(f"full text available for {n} of them")
    hits = scan(papers, shots)
    (HERE / "literature_hits.json").write_text(
        json.dumps({"confirmed": {str(k): v for k, v in CONFIRMED.items()},
                    "false_positives": {str(k): v for k, v in FALSE_POSITIVES.items()},
                    "sweep_hits": hits, "n_papers": len(papers), "n_fulltext": n},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    found = sorted({h["shot"] for h in hits} - set(FALSE_POSITIVES))
    print(f"\nsweep found dataset shots: {found}")
    for h in sorted(hits, key=lambda x: x["shot"]):
        if h["shot"] in FALSE_POSITIVES:
            continue
        print(f"\n--- #{h['shot']}  ({h['year']}) {h['title'][:90]}\n    {h['doi']}")
        for c in h["context"][:2]:
            print(f"      ...{c[:300]}...")
    print()
    report(shots)


if __name__ == "__main__":
    main()
