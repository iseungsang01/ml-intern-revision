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

Getting the full text is the hard part. A publisher link is not a readable PDF: IOP in
particular bot-blocks direct fetches, and OpenAlex's `best_oa_location` often points at a
landing page. So a paper that fails the direct download is chased two more ways before it
is given up on (`fetch_fallback`):
  1. Unpaywall by DOI -- every OA location it knows, including repository copies that
     OpenAlex missed.
  2. arXiv BY TITLE -- most KSTAR papers in Nucl. Fusion / PPCF have a preprint. The title
     is looked up through the arXiv API and accepted only when the normalised titles match
     at >= 0.90, so a same-topic-different-paper preprint cannot slip in.
The run prints how many papers are still without full text, so the coverage claim stays
quantitative instead of "we searched the literature".

Known limits, stated so the result is not oversold:
  * Only open-access full text is searchable. A shot absent from this report is *not*
    proven absent from the literature -- see the coverage line the run prints.
  * SUPPLEMENTARY material is not swept. This is not hypothetical: the Supplementary
    Information of 10.1038/s41467-024-45454-1 names #31184, #31185 and #31189, none of
    which appear in the main text. They are recorded by hand in IN_RANGE_NOT_OURS. Adding
    a publisher-specific supplement crawler is the fix if this screen is ever rerun in
    anger.
  * The two FIRE-mode papers are IOP-blocked to this script and were read through their
    article pages. Doing that is what turned up #31923 in the SECOND of them, which the
    PDF sweep could never have seen.
  * Paywalled papers, conference proceedings, and theses are invisible here.
  * The two FIRE-mode papers that give us #31921 and #31923 were read through their
    abstract/HTML pages, not this sweep; they are recorded in CONFIRMED below.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
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
CHASE_LOG = CACHE / "chase.json"   # per-paper verdict, so a rerun resumes instead of restarting
MAIL = "lss010330@snu.ac.kr"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/122.0 Safari/537.36")
FROM_DATE = "2022-06-01"
BARE_RE = re.compile(r"\b(30[89]\d{2}|3[12]\d{3})\b")

# "KSTAR" is not a unique string. The OpenAlex title/abstract search for it also returns
# the Weka K* (KStar) instance-based classifier, the K* search algorithm in top-k planning,
# and a Nature Communications paper literally named KSTAR about kinase activity -- 39 % of
# the full texts we managed to download are about concrete strength, drug toxicity, polymer
# science or e-commerce reviews. That contamination is exactly where the hand-maintained
# false positives came from (#30907 is a page range in a biochemistry reference), so the
# corpus is filtered on the BODY text before any shot number is believed. Title alone is
# not enough: the kinase paper has KSTAR in its title.
FUSION_RE = re.compile(r"tokamak|plasma|fusion|divertor|pedestal|\bELM\b|H-mode", re.I)
FUSION_TITLE_RE = re.compile(
    r"tokamak|plasma|fusion|divertor|pedestal|disrupt|KSTAR|\bITER\b|stellarator|"
    r"magnetohydro|\bMHD\b", re.I)
FUSION_MIN_HITS = 3   # in the first four pages

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
             "fig.2: L-mode -> FIRE transition, WCM ~50 kHz on BES_0206 at r/a = 0.95, 3-6 s"),
            ("On FIRE mode in KSTAR", "10.1088/1741-4326/ae332f",
             "fig.11: BES spectrogram of edge density fluctuations at r/a = 0.95, 3-6 s; "
             "fig.12: radial profile of summed 30-70 kHz coherence between two poloidally "
             "connected BES channels; fig.13: poloidal wave number at r/a = 0.95, 5.5-5.7 s")],
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
# Published shots that sit inside our shot-number range (30801-32751) but whose CSV we do
# NOT hold. Recorded so nobody re-derives "the campaign has only eight published shots":
# what we have is a 641-shot SAMPLE of the campaign, not the campaign.
IN_RANGE_NOT_OURS = {
    31184: ("Tailoring tokamak error fields...", "10.1038/s41467-024-45454-1",
            "Supplementary fig.: CRMP with adaptive feedback control"),
    31185: ("Tailoring tokamak error fields...", "10.1038/s41467-024-45454-1",
            "Supplementary fig.: CRMP with pre-set constant I_RMP"),
    31189: ("Tailoring tokamak error fields...", "10.1038/s41467-024-45454-1",
            "Supplementary fig.: ERMP with adaptive feedback control; also KSTAR overview "
            "10.1088/1741-4326/ad3b1d -- RMP triggered at the L-H transition by an ML "
            "classifier"),
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


def norm_title(t):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())


def dedupe(papers):
    """One entry per paper. arXiv and OpenAlex return the same work twice, which would
    otherwise inflate both the paper count and the per-shot mention count."""
    best = {}
    for p in papers:
        k = norm_title(p["title"])[:120] or p["key"]
        cur = best.get(k)
        if cur is None or (not cur.get("pdf") and p.get("pdf")):
            best[k] = p
    return list(best.values())


def try_download(url, dst, timeout=90):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Accept": "application/pdf,*/*"})
        data = urllib.request.urlopen(req, timeout=timeout).read()
    except Exception:
        return False
    if len(data) > 20000 and data[:5].startswith(b"%PDF"):
        dst.write_bytes(data)
        return True
    return False


def fetch_pdfs(papers):
    CACHE.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, p in enumerate(papers, 1):
        dst = CACHE / f"{p['key']}.pdf"
        if dst.exists() and dst.stat().st_size > 20000:
            p["file"], p["source"] = str(dst), p.get("source", "cache")
            n += 1
            continue
        if not p.get("pdf"):
            p["miss"] = "no pdf link"
            continue
        if try_download(p["pdf"], dst):
            p["file"], p["source"] = str(dst), "direct"
            n += 1
        else:
            p["miss"] = "download blocked"
        if i % 25 == 0:
            print(f"  pdf {i}/{len(papers)} (ok {n})")
        time.sleep(0.6)
    return n


def unpaywall_pdf(doi):
    """Every OA location Unpaywall knows for this DOI, not just the publisher's."""
    doi = (doi or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi or doi.lower().startswith("arxiv:"):
        return []
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={MAIL}"
    try:
        r = json.load(urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=25))
    except Exception:
        return []
    out = []
    for loc in [r.get("best_oa_location")] + (r.get("oa_locations") or []):
        if loc and loc.get("url_for_pdf"):
            out.append(loc["url_for_pdf"])
    return list(dict.fromkeys(out))


def arxiv_by_title(title):
    """The preprint of a paywalled paper, found by title.

    Accepted only when the normalised titles match at >= 0.90 -- an arXiv search for a
    plasma-physics title returns plenty of same-topic papers, and scanning the wrong PDF
    would put a shot number in the report that the cited paper never mentions."""
    want = norm_title(title)
    if len(want) < 25:
        return None
    phrase = " ".join(want.split()[:16])
    for q in (f'ti:"{phrase}"',):
        url = (f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}"
               f"&max_results=8")
        try:
            raw = urllib.request.urlopen(url, timeout=90).read().decode()
            entries = ET.fromstring(raw).findall("a:entry",
                                                 {"a": "http://www.w3.org/2005/Atom"})
        except Exception:
            entries = []
        time.sleep(3)
        for e in entries:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            got = norm_title(" ".join(e.find("a:title", ns).text.split()))
            if difflib.SequenceMatcher(None, want, got).ratio() >= 0.90:
                return e.find("a:id", ns).text.rsplit("/", 1)[-1]
    return None


def fetch_fallback(papers, budget_s=None):
    """Chase the papers the direct download could not get: Unpaywall, then arXiv by title.

    Every verdict -- recovered or not -- is written to CHASE_LOG, so this is resumable: a
    rerun skips the papers already decided and only spends its arXiv rate limit on new
    ones. `budget_s` stops the pass early (the caller reruns to finish)."""
    try:
        log = json.loads(CHASE_LOG.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    todo, skipped = [], 0
    for p in papers:
        if p.get("file"):
            continue
        prev = log.get(p["key"])
        if prev == "dead-end":
            p["miss"] = p.get("miss", "chased, no OA full text")
            skipped += 1
        else:
            todo.append(p)
    print(f"chasing {len(todo)} papers without full text ({skipped} already chased in vain)")
    n, t0 = 0, time.time()

    def by_unpaywall(p):
        dst = CACHE / f"{p['key']}.pdf"
        for url in unpaywall_pdf(p.get("doi", "")):
            if try_download(url, dst, timeout=25):
                return p, str(dst)
        return p, None

    # Pass 1 -- Unpaywall, in parallel. Independent per paper and network-bound.
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, (p, path) in enumerate(pool.map(by_unpaywall, todo), 1):
            if path:
                p["file"], p["source"] = path, "unpaywall"
                p.pop("miss", None)
                n += 1
            if i % 25 == 0:
                print(f"  unpaywall {i}/{len(todo)} (recovered {n})", flush=True)
    print(f"  unpaywall pass: {n} recovered in {time.time() - t0:.0f}s", flush=True)

    # Pass 2 -- arXiv by title, serial: the API asks for one request every 3 s.
    # Only a paper that got through BOTH passes may be recorded as a dead end; one the
    # budget cut off is left unrecorded so the next run retries it.
    done = {p["key"] for p in todo if p.get("file")}
    rest = [p for p in todo if not p.get("file")]
    print(f"  arxiv-by-title on {len(rest)} still missing", flush=True)
    for i, p in enumerate(rest, 1):
        if budget_s and time.time() - t0 > budget_s:
            print(f"  budget reached at {i}/{len(rest)} -- rerun to continue", flush=True)
            break
        dst = CACHE / f"{p['key']}.pdf"
        aid = arxiv_by_title(p["title"])
        if aid and try_download(f"https://arxiv.org/pdf/{aid}", dst, timeout=45):
            p["file"], p["source"] = str(dst), f"arxiv-by-title:{aid}"
            p.pop("miss", None)
            n += 1
        done.add(p["key"])
        if i % 10 == 0:
            print(f"  arxiv {i}/{len(rest)} (total recovered {n})", flush=True)
            CHASE_LOG.write_text(json.dumps(log), encoding="utf-8")

    for p in todo:
        if p.get("file"):
            log[p["key"]] = p["source"]
        elif p["key"] in done:
            log[p["key"]] = "dead-end"
            p["miss"] = "chased, no OA full text"
    CHASE_LOG.write_text(json.dumps(log), encoding="utf-8")
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
            head = "\n".join(pg.get_text() for pg in doc[:4])
            text = "\n".join(pg.get_text() for pg in doc)
            doc.close()
        except Exception:
            continue
        # Off-topic paper: any five-digit number in it is a page range or an ID, never one
        # of our discharges. Recorded, not scanned.
        if len(FUSION_RE.findall(head)) < FUSION_MIN_HITS:
            p["offtopic"] = True
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
    print("\npublished, in our shot-number range, but NOT in our 641 CSVs:")
    for s_, (title, doi, role) in sorted(IN_RANGE_NOT_OURS.items()):
        print(f"    #{s_}  {title}\n      {doi}\n      {role}")


def main():
    # Paper text carries ligatures and dashes that the Windows console codepage cannot
    # encode; without this the run dies on the report AFTER doing all the work.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the verified table only")
    args = ap.parse_args()
    shots = dataset_shots()
    if args.report:
        report(shots)
        return
    papers = dedupe(arxiv_list() + openalex_list())
    print(f"papers since {FROM_DATE}: {len(papers)} distinct "
          f"(with a pdf link: {sum(1 for p in papers if p.get('pdf'))})")
    n = fetch_pdfs(papers)
    print(f"full text after the direct download: {n}")
    n += fetch_fallback(papers, budget_s=float(os.environ.get("LIT_CHASE_BUDGET_S", 0)) or None)
    # NOTE: the arXiv-by-title fallback earns its keep only if the corpus is clean. Over a
    # full pass it recovered exactly one paper -- and that paper was an EEG eye-state
    # classifier whose abstract mentions the Weka "KStar" classifier. The title match was
    # correct; the corpus was not.
    by_source = {}
    for p in papers:
        if p.get("file"):
            src = p.get("source", "?").split(":")[0]
            by_source[src] = by_source.get(src, 0) + 1
    print(f"full text after the chase: {n} of {len(papers)}   {by_source}")
    hits = scan(papers, shots)
    held = [p for p in papers if p.get("file")]
    offtopic = [p for p in held if p.get("offtopic")]
    relevant_missing = [p for p in papers if not p.get("file")
                        and FUSION_TITLE_RE.search(p.get("title") or "")]
    n_rel_held = len(held) - len(offtopic)
    n_rel_pop = n_rel_held + len(relevant_missing)
    print(f"\ncorpus: {len(papers)} papers matched 'KSTAR', {len(held)} with full text, "
          f"of which {len(offtopic)} are NOT about fusion (Weka K*, K* search, the KSTAR "
          f"kinase paper, ...)")
    print(f"fusion-relevant coverage: {n_rel_held} full texts of ~{n_rel_pop} relevant "
          f"papers ({100 * n_rel_held / max(n_rel_pop, 1):.0f} %)  <- quote THIS, not "
          f"{len(held)}/{len(papers)}")
    (HERE / "literature_hits.json").write_text(
        json.dumps({"confirmed": {str(k): v for k, v in CONFIRMED.items()},
                    "false_positives": {str(k): v for k, v in FALSE_POSITIVES.items()},
                    "sweep_hits": hits, "n_papers": len(papers), "n_fulltext": n,
                    "coverage": {"by_source": by_source,
                                 "n_offtopic_fulltext": len(offtopic),
                                 "n_relevant_fulltext": n_rel_held,
                                 "n_relevant_population": n_rel_pop,
                                 "still_missing": [
                        {"title": p["title"], "doi": p["doi"], "why": p.get("miss", "?")}
                        for p in papers if not p.get("file")]}},
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
