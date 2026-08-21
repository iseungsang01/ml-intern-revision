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
  * SUPPLEMENTARY material is swept for Nature-family papers only (`springer_fulltext`).
    It had to be: the Supplementary Information of 10.1038/s41467-024-45454-1 names
    #31184, #31185 and #31189, none of which appear in the main text. Which publishers
    this reaches was measured rather than assumed -- nature.com HTML and
    static-content.springer.com SI files serve this script, IOP answers with a Radware
    bot-challenge page and AIP with 403, so IOP/AIP/Elsevier supplements stay unread.
  * The two FIRE-mode papers are IOP-blocked to this script and were read through their
    article pages. Doing that is what turned up #31923 in the SECOND of them, which the
    PDF sweep could never have seen.
  * Paywalled papers, conference proceedings, and theses are invisible here.
  * The two FIRE-mode papers that give us #31921 and #31923 were read through their
    abstract/HTML pages, not this sweep; they are recorded in CONFIRMED below.
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures as cf
import difflib
import glob
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
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
# `fusion` on its own is not a plasma-physics word. It let in "Fusexins, HAP2/GCS1 and
# Evolution of Gamete Fusion", a cell-biology paper, which then contributed two spurious
# shots. Every token here has to be one that a membrane-biology title cannot carry.
FUSION_TITLE_RE = re.compile(
    r"tokamak|plasma|divertor|pedestal|disrupt|KSTAR|\bITER\b|stellarator|"
    r"magnetohydro|\bMHD\b|gyrokinet|cyclotron|tearing mode|scrape-?off|H-mode|"
    r"\bELM\b|confinement|fusion (energy|reactor|performance|power|device|born|alpha)",
    re.I)
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
    # Hand-verified 2026-08-21 from the full PDF (승상님 fetched it through the IOP wall).
    # The index hit was genuine: IOP typesets the number with a thin thousands space,
    # "#32 092", which is why the bare-digit grep of the extracted text missed it.
    32092: [("Spatiotemporal structure of edge harmonic oscillation and its role in "
             "ELM-free QH-mode at KSTAR", "10.1088/1741-4326/ae8679",
             "THE representative QH-mode discharge (6 mentions): fig.2 time evolution, "
             "fig.3 kinetic profiles, fig.4 peeling-ballooning at t = 4.9 s (ELMy) vs "
             "t = 7.6 s (ELM-free), table 1: EHO n = 1 (+harmonics 2,3,4) at ~4 kHz "
             "(n=1) / ~8 kHz (n=2), measured on Mirnov coils + ECEI -- exactly the band "
             "the 100 Hz grid aliases away")],
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
    31589: "DOI fragment 10.1038/s41467-022-31589-6 (Bierwage et al., Nat. Commun. 13, "
           "3941) -- verified in the reference list of 10.1007/s41614-025-00199-2",
    # Discharge numbers are not unique across machines, and this one is the clearest case:
    # the NLED-AUG benchmark equilibrium is "the AUG shot considered is the #31213, at
    # t = 0.84 s" (10.1007/s41614-025-00199-2, sect. 2). ASDEX Upgrade, not KSTAR.
    31213: "ASDEX Upgrade shot #31213 (NLED-AUG benchmark case), not a KSTAR discharge",
    31886: "five-digit number inside a cell-biology paper ('Fusexins, HAP2/GCS1 and "
           "Evolution of Gamete Fusion'); it passed an earlier title filter on 'Fusion'",
    31913: "same cell-biology paper as 31886",
    # Fourth class, established 2026-08-21: AIP article numbers. Physics of Plasmas ids are
    # iissnn (issue, section, sequence) -- 032302, 032111, 032309 ... -- and OpenAlex's
    # full-text index matches the bare five digits inside them, so any KSTAR paper whose
    # reference list cites a PoP issue-3 article "hits" the matching campaign shot number.
    # Proof by hand: the readable (arXiv/Springer/Nature) versions of the indexed works
    # contain the ZERO-PADDED id in their bibliographies and never the bare number:
    32302: "PoP id: 'Phys. Plasmas 13(3) 032302 (2006)' in arXiv:2412.09522 refs",
    32305: "PoP id: 'Phys. Plasmas 28(3) 032305 (2021)' in arXiv:2201.07941 refs",
    32111: "PoP id: 'Phys. Plasmas 26(3) 032111' + '24, 032111' in two readable refs",
    32115: "PoP id: 'Phys. Plasmas 31, 032115 (2024)' in arXiv:2410.11498 refs",
    32309: "PoP id: 'Phys. Plasmas 20, 032309 (2013)' in arXiv:2306.05607 refs",
    # Same mechanism, established indirectly: every readable version of the indexed works
    # lacks the bare number, works predate the campaign (arXiv 2207.06610 = July 2022) or
    # belong to other machines (NSTX, stellarator), and PoP ids 032303/032304/032308/032310
    # are real citable articles. 32310's ONLY indexed work is shared with 32303's list.
    32303: "AIP id collision (readable versions of all 4 indexed works lack the number; "
           "one indexed work predates the 2022 campaign)",
    32304: "AIP id collision (NSTX paper among the works; readable versions negative)",
    32308: "AIP id collision (3 readable versions negative, incl. a work predating the "
           "campaign window)",
    32310: "AIP id collision (only work is shared with 32303's list -- one unreadable "
           "PoP paper indexing both numbers = two reference-list ids)",
    32004: "plausible AIP id collision (032004 fits the iissnn format, unlike 031097 from "
           "the same unreadable PoP 5.0237640) + a DIII-D paper among the works; context "
           "unreadable -- remove this entry if the paper is ever read and names the shot",
    32151: "five-digit number inside a malaria cell-biology paper (PLoS Pathogens "
           "'Spinster-like Transporter...'), the 31886 class again",
}

# A tokamak paper that is not a KSTAR paper is not evidence about a KSTAR discharge. ASDEX
# Upgrade, DIII-D, EAST, JET and JT-60 all number shots in five digits and their ranges
# overlap the 2022 KSTAR campaign, so a bare number in a generic paper is ambiguous by
# construction -- that is exactly how #31213 entered the ledger as an AUG shot.
KSTAR_RE = re.compile(r"KSTAR", re.I)
OTHER_MACHINE_RE = re.compile(
    r"ASDEX|\bAUG\b|DIII-?D|\bJET\b|\bEAST\b|JT-?60|W7-X|\bTCV\b|\bMAST\b|"
    r"HL-2A|\bNSTX\b|TFTR|Tore Supra|WEST", re.I)


def hit_verdict(shot, work):
    """How much a full-text index hit is worth, without pretending it is worth more.

    `rejected`   hand-checked and known not to be a discharge of ours.
    `confirmed`  hand-verified against the paper itself.
    `kstar`      the citing paper is about KSTAR, so a number in campaign range is
                 plausibly one of its discharges -- still not proof, but attributable.
    `unverified` a fusion paper that never names KSTAR. Could be another machine's shot,
                 a DOI fragment or a page number. Never select on this alone.
    """
    if shot in FALSE_POSITIVES:
        return "rejected"
    if shot in CONFIRMED:
        return "confirmed"
    return "kstar" if KSTAR_RE.search(work.get("title") or "") else "unverified"


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
    otherwise inflate both the paper count and the per-shot mention count.

    Merge rather than pick: the arXiv record almost always carries the readable PDF while
    only the OpenAlex record carries the publisher DOI, and dropping either loses something.
    Keeping just the arXiv record is what hid the Nature Communications ELM paper from the
    supplementary pass -- its surviving `doi` was `arXiv:2405.05452`, so a `10.1038/` test
    could not see it, even though that paper is one of the screen's most important hits.
    """
    best = {}
    for p in papers:
        k = norm_title(p["title"])[:120] or p["key"]
        cur = best.get(k)
        if cur is None:
            best[k] = dict(p)
            continue
        if not cur.get("pdf") and p.get("pdf"):
            cur["pdf"] = p["pdf"]
            cur["key"] = p["key"]
        cur_doi, new_doi = cur.get("doi") or "", p.get("doi") or ""
        if new_doi and (not cur_doi or cur_doi.lower().startswith("arxiv:")
                        or "arxiv" in cur_doi.lower()) and "arxiv" not in new_doi.lower():
            cur["doi"] = new_doi
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


# Nature serves supplementary files from media.springernature.com in the article HTML and
# from static-content.springer.com in older/redirected links; both are live, so match both.
# (Scraping the link beats constructing it: the MOESM index is not derivable from the DOI.)
SPRINGER_ESM_RE = re.compile(
    r"https://(?:media\.springernature\.com/original/springer-static|static-content\.springer\.com)"
    r"/esm/[^\"'\s<>\\]+")
TAG_RE = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)


def html_to_text(raw):
    import html as _html
    return " ".join(_html.unescape(TAG_RE.sub(" ", raw)).split())


def springer_fulltext(papers):
    """Nature-family articles, plus their SUPPLEMENTARY files.

    This is the gap the first pass named and did not close. The Supplementary Information
    of 10.1038/s41467-024-45454-1 cites three discharges that appear nowhere in its main
    text, so "not in the paper" was really "not in the part of the paper we downloaded".

    Publisher routes were measured, not assumed (2026-08-18): nature.com HTML and
    static-content.springer.com SI PDFs both serve this script; IOP returns its Radware
    challenge page and AIP returns 403. So this pass covers 10.1038 DOIs and nothing else,
    and the coverage line stays honest about the rest.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    extras, seen_art = [], set()
    targets = [p for p in papers
               if "10.1038/" in (p.get("doi") or "") and FUSION_TITLE_RE.search(p.get("title") or "")]
    print(f"springer/nature pass: {len(targets)} Nature-family papers")
    for i, p in enumerate(targets, 1):
        suffix = (p["doi"].split("10.1038/", 1)[1]).strip("/")
        if not suffix or suffix in seen_art:
            continue
        seen_art.add(suffix)
        art_txt = CACHE / f"nature_{suffix}.txt"
        raw = ""
        if art_txt.exists():
            raw = art_txt.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                req = urllib.request.Request(f"https://www.nature.com/articles/{suffix}",
                                             headers={"User-Agent": UA})
                page = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "replace")
            except Exception as exc:
                print(f"  [{i}/{len(targets)}] {suffix}: article {type(exc).__name__}")
                page = ""
            if page:
                raw = page
                art_txt.write_text(page, encoding="utf-8")
        if not raw:
            continue
        extras.append({**p, "key": f"{p['key']}-nature", "file": str(art_txt),
                       "source": "nature-html", "title": p["title"] + " [main text, HTML]"})
        for j, esm in enumerate(dict.fromkeys(SPRINGER_ESM_RE.findall(raw)), 1):
            esm = esm.rstrip("&quot;").rstrip(",.;")
            ext = ".pdf" if esm.lower().endswith(".pdf") else Path(esm).suffix or ".bin"
            dst = CACHE / f"esm_{suffix}_{j}{ext}"
            if not dst.exists():
                if not try_download(esm, dst):
                    continue
                time.sleep(0.5)
            extras.append({**p, "key": f"{p['key']}-esm{j}", "file": str(dst),
                           "source": "springer-esm",
                           "title": p["title"] + f" [supplementary {j}]"})
        print(f"  [{i}/{len(targets)}] {suffix}: "
              f"{sum(1 for e in extras if e['key'].startswith(p['key']))} scan targets", flush=True)
    return extras


class DailyBudgetExhausted(RuntimeError):
    """OpenAlex's metered daily allowance is gone; nothing but waiting refills it."""


def _retry_after_seconds(exc):
    """Seconds OpenAlex asks us to wait, or None if it did not say."""
    try:
        raw = (exc.headers.get("retry-after") or "").strip()
    except Exception:
        return None
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    try:                                        # HTTP-date form
        from email.utils import parsedate_to_datetime
        import datetime as _dt
        when = parsedate_to_datetime(raw)
        now = _dt.datetime.now(_dt.timezone.utc)
        return max(0, int((when - now).total_seconds()))
    except Exception:
        return None


# A 429 that clears in a minute and a 429 that clears at midnight UTC are the same status
# code and completely different situations. Anything past this many seconds is the daily
# budget, not a burst limit.
QUOTA_SECONDS = 900


def openalex_api(url, tries=6):
    """OpenAlex, distinguishing a burst limit from the metered daily budget.

    The earlier version treated every 429 as burst throttling and escalated the wait to
    minutes. That was the wrong read: OpenAlex now meters this API per request, and when
    the free daily allowance is gone it answers 429 with `retry-after` in the *hours* --
    `Insufficient budget ... Resets at midnight UTC`. No backoff outlasts that, and since
    each retry is itself a billable request, waiting simply spends more of tomorrow's
    budget on nothing.

    So: read `retry-after` first. Past QUOTA_SECONDS it is the daily quota and we raise
    immediately -- the scans are resumable, so stopping costs one request, not the pass.
    Below it, it really is a burst limit and the old backoff is right.
    """
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ces-lit/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            after = _retry_after_seconds(exc)
            if after is not None and after > QUOTA_SECONDS:
                try:
                    body = exc.read().decode("utf-8", "replace")[:200]
                except Exception:
                    body = ""
                raise DailyBudgetExhausted(
                    f"daily budget spent; retry-after {after}s "
                    f"(~{after / 3600:.1f} h). {body}") from None
            if attempt == tries - 1:
                raise
            wait = after if after else 30 * (2 ** attempt)
            print(f"    429 (burst, retry-after {after}); waiting {wait}s before "
                  f"retry {attempt + 2}/{tries}", flush=True)
            time.sleep(wait)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fulltext_index_scan(shots, pace=0.8, control_n=20):
    """Ask OpenAlex's full-text index for each shot number, instead of grepping PDFs.

    This inverts the sweep, and it reaches what the sweep cannot. Validated against four
    shots already known to be published, it returns exactly the right papers -- including
    10.1088/1741-4326/adacfc, an IOP article that has never once been downloadable by this
    script. Coverage stops being bounded by what we can fetch and starts being bounded by
    what OpenAlex has indexed, which is a much larger set.

    Two things keep the result honest:
      * a paper published before the 2022 campaign cannot be citing its discharges, so
        anything older than 2022 is a coincidence and is dropped;
      * `control_n` five-digit numbers from OUTSIDE the campaign range are queried the same
        way. Every hit there is by construction a false positive -- a page number, an
        identifier, a year -- so that rate is this screen's measured coincidence rate,
        reported next to the real hits rather than assumed to be zero.

    OpenAlex meters this API: the free daily allowance is a few hundred requests and a 429
    says `Resets at midnight UTC`. 641 shots does not fit in one day's budget, so the scan
    is RESUMABLE -- every answered shot is written to `fulltext_index_hits.json` as it is
    learned, and a rerun skips them. Interruption costs the current shot, not the pass.
    """
    kinase = "10.1038/s41467-022-32017-5"   # a paper literally titled KSTAR, about kinases
    out_path = HERE / "fulltext_index_hits.json"
    state = {"hits": {}, "queried": [], "control_numbers_tested": 0, "control_hits": 0}
    if out_path.exists():
        try:
            state.update(json.loads(out_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    done = set(state.get("queried") or [])

    def save():
        state["queried"] = sorted(done)
        state["control_rate"] = (state["control_hits"]
                                 / max(state["control_numbers_tested"], 1))
        out_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")

    def query(n):
        url = ("https://api.openalex.org/works?filter=" +
               urllib.parse.quote(f"fulltext.search:KSTAR {n}", safe=":,") +
               f"&per-page=10&select=id,doi,title,publication_year&mailto={MAIL}")
        out = []
        for w in openalex_api(url)["results"]:
            title, doi = w.get("title") or "", (w.get("doi") or "")
            year = w.get("publication_year") or 0
            if kinase in doi.lower() or not FUSION_TITLE_RE.search(title) or year < 2022:
                continue
            out.append({"title": title, "doi": doi, "year": year})
        return out

    todo = [s for s in sorted(shots) if s not in done]
    print(f"index scan: {len(done)} shots already answered, {len(todo)} to go")
    for i, s in enumerate(todo, 1):
        try:
            found = query(s)
        except Exception as exc:
            code = getattr(exc, "code", "")
            print(f"  index scan stopped at #{s}: {type(exc).__name__} {code} "
                  f"({len(done)}/{len(shots)} answered; rerun to resume)", flush=True)
            break
        done.add(s)
        if found:
            for w in found:
                w["verdict"] = hit_verdict(s, w)
            state["hits"][str(s)] = found
            print(f"  INDEX HIT #{s}: " + " | ".join(
                f"{w['verdict']} {w['year']} {w['title'][:56]}" for w in found), flush=True)
        if i % 25 == 0:
            save()
            print(f"  index scan {len(done)}/{len(shots)}", flush=True)
        time.sleep(pace)
    save()

    base = max(shots) + 5000
    for j in range(control_n):
        n = base + j * 137
        if n in done:
            continue
        try:
            found = query(n)
        except Exception:
            break
        done.add(n)
        state["control_numbers_tested"] += 1
        if found:
            state["control_hits"] += 1
            print(f"  control {n} (out of range) returned {len(found)}: "
                  f"{found[0]['title'][:60]}", flush=True)
        time.sleep(pace)
    save()
    print(f"\nindex scan: {len(state['hits'])} hits over "
          f"{len([s for s in done if s in shots])}/{len(shots)} shots answered; "
          f"coincidence control {state['control_hits']}/{state['control_numbers_tested']}")
    return state


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
            if p["file"].lower().endswith((".txt", ".html", ".htm")):
                text = html_to_text(Path(p["file"]).read_text(encoding="utf-8", errors="replace"))
                head = text[:12000]
            else:
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



KINASE_DOI = "10.1038/s41467-022-32017-5"   # a paper literally titled KSTAR, about kinases


def batch_ask(group, per_page=25):
    """Ask OpenAlex's full-text index about a whole group of shot numbers in one request.

    `fulltext.search:KSTAR AND (a OR b OR ...)` -- the boolean form is what makes a 641
    shot sweep affordable under a metered API. It is also the single point of failure of
    this whole screen: if the operators are not honoured, every batch comes back empty and
    641 shots read as `absent from the literature` when they were never actually asked
    about. `syntax_control()` is what stops that reading, and it calls THIS function, not
    a copy of it.
    """
    expr = " OR ".join(str(g) for g in group)
    url = ("https://api.openalex.org/works?filter=" +
           urllib.parse.quote(f"fulltext.search:KSTAR AND ({expr})", safe=":,") +
           f"&per-page={per_page}&select=id,doi,title,publication_year&mailto={MAIL}")
    found = []
    for w in openalex_api(url)["results"]:
        title, doi = w.get("title") or "", (w.get("doi") or "")
        year = w.get("publication_year") or 0
        if KINASE_DOI in doi.lower() or not FUSION_TITLE_RE.search(title) or year < 2022:
            continue
        found.append({"title": title, "doi": doi, "year": year})
    return found


# Shots whose publication is hand-verified in CONFIRMED and which the batch form therefore
# HAS to return. They are the positive control.
CONTROL_POSITIVE = [31921, 31359, 31873, 32027]


def decoys(shots, n, start=5000, step=137):
    """`n` five-digit numbers from outside the campaign -- nothing can legitimately cite
    them, so a hit on one is this screen's coincidence rate, not a discharge."""
    base = max(shots) + start
    return [base + j * step for j in range(n)]


def syntax_control(shots, pace=1.0, batch=32):
    """Two requests that decide whether the batched scan means anything.

    A batch that returns nothing is ambiguous by construction -- it means either `no paper
    cites these 32 shots` or `the query did not run`. One request each way separates them:

      POSITIVE  four hand-verified published shots, hidden among out-of-range decoys.
                Empty here means the operators are being dropped and every negative in the
                sweep would be an artefact. Stop.
      NEGATIVE  the same batch shape with decoys ONLY. Hits here mean the numbers are being
                ignored and `KSTAR` alone is driving the match, so every positive in the
                sweep would be an artefact. Also stop.

    Passing both is what licenses reading a batch miss as real absence.
    """
    filler = decoys(shots, batch - len(CONTROL_POSITIVE))
    pos_group = sorted(CONTROL_POSITIVE + filler)
    neg_group = decoys(shots, batch, start=9000, step=211)

    print("=" * 78)
    print(f"BATCH SYNTAX CONTROL  (2 requests, batch of {batch})")
    print(f"  positive: {CONTROL_POSITIVE} + {len(filler)} out-of-range decoys")
    got_pos = batch_ask(pos_group)
    for w in got_pos:
        print(f"    {w['year']}  {w['title'][:78]}\n           {w['doi']}")
    print(f"  -> {len(got_pos)} fusion papers")
    time.sleep(pace)

    print(f"  negative: {batch} out-of-range decoys, no real shot")
    got_neg = batch_ask(neg_group)
    for w in got_neg:
        print(f"    {w['year']}  {w['title'][:78]}\n           {w['doi']}")
    print(f"  -> {len(got_neg)} fusion papers")

    ok = bool(got_pos) and not got_neg
    print("-" * 78)
    if not got_pos:
        print("FAIL (positive empty): the AND/OR form is not being honoured. Every batch\n"
              "     would come back empty and that is NOT evidence of absence. Do not run\n"
              "     the sweep; fix the query form first.")
    elif got_neg:
        print(f"FAIL (negative returned {len(got_neg)}): the shot numbers are not\n"
              "     constraining the match -- `KSTAR` alone is. Every batch would look\n"
              "     positive and bisection would burn the budget chasing nothing.")
    else:
        print("PASS: positives are found, decoys are not. A batch miss can be read as the\n"
              "      shot being absent from the indexed literature.")
    print("=" * 78)
    return ok, got_pos, got_neg


def vt_priority(shots):
    """Shots ordered by how many valid `V_rot` rows they carry, richest first.

    Order matters because the budget runs out mid-sweep, not because the query changes.
    `V_rot` is the target no arm beats causal GP on, so valid `V_rot` rows are the first
    selection metric (SELECTION.md); and the open question -- one literature set, or a
    literature tier plus a separate `V_rot` tier -- turns only on whether a shot with
    BOTH a citation and >= 200 valid `V_rot` rows exists. Shots with almost no `V_rot`
    cannot answer that no matter what the literature says, so they go last.

    Falls back to numeric order if the metrics table is missing.
    """
    path = HERE / "shot_metrics.csv"
    if not path.exists():
        return sorted(shots)
    rank = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rank[int(row["shot"])] = float(row["vt_clean_n"] or 0)
            except (TypeError, ValueError):
                continue
    return sorted(shots, key=lambda s: (-rank.get(s, -1.0), s))


def batched_index_scan(shots, batch=32, pace=1.0, priority=False):
    """The same index screen, but asking about 32 shots per request instead of one.

    OpenAlex moved to a metered budget: the free daily allowance ran out after ~100
    requests with `Insufficient budget ... Resets at midnight UTC`, which puts 641
    one-shot queries six days away. The API supports uppercase boolean operators inside
    a search filter, so a batch asks `KSTAR AND (a OR b OR ... )` and one request rules
    out 32 shots at a time. Positive batches are then bisected, which costs ~log2(32) = 5
    requests per shot that is actually cited.

    Expected cost at the hit rate measured on the first 100 shots (2 hits): ~20 batch
    requests plus ~5 per hit -- inside one day's allowance instead of six days'.

    Resumable at batch granularity, and it shares `fulltext_index_hits.json` with the
    one-at-a-time scan, so shots already answered are skipped and the two can be mixed.
    """
    out_path = HERE / "fulltext_index_hits.json"
    state = {"hits": {}, "queried": [], "control_numbers_tested": 0, "control_hits": 0}
    if out_path.exists():
        try:
            state.update(json.loads(out_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    done = set(state.get("queried") or [])

    def save():
        state["queried"] = sorted(done)
        state["control_rate"] = (state["control_hits"]
                                 / max(state["control_numbers_tested"], 1))
        out_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")

    def ask(group):
        found = batch_ask(group)
        time.sleep(pace)
        return found

    def bisect(group, known):
        """`group` is known positive; split until each hit is attributed to one shot."""
        if len(group) == 1:
            shot = group[0]
            for w in known:
                w["verdict"] = hit_verdict(shot, w)
            state["hits"][str(shot)] = known
            worth = max((w["verdict"] for w in known),
                        key=["rejected", "unverified", "kstar", "confirmed"].index)
            print(f"  INDEX HIT #{shot} [{worth}]: " + " | ".join(
                f"{w['year']} {w['title'][:60]} {w['doi']}" for w in known), flush=True)
            return
        mid = len(group) // 2
        first, second = group[:mid], group[mid:]
        got_first = ask(first)
        if got_first:
            bisect(first, got_first)
        # The parent batch is already paid for, so use what it said. Any paper it returned
        # that the first half does not account for has to be cited from the second half --
        # that half is positive and asking again would only re-buy the same answer. Only
        # when the first half explains everything is the second half genuinely unknown.
        seen = {w["doi"] for w in got_first}
        unexplained = [w for w in known if w["doi"] not in seen]
        if unexplained:
            bisect(second, unexplained)
        else:
            got_second = ask(second)
            if got_second:
                bisect(second, got_second)

    order = vt_priority(shots) if priority else sorted(shots)
    todo = [s for s in order if s not in done]
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    print(f"batched index scan: {len(done)} answered, {len(todo)} to go "
          f"in {len(groups)} groups of {batch}"
          f"{' (richest V_rot first)' if priority else ''}", flush=True)
    for gi, group in enumerate(groups, 1):
        try:
            found = ask(group)
            if found:
                bisect(group, found)
        except Exception as exc:
            code = getattr(exc, "code", "")
            print(f"  stopped in group {gi}/{len(groups)}: {type(exc).__name__} {code} "
                  f"({len(done)} answered; rerun to resume)", flush=True)
            save()
            return
        done.update(group)
        save()
        print(f"  group {gi}/{len(groups)} done ({len(done)}/{len(shots)} shots)", flush=True)
    save()
    print(f"batched scan complete: {len(state['hits'])} shots cited in the literature")

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
    ap.add_argument("--pace", type=float, default=0.8,
                    help="seconds between index-scan queries (raise it if 429s persist)")
    ap.add_argument("--control", action="store_true",
                    help="2-request positive/negative check that the batch query form works")
    ap.add_argument("--priority", action="store_true",
                    help="scan shots with the most valid V_rot first (budget runs out mid-sweep)")
    ap.add_argument("--batched", action="store_true",
                    help="index scan in OR-batches of 32 (OpenAlex is metered per request)")
    ap.add_argument("--fulltext-index", action="store_true",
                    help="query OpenAlex's full-text index per shot (reaches IOP, no PDFs)")
    args = ap.parse_args()
    shots = dataset_shots()
    if args.report:
        report(shots)
        return
    if args.control:
        ok, _, _ = syntax_control(shots, pace=args.pace)
        raise SystemExit(0 if ok else 1)
    if args.fulltext_index:
        if args.batched:
            batched_index_scan(shots, pace=args.pace, priority=args.priority)
        else:
            fulltext_index_scan(shots, pace=args.pace)   # resumable; writes as it goes
        return
    papers = dedupe(arxiv_list() + openalex_list())
    print(f"papers since {FROM_DATE}: {len(papers)} distinct "
          f"(with a pdf link: {sum(1 for p in papers if p.get('pdf'))})")
    # A corpus pull can fail SOFTLY: OpenAlex meters its API, and once the daily budget is
    # spent `openalex_list` returns nothing and the run continues on the ~66 arXiv papers
    # alone. That would overwrite a good literature_hits.json with a far worse one and the
    # coverage line would simply report the smaller number as if it were the finding. Refuse.
    prev = HERE / "literature_hits.json"
    if prev.exists():
        try:
            before = json.loads(prev.read_text(encoding="utf-8")).get("n_papers", 0)
        except Exception:
            before = 0
        if before and len(papers) < 0.5 * before:
            raise SystemExit(
                f"corpus collapsed: {len(papers)} papers now vs {before} in "
                f"literature_hits.json. The pull failed (OpenAlex rate limit resets at "
                f"midnight UTC); refusing to overwrite the artifact with a partial sweep.")
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
    # Supplementary material is scanned as its own set of targets. It is deliberately kept
    # out of the coverage arithmetic below: an SI file is not another paper, and counting it
    # as one would inflate exactly the ratio this script exists to keep honest.
    extras = springer_fulltext(papers)
    hits = scan(papers + extras, shots)
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
                                 "n_supplementary_targets": len(extras),
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
