"""Semantic Scholar enrichment — port of paper-orchestra utils/scholar_utils.py.

Pipeline role (Phase 1b of the orchestra lit-review agent): take LLM-discovered
candidate papers {title, year, reason, section}, verify each against the real
Semantic Scholar Graph API, keep only matches with fuzzy title ratio > 70
(+10 bonus when the year matches), enrich with authors/venue/abstract/citations,
dedup by normalized title, and emit enriched JSON + a generated .bib file.

Differences from upstream (documented deviations):
- difflib instead of thefuzz (no extra dependency; same 0-100 ratio scale).
- Keyless operation hardened: exponential backoff on 429/5xx (S2 keyless pool
  is rate-limited), 1.1 s pacing between calls (upstream used 1.0 s).
- Papers without an abstract are KEPT but flagged (upstream dropped them);
  fusion instrument papers often lack abstracts on S2, and targeted must-have
  citations cannot be dropped. The relevance filter sees the flag.
- Adds --keyword mode: direct S2 relevance search (upstream only did title
  enrichment); used to widen the novelty-check recall.

Usage:
  py s2_enrich.py --candidates candidates.json --out enriched.json --bib refs_s2.bib
  py s2_enrich.py --keyword "charge exchange spectroscopy neural network" --limit 20 --out kw.json
"""

import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,venue,year,abstract,citationCount,journal,publicationDate,externalIds"
OPENALEX = "https://api.openalex.org/works"
MAILTO = "lss010330@snu.ac.kr"  # OpenAlex polite pool
PACING_S = 1.1
MAX_RETRIES = 4


def http_get_json(url: str) -> dict:
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ces-litreview/1.0"})
            with urllib.request.urlopen(req, timeout=15) as rsp:
                return json.loads(rsp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                print(f"    .. HTTP {e.code}, backoff {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
    return {}


def fuzz_ratio(a: str, b: str) -> float:
    return 100.0 * difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def s2_title_search(title_query: str, year_hint=None):
    """Port of scholar_utils.s2_title_search (keyless, no cutoff — our draft is current)."""
    q = urllib.parse.quote(title_query)
    data = http_get_json(f"{S2_SEARCH}?query={q}&limit=3&fields={FIELDS}")
    results = data.get("data") or []
    best, best_ratio = None, 0.0
    for r in results:
        if not r.get("title"):
            continue
        ratio = fuzz_ratio(title_query, r["title"])
        if year_hint and r.get("year") == year_hint:
            ratio += 10
        if ratio > best_ratio:
            best_ratio, best = ratio, r
    if best_ratio > 70:
        return best, best_ratio
    return None, best_ratio


def s2_keyword_search(query: str, limit: int = 20):
    q = urllib.parse.quote(query)
    data = http_get_json(f"{S2_SEARCH}?query={q}&limit={limit}&fields={FIELDS}")
    return data.get("data") or []


# --- OpenAlex fallback backend (keyless, generous limits) -------------------

def _oa_abstract(inv_idx) -> str:
    if not inv_idx:
        return ""
    pos = {}
    for word, idxs in inv_idx.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _oa_to_s2_shape(w: dict) -> dict:
    """Convert an OpenAlex work into the S2 result shape the rest expects."""
    src = ((w.get("primary_location") or {}).get("source") or {})
    biblio = w.get("biblio") or {}
    pages = None
    if biblio.get("first_page"):
        pages = biblio["first_page"]
        if biblio.get("last_page"):
            pages += f"--{biblio['last_page']}"
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    return {
        "title": w.get("title") or w.get("display_name"),
        "authors": [{"name": (a.get("author") or {}).get("display_name", "?")}
                    for a in (w.get("authorships") or [])],
        "venue": src.get("display_name") or "",
        "year": w.get("publication_year"),
        "abstract": _oa_abstract(w.get("abstract_inverted_index")),
        "citationCount": w.get("cited_by_count") or 0,
        "journal": {"name": src.get("display_name"),
                    "volume": biblio.get("volume"), "pages": pages}
                   if src.get("display_name") else None,
        "publicationDate": w.get("publication_date"),
        "externalIds": {"DOI": doi},
    }


def openalex_title_search(title_query: str, year_hint=None):
    q = urllib.parse.quote(title_query)
    data = http_get_json(f"{OPENALEX}?search={q}&per-page=3&mailto={MAILTO}")
    best, best_ratio = None, 0.0
    for w in data.get("results") or []:
        t = w.get("title") or w.get("display_name")
        if not t:
            continue
        ratio = fuzz_ratio(title_query, t)
        if year_hint and w.get("publication_year") == year_hint:
            ratio += 10
        if ratio > best_ratio:
            best_ratio, best = ratio, w
    if best_ratio > 70:
        return _oa_to_s2_shape(best), best_ratio
    return None, best_ratio


def openalex_keyword_search(query: str, limit: int = 20):
    q = urllib.parse.quote(query)
    data = http_get_json(f"{OPENALEX}?search={q}&per-page={min(limit, 50)}&mailto={MAILTO}")
    return [_oa_to_s2_shape(w) for w in (data.get("results") or [])]


def title_search_any(title_query: str, year_hint=None, backend: str = "auto"):
    """Try S2 first, fall back to OpenAlex on failure (or use one directly)."""
    if backend in ("auto", "s2"):
        try:
            return (*s2_title_search(title_query, year_hint), "s2")
        except Exception as e:
            if backend == "s2":
                raise
            print(f"    .. S2 unavailable ({type(e).__name__}), using OpenAlex", flush=True)
    hit, ratio = openalex_title_search(title_query, year_hint)
    return hit, ratio, "openalex"


STOPWORDS = {"the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is",
             "are", "with", "by", "study"}


def generate_key(authors, year, title) -> str:
    """Port of HybridLiteratureAgent._generate_key."""
    if not authors:
        first = "Unknown"
    else:
        name = authors[0]["name"] if isinstance(authors[0], dict) else str(authors[0])
        first = name.split()[-1]
    clean_author = re.sub(r"[^a-zA-Z]", "", first).capitalize()
    year_str = str(year) if year else "2024"
    words = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower()).split()
    meaningful = [w.capitalize() for w in words if w not in STOPWORDS]
    title_part = "".join(meaningful[:2]) if meaningful else "Paper"
    return f"{clean_author}{year_str}{title_part}"


def to_record(s2: dict, cand: dict, match_ratio: float) -> dict:
    authors = s2.get("authors") or []
    j = s2.get("journal") or {}
    abstract = (s2.get("abstract") or "")[:1500]
    ext = s2.get("externalIds") or {}
    return {
        "citation_key": generate_key(authors, s2.get("year"), s2["title"]),
        "title": s2["title"],
        "authors": [a["name"] for a in authors],
        "venue": s2.get("venue") or "arXiv",
        "year": s2.get("year") or cand.get("year"),
        "abstract": abstract,
        "has_abstract": bool(abstract),
        "citation_count": s2.get("citationCount") or 0,
        "found_in_section": cand.get("section", "General"),
        "reason": cand.get("reason", ""),
        "journal": j.get("name"),
        "volume": j.get("volume"),
        "pages": j.get("pages"),
        "publication_date": s2.get("publicationDate"),
        "doi": ext.get("DOI"),
        "arxiv": ext.get("ArXiv"),
        "match_ratio": round(match_ratio, 1),
        "candidate_title": cand.get("title"),
    }


def generate_bibtex(papers) -> str:
    """Port of HybridLiteratureAgent._generate_bibtex (+ optional doi field)."""
    entries, seen = [], set()
    for p in papers:
        base = p["citation_key"]
        key, suffix = base, "a"
        while key in seen:
            key = f"{base}{suffix}"
            suffix = chr(ord(suffix) + 1)
        p["citation_key"] = key
        seen.add(key)
        author_str = " and ".join(p["authors"]) if p["authors"] else "Unknown"
        etype = "article" if p.get("journal") else "inproceedings"
        e = f"@{etype}{{{key},\n  title={{{p['title']}}},\n  author={{{author_str}}},\n"
        if p.get("journal"):
            e += f"  journal={{{p['journal']}}},\n"
        else:
            e += f"  booktitle={{{p['venue']}}},\n"
        e += f"  year={{{p['year']}}}"
        if p.get("volume"):
            e += f",\n  volume={{{p['volume']}}}"
        if p.get("pages"):
            e += f",\n  pages={{{p['pages']}}}"
        if p.get("doi"):
            e += f",\n  doi={{{p['doi']}}}"
        e += "\n}"
        entries.append(e)
    return "\n\n".join(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", help="JSON array of {title, year, reason, section}")
    ap.add_argument("--keyword", help="direct S2 relevance query instead of title enrichment")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bib")
    ap.add_argument("--backend", default="auto", choices=["auto", "s2", "openalex"])
    args = ap.parse_args()

    if args.keyword:
        try:
            rows = s2_keyword_search(args.keyword, args.limit)
        except Exception as e:
            print(f"  S2 unavailable ({type(e).__name__}), using OpenAlex", flush=True)
            rows = openalex_keyword_search(args.keyword, args.limit)
        out = [to_record(r, {"section": f"Keyword: {args.keyword}", "reason": ""}, 100.0)
               for r in rows if r.get("title")]
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"keyword '{args.keyword}': {len(out)} results -> {args.out}")
        return

    cands = json.load(open(args.candidates, encoding="utf-8"))
    registry = {}
    misses = []
    for i, cand in enumerate(cands):
        title = cand.get("title")
        if not title:
            continue
        norm = normalize_title(title)
        if norm in registry:
            continue
        time.sleep(PACING_S)
        try:
            s2, ratio, backend = title_search_any(title, cand.get("year"), args.backend)
        except Exception as e:
            print(f"  [{i+1}/{len(cands)}] ERROR '{title[:50]}': {e}", flush=True)
            misses.append({**cand, "error": str(e)})
            continue
        if s2 is None:
            print(f"  [{i+1}/{len(cands)}] MISS ({ratio:.0f}) '{title[:60]}'", flush=True)
            misses.append({**cand, "best_ratio": round(ratio, 1)})
            continue
        rec = to_record(s2, cand, ratio)
        rec["backend"] = backend
        final_norm = normalize_title(rec["title"])
        if final_norm in registry:
            continue
        registry[final_norm] = rec
        flag = "" if rec["has_abstract"] else " [no-abstract]"
        print(f"  [{i+1}/{len(cands)}] OK ({ratio:.0f}, {backend}) {rec['title'][:60]}{flag}", flush=True)

    papers = list(registry.values())
    json.dump({"papers": papers, "misses": misses},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n{len(papers)} enriched, {len(misses)} misses -> {args.out}")
    if args.bib:
        open(args.bib, "w", encoding="utf-8").write(generate_bibtex(papers))
        print(f"bibtex -> {args.bib}")


if __name__ == "__main__":
    main()
