"""OpenAlex helper for the novelty critique (2026-11-20).

Usage (from the repo root):
  py docs/paper/review_loop/critique_20261120_novelty_checks/oa.py search "<query>" [n] [extra_filter]
  py docs/paper/review_loop/critique_20261120_novelty_checks/oa.py full "<phrase>" [n] [extra_filter]
  py docs/paper/review_loop/critique_20261120_novelty_checks/oa.py doi <doi>
  py docs/paper/review_loop/critique_20261120_novelty_checks/oa.py cites <openalex_id> [n] [search]

No personal or project data is sent; no mailto parameter; generic user agent.
Every call is appended to log.jsonl next to this script (query, ids, titles) so
the critique can state what was retrieved. Abstracts are reconstructed from
abstract_inverted_index. ASCII-only printing.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "log.jsonl")
UA = "Mozilla/5.0 (literature-check script)"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 + 2 * attempt)


def abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = []
    for word, idx in inv.items():
        for i in idx:
            pos.append((i, word))
    pos.sort()
    return " ".join(p[1] for p in pos)


def asc(s):
    return (s or "").encode("ascii", "replace").decode("ascii")


def show(works, tag, query, full_abs=True):
    rec = {"tag": tag, "query": query, "hits": []}
    for w in works:
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        doi = w.get("doi")
        ab = abstract(w)
        rec["hits"].append({"id": w.get("id"), "doi": doi, "title": w.get("display_name"),
                            "year": w.get("publication_year"), "venue": venue,
                            "cited_by": w.get("cited_by_count")})
        print("=" * 100)
        print(asc("%s | %s | %s | cited %s" % (w.get("publication_year"), venue, doi, w.get("cited_by_count"))))
        print(asc("ID " + str(w.get("id"))))
        print(asc("TITLE: " + str(w.get("display_name"))))
        if ab:
            lim = int(os.environ.get("OA_ABS", "900"))
            print(asc("ABSTRACT: " + (ab if (full_abs and lim <= 0) else ab[:lim])))
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def main():
    mode = sys.argv[1]
    if mode == "search":
        q = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        filt = sys.argv[4] if len(sys.argv) > 4 else ""
        url = BASE + "?search=" + urllib.parse.quote(q) + "&per-page=%d" % n
        if filt:
            url += "&filter=" + urllib.parse.quote(filt, safe=":,|<>")
        show(get(url).get("results", []), "search", q + " | " + filt)
    elif mode == "full":
        q = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        filt = "fulltext.search:" + q
        if len(sys.argv) > 4 and sys.argv[4]:
            filt += "," + sys.argv[4]
        url = BASE + "?filter=" + urllib.parse.quote(filt, safe=":,|<>") + "&per-page=%d" % n
        show(get(url).get("results", []), "fulltext", filt)
    elif mode == "doi":
        d = sys.argv[2]
        url = BASE + "/https://doi.org/" + d
        show([get(url)], "doi", d)
    elif mode == "cites":
        wid = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        q = sys.argv[4] if len(sys.argv) > 4 else ""
        url = BASE + "?filter=cites:" + wid + "&per-page=%d" % n
        if q:
            url += "&search=" + urllib.parse.quote(q)
        show(get(url).get("results", []), "cites", wid + " | " + q, full_abs=False)
    else:
        raise SystemExit("unknown mode")


if __name__ == "__main__":
    main()
