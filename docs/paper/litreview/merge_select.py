"""Merge enriched pools, apply final selection, emit selected.json + refs_litreview.bib.

Selection = relevance-filter output of the (Fable) reviewing pass:
- keep all verified targeted + fusion/novelty + ml-methods papers
- add two keyword-sweep finds (Jung2026 KSTAR profile reconstruction, PanoMHD)
- drop unverifiable candidates (never made it past title enrichment)
"""
import json
import re
import sys

sys.path.insert(0, ".")
from s2_enrich import generate_bibtex, normalize_title

POOLS = [
    "enriched_targeted.json",
    "enriched_fusion_novelty.json",
    "enriched_ml_methods.json",
    "enriched_round2.json",
]

EXTRA_FROM_KW = [
    ("kw_kstar_ml.json", "machine learning-based plasma profile reconstruction in kstar",
     "Related Work: C real-time reconstruction",
     "Jung et al. 2026 — EPED-parameterized rapid profile reconstruction on KSTAR (adjacent, same device)"),
    ("kw_mirnov2.json", "panomhd multimodal modelling of plasma dynamics towards tokamak control",
     "Related Work: C multimodal signal modelling (concurrent)",
     "PanoMHD 2026 — causal multimodal transformer forecasting magnetic-fluctuation signals (concurrent, forecasting not gap-filling)"),
]

registry = {}
for pool in POOLS:
    data = json.load(open(pool, encoding="utf-8"))
    for p in data["papers"]:
        registry.setdefault(normalize_title(p["title"]), p)

for kw_file, title_frag, section, reason in EXTRA_FROM_KW:
    for r in json.load(open(kw_file, encoding="utf-8")):
        if title_frag in normalize_title(r["title"]) or normalize_title(title_frag)[:30] in normalize_title(r["title"]):
            r["found_in_section"] = section
            r["reason"] = reason
            registry.setdefault(normalize_title(r["title"]), r)
            break

papers = sorted(registry.values(), key=lambda p: (p["found_in_section"], -(p["year"] or 0)))
json.dump(papers, open("selected.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open("refs_litreview.bib", "w", encoding="utf-8").write(generate_bibtex(papers))

print(f"{len(papers)} papers selected")
for p in papers:
    print(f"  {p['citation_key']:<42} {p['year']} | {p['found_in_section'][:50]}")
