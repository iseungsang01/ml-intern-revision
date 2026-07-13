# -*- coding: utf-8 -*-
"""Bibliography cleanup for refs.bib (critic items 4-6).

- Brace-protect proper nouns in titles (unsrtnat downcases otherwise)
- Fix garbled non-ASCII author names (OpenAlex dropped diacritics)
- Fix missing hyphens / casing in author names
- Deduplicate article-number-style pages (X--X -> X)
- Repair weak venues for well-known ML papers (arXiv -> actual venue)
"""
import re

PATH = r"C:\Users\lss\Documents\GitHub\ml-intern-revision\docs\paper\refs.bib"
text = open(PATH, encoding="utf-8").read()

# 1) pages dedupe: {A--A} or {A - A} -> {A}
text = re.sub(r"pages=\{([^{}]+?)\s*-{1,2}\s*\1\}", r"pages={\1}", text)

# 2) author-name repairs (literal, safe)
name_fixes = {
    "M. Imríek": r"M. Imr{\'i}{\v s}ek",
    "Michal Odloilík": r"Michal Odlo{\v z}il{\'i}k",
    "M. os": r"M. {\v S}os",
    "Baoyue CHAI": "Baoyue Chai",
    "HyeongJun NOh": "HyeongJun Noh",
    "LouisPhilippe Morency": "Louis-Philippe Morency",
    "DitYan Yeung": "Dit-Yan Yeung",
    "TieYan Liu": "Tie-Yan Liu",
    "WaiKin Wong": "Wai-Kin Wong",
    "Wangchun WOO": "Wang-chun Woo",
}
for old, new in name_fixes.items():
    text = text.replace(old, new)

# 3) brace-protect proper nouns inside title={...} lines only
NOUNS = ["KSTAR", "JET", "EAST", "COMPASS", "DIII-D", "W7-X", "NSTX-U", "HL-3",
         "Bayesian", "LSTM", "GS-DeepNet", "PanoMHD", "FusionMAE", "Thomson",
         "Gaussian", "Mirnov", "Grad--Shafranov", "Grad-Shafranov", "Tokamak",
         "EPED", "MHD", "ECE"]

def protect(m):
    title = m.group(1)
    for n in NOUNS:
        # wrap standalone occurrences not already braced
        title = re.sub(rf"(?<!\{{)\b({re.escape(n)})\b(?!\}})", r"{\1}", title)
    return "title={" + title + "}"

text = re.sub(r"title=\{([^{}]+)\}", protect, text)

# 4) venue repairs for canonical ML papers (match on citation key's entry block)
venue_fixes = {
    "Ngiam2011MultimodalDeep": ("booktitle", "Proceedings of the 28th International Conference on Machine Learning (ICML)"),
    "Xiong2020LayerNormalization": ("booktitle", "Proceedings of the 37th International Conference on Machine Learning (ICML)"),
    "Ilse2018AttentionbasedDeep": ("booktitle", "Proceedings of the 35th International Conference on Machine Learning (ICML)"),
    "Shukla2021MultitimeAttention": ("booktitle", "International Conference on Learning Representations (ICLR)"),
    "Rubanova2019LatentOrdinary": ("booktitle", "Advances in Neural Information Processing Systems 32 (NeurIPS)"),
    "Shi2015ConvolutionalLstm": ("booktitle", "Advances in Neural Information Processing Systems 28 (NeurIPS)"),
    "Noh2026PanomhdMultimodal": ("booktitle", "arXiv preprint"),
    "Joung2020DeepNeural": ("journal", "Nuclear Fusion"),
}

def fix_entry(key, field, value, text):
    # find the entry block
    m = re.search(rf"(@\w+\{{{re.escape(key)},.*?\n\}})", text, re.S)
    if not m:
        print(f"  !! entry not found: {key}")
        return text
    block = m.group(1)
    new_block = block
    # replace existing journal/booktitle line whatever it is
    new_block = re.sub(r"\n  (journal|booktitle)=\{[^{}]*\},?", "", new_block, count=1)
    # re-insert after author line
    new_block = re.sub(r"(\n  author=\{[^{}]*\},)", rf"\1\n  {field}={{{value}}},", new_block, count=1)
    if new_block != block:
        text = text.replace(block, new_block)
        print(f"  venue fixed: {key} -> {field}={value[:40]}...")
    return text

for key, (field, value) in venue_fixes.items():
    text = fix_entry(key, field, value, text)

open(PATH, "w", encoding="utf-8").write(text)
print("refs.bib cleaned")
