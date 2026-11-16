# -*- coding: utf-8 -*-
"""포스터에 적힌 모든 숫자를 원천과 대조한다.

  py docs/presentation/check_poster_numbers.py [포스터.pptx]

토큰 = 소수, 또는 세 자리 이상 정수(부호 유지). 원천: PPCF 원고(main_ppcf.tex), 참고문헌(refs.bib),
docs/paper/file_rules.md, 그림 원자료(fig_missing.json, fig_trace.json). 원천에 그대로 없는 토큰은
DERIVED 에 유도 과정을 적어야 통과한다. 옮겨 적다 생긴 오류를 잡는 검사이며, 숫자가 맞는 문장에
붙었는지는 사람이 읽어 확인한다.
"""
import io
import json
import re
import sys
from pathlib import Path

from pptx import Presentation

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PPTX = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "KSTAR_CES_포스터_초안.pptx"

DERIVED = {
    "2010": "Ko et al., Rev. Sci. Instrum. 81, 10D740 (2010) — refs.bib year field",
    "2024": "Jackson et al., SoftwareX 27, 101869 (2024) — refs.bib year field",
    "2025": "Heo et al., Nucl. Fusion 65, 036003 (2025) — refs.bib year field",
    "2026": "Lee et al., Fusion Eng. Des. 222, 115518 (2026) — refs.bib year field",
}


def norm(s):
    for a_, b_ in ((chr(0x2212), "-"), (chr(0xA0), " "), ("{,}", ","), (chr(92) + ",", " "), ("$", " "),
                   (chr(0x2013), " to "), (chr(0x2026), " to ")):
        s = s.replace(a_, b_)
    return s


def tokens(s):
    out = set()
    for t in re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?", norm(s)):
        t = t.rstrip(",")
        body = t.lstrip("+-")
        if "." in body or len(body.replace(",", "")) >= 3:
            out.add(t)
    return out


def poster_text():
    prs = Presentation(PPTX)
    parts = []

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == 6:
                walk(sh.shapes)
            elif sh.has_text_frame:
                parts.append("\n".join("".join(r.text for r in p.runs) for p in sh.text_frame.paragraphs))
    for slide in prs.slides:
        walk(slide.shapes)
    return "\n".join(parts)


def main():
    src = [(ROOT / "docs/paper/ppcf/main_ppcf.tex").read_text(encoding="utf-8"),
           (ROOT / "docs/paper/ppcf/refs.bib").read_text(encoding="utf-8"),
           (ROOT / "docs/paper/file_rules.md").read_text(encoding="utf-8"),
           # the record in use; its own numbers are checked by ces_prediction/check_records.py
           (ROOT / "PROJECT_KNOWLEDGE.md").read_text(encoding="utf-8")]
    fm = json.loads((ROOT / "docs/paper/ppcf/fig_missing.json").read_text(encoding="utf-8"))
    extra = [f"{fm['n_rows']:,}"]
    for k in ("CES_TI", "CES_VT"):
        extra += ["%.1f" % fm[k][q] for q in ("missing_pct", "held_pct_of_grid", "independent_pct")]
    ft = json.loads((ROOT / "docs/paper/ppcf/fig_trace.json").read_text(encoding="utf-8"))
    for f in ("s31921.csv", "s31923.csv", ft["typical_discharge"]["file"]):
        extra += ["%.1f" % ft[f][k]["rmse_stored"] for k in ("model", "causal_gp", "persistence")]
        extra.append(f[1:6])                                   # the discharge number
    have = tokens("\n".join(src) + "\n" + " ".join(extra))
    have_abs = {t.lstrip("+-") for t in have}
    bad = []
    toks = sorted(tokens(poster_text()))
    for t in toks:
        body = t.lstrip("+-")
        if t in have or t in DERIVED or (t[0] not in "+-" and body in have_abs) or (t[0] == "+" and body in have_abs):
            continue
        bad.append(t)
    print("poster tokens checked: %d, not found in any source: %d" % (len(toks), len(bad)))
    for t in bad:
        print("  MISSING", t)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
