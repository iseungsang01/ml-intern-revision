# -*- coding: utf-8 -*-
"""포스터에 적힌 모든 숫자를 원천과 대조하고, 쓰지 않기로 한 낱말을 찾는다.

  py docs/presentation/check_poster_numbers.py [포스터.pptx]

토큰 = 소수, 또는 세 자리 이상 정수(부호 유지). 원천은 SOURCES 의 파일들이다(기록, B.54 · B.59 · B.64 · B.65 ·
B.66 판독, PPCF 원고, 참고문헌, 파일 규칙). 원천에 그대로 없는 토큰은 DERIVED 에 유도 과정을 적어야 통과한다.
옮겨 적다 생긴 오류를 잡는 검사이며, 숫자가 맞는 문장에 붙었는지는 사람이 읽어 확인한다.
그림 안의 숫자는 make_figures_poster.py 가 기록 파일에서 읽고 단언한다.
"""
import io
import re
import sys
from pathlib import Path

from pptx import Presentation

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PPTX = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "KSTAR_CES_포스터_초안.pptx"

SOURCES = ["PROJECT_KNOWLEDGE.md",
           "docs/paper/b64_two_stage.md",
           "docs/paper/b65_inclusive.md",
           "docs/paper/b59_inputs.md",
           "docs/paper/b66_regimes.md",
           "docs/paper/b54_pooled_rows.md",
           "docs/paper/ppcf/main_ppcf.tex",
           "docs/paper/ppcf/refs.bib",
           "docs/paper/file_rules.md"]
DERIVED = {}
# 포스터 명세의 금지어 ('첫'은 '첫 단계'로 쓰이므로 여기서 찾지 않는다)
BANNED = ["wins", "beats", "outperform", "novel", "최초", "모수"]


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
    have = tokens("\n".join((ROOT / f).read_text(encoding="utf-8") for f in SOURCES))
    have_abs = {t.lstrip("+-") for t in have}
    text = poster_text()
    bad = []
    toks = sorted(tokens(text))
    for t in toks:
        body = t.lstrip("+-")
        if t in have or t in DERIVED or (t[0] != "-" and body in have_abs):
            continue
        bad.append(t)
    print("poster tokens checked: %d, not found in any source: %d" % (len(toks), len(bad)))
    for t in bad:
        print("  MISSING", t)
    words = [w for w in BANNED if w in text]
    print("banned words found: %d" % len(words))
    for w in words:
        print("  BANNED", w)
    sys.exit(1 if bad or words else 0)


if __name__ == "__main__":
    main()
