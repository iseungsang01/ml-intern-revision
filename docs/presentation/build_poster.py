# -*- coding: utf-8 -*-
"""KSTAR CES nowcasting 포스터, 영어판과 한국어판 (세로 45 x 60 cm; 200 % 로 인쇄하면 90 x 120 cm).

  py docs/presentation/make_figures_poster.py               # 그림 먼저 (figures/poster_20261119/, 두 언어)
  py docs/presentation/build_poster.py --lang en            # -> KSTAR_CES_poster_en.pptx + .png
  py docs/presentation/build_poster.py --lang ko            # -> KSTAR_CES_포스터_초안.pptx + .png
  py docs/presentation/check_poster_numbers.py <pptx>       # 포스터의 모든 숫자를 원천과 대조
  py docs/presentation/build_poster.py --lang en --anon --out X.pptx   # 저자 줄을 익명으로 바꾼 사본

* 틀: 남색 제목 띠, 두 단(1–3 | 4–7)과 가운데 가는 세로선. 구역은 상자 없이 번호 + 굵은 남색 제목 +
  단 너비의 선으로 나누고, 요점은 왼쪽에 굵은 남색 막대가 있는 연한 상자, 표는 booktabs(위아래 굵은 선,
  머리 아래 가는 선), 그림·표 설명은 회색 기울임.
* 결과 수치는 표에만 둔다(문장에는 없음). 강조도 표 안에서: 두 단계 estimator 열은 굵게, 강조할 행은
  연한 청회색 바탕(표 1은 인과 GP · offline GP 행, 표 2 · 3은 구간이 0을 제외하는 행). 자료를 설명하는 수
  (방전 수, 행 수, R, 채널 수, 10 ms)는 문장에 남는다.
* 글은 전부 TEXT[언어] 에 있고 배치는 두 언어가 같다. 수치 칸은 VALUES 하나를 두 언어가 같이 쓴다.
* 글상자 높이는 Pretendard 실제 글자 폭으로 줄 수를 세어 정한다(줄 간격은 pt 고정; PowerPoint 의 폭은
  PIL 계측의 0.996 배로 시험해 확인). 본문 크기는 두 단 모두에 MIN_FREE 이상의 여백이 남는 가장 큰 값을
  SIZE_STEPS 에서 고르고, 남는 높이는 간격에 나눠 두 단의 아래 끝을 맞춘다.
* 도형은 전부 PowerPoint 기본 도형과 글상자이다 — 받아서 그대로 고칠 수 있다.
* PNG 미리보기는 PowerPoint COM(Slides(1).Export)으로 내보낸다. 열린 문서가 없을 때만 Quit 한다.
"""
import argparse
import io
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "poster_20261119"      # make_figures_poster.py 가 포스터 크기로 그린 그림

# ================================================================== 글 (언어별)
REFS_LINE = ("[1] W. H. Ko et al., Rev. Sci. Instrum. 81, 10D740 (2010).   [2] J. K. Lee et al., Fusion Eng. Des. 222, "
             "115518 (2026).   [3] C. Heo et al., Nucl. Fusion 65, 036003 (2025).   [4] J. H. Friedman, Ann. Stat. 29, "
             "1189 (2001).")
# 결과 수치: 두 언어가 같이 쓴다. '**굵게**', '{작은 회색}' 표기.
V_T1 = [("+0.378 {[+0.358, +0.397]}", "**+0.436** {[+0.418, +0.452]}"),
        ("+0.205 {[+0.173, +0.234]}", "**+0.279** {[+0.251, +0.304]}"),
        ("+0.092 {[+0.068, +0.114]}", "**+0.176** {[+0.155, +0.196]}"),
        ("−0.033 {[−0.062, −0.005]}", "**+0.063** {[+0.038, +0.086]}"),
        ("+0.053 {[+0.028, +0.076]}", "**+0.097** {[+0.077, +0.116]}"),
        ("184.2", "**175.4**"),
        ("45.8", "**41.2**"),
        ("—", "**0.909** {[0.889, 0.931]}")]
V_T2 = ["+0.053 {[+0.028, +0.076]}", "−0.006 {[−0.025, +0.016]}", "+0.044 {[+0.020, +0.066]}",
        "−0.011 {[−0.092, +0.022]}", "+0.060 {[+0.023, +0.085]}"]
V_T3 = ["+0.151 {[+0.093, +0.215]}", "+0.050 {[+0.027, +0.075]}", "+0.007 {[−0.023, +0.031]}",
        "+0.042 {[+0.009, +0.074]}", "+0.092 {[+0.040, +0.137]}"]
V_T4 = ["**0.810** {[0.753, 0.866]}", "**135.7** {[126.4, 145.7]} eV"]
V_CALC = ["0.49 ms", "0.64 ms", "1.257 ms"]

TEXT = {
    "en": {
        "out": "KSTAR_CES_poster_en.pptx", "lang_tag": "en-US",
        "title": "Nowcasting the ion temperature at the KSTAR pedestal top every 10 ms",
        "subtitle": "A two-stage estimator that reads the co-located beam emission (BES) with the CES history",
        "author": "**Seungsang Lee** · Department of Nuclear Engineering, Seoul National University",
        "anon_author": "Presenter · Affiliation", "doc_author": "Seungsang Lee",
        "data_line": "Data: KSTAR 2022 campaign, 639 discharges (provided by the KSTAR team)",
        "sections": ("Introduction", "Data and scoring", "Two-stage estimator", "Results",
                     "Where the information is (stage-1 readouts)", "Conclusion", "Limitations and future work"),
        "s1_paras": [
            "**Problem:** CES measures the ion temperature T_i once every 10 ms. At the pedestal-top chord the signal "
            "is weak. Part of the T_i values are missing, and not at random: they are missing more often when the BES "
            "level is low (Fig. 1).",
            "**Constraint:** monitoring and control need the present value. Interpolation waits for the next value, so "
            "it cannot be used.",
            "**Idea:** nine BES and four ECEI channels view the same location without gaps. They do not measure T_i, "
            "but they carry the change of the plasma in the present frame."],
        "question": "Question: without the next CES value, can the present T_i be estimated more accurately than from "
                    "the CES history alone?",
        "fig1_alt": "Share of frames without T_i by BES-level quintile: lowest 18.0 %, all 7.7 %, highest 3.4 %",
        "fig1_cap": "Fig. 1. Share of frames without T_i by BES-level quintile (245,925 rows of the 10 ms grid).",
        "s2_data": "**Data:** KSTAR 2022 campaign, 639 discharges, 245,925 rows on a 10 ms grid. Chord at R = 2.198 m, "
                   "the pedestal top chosen by the data provider. Each row holds 9 BES, 4 ECEI and 2 Mirnov channels at "
                   "the same location, and the CES value.",
        "frames": ["CES frame", "previous frame\n(used by hold)", "**target frame**\n(exposure starts\nat t)",
                   "blip: no frame\n(missing row)", "CES frame"],
        "row_ces": "CES", "row_beam": "beam", "beam_off": "off (about 10 ms)",
        "snapshot": "fast channels (BES·ECEI·Mirnov): one sample at time t",
        "fig2_cap": "Fig. 2. What a row holds. Each fast channel is one raw sample at time t; the CES value is the frame "
                    "whose exposure starts at t. A frame during a beam blip (beam off for about 10 ms) has no value.",
        "base_head": ("reference", "what it uses"),
        "base_rows": [("**hold**", "the last value"),
                      ("**causal GP**", "a Gaussian process fitted to the CES history only"),
                      ("**offline GP**", "also the next value; a reference that cannot run in real time"),
                      ("**history-only control**",
                       "the same network without the fast channels; the difference is the fast-channel increment")],
        "skill_note": "skill = 1 − MSE(estimator) / MSE(reference). Above 0: lower mean squared error than the reference.",
        "s2_read": "**Readout:** the discharges are split into five groups, each tested in turn, and the split is repeated "
                   "and pooled. Intervals are 95 %, by resampling discharges. Values above 3 keV (mostly fit failures) "
                   "are removed. Statements and decision rules were registered before each run.",
        "pipe_fast": ["**Fast channels (15)**", "BES 9 · ECEI 4 · Mirnov 2", "at row t"],
        "pipe_ces": ["**CES history (7)**", "last value · age · observed flag", "before row t"],
        "pipe_st1": ["**Stage 1: LSTM**", "2 layers · 160 units", "every grid row"],
        "pipe_st2": ["**Stage 2:**", "**gradient boosting**", "estimates the", "stage-1 error"],
        "pipe_fit": "fitted on out-of-fold predictions of training discharges",
        "pipe_w": (5.9, 3.7, 3.9, 2.05, 1.7),          # 입력, 첫 단계, 둘째 단계, 출력 둘 (cm)
        "fig3_cap": "Fig. 3. The two-stage estimator. The history-only control has the same structure without the fast "
                    "channels; its second stage reads the CES history only.",
        "s3_items": [
            "**Every row in order:** frames without a value stay on the grid;\nthe loss counts only rows with a value.",
            "**Residual correction:** 15 inputs (8 from BES and ECEI windows, 7 from the CES history and the stage-1 "
            "output; no Mirnov), fitted only on out-of-fold predictions of the training discharges.",
            "**Cost:** one row finishes within a 10 ms frame on one CPU thread (table)."],
        "calc_head": ("per-row time, one CPU thread", "median"),
        "calc_rows": ["network", "causal GP", "stage 2"],
        "t1_cap": "Table 1. Skill over each reference (pooled, three re-partitions, 15 folds). Above 0: lower error.",
        "t1_head": ("reference", "network alone", "**two-stage estimator**"),
        "t1_rows": ["hold", "PCHIP interpolation (uses the next value)", "causal GP", "offline GP (uses the next value)",
                    "same-stage control (increment)", "RMSE (eV)", "median absolute error (eV)",
                    "median error ÷ that of hold"],
        "t1_c0": 8.9,
        "t1_note": "Causal GP: RMSE 193.3 eV, median 44.1 eV. Hold: median 45.4 eV.",
        "s4_bullets": [
            "**Over the causal GP and over the offline GP, which uses the next value:**\nboth intervals of the "
            "two-stage estimator lie above 0.",
            "**The second stage** raises the skill over every reference and moves the interval\nover the offline GP "
            "from below 0 to above 0.",
            "**Median error:** below that of hold.",
            "**Fast-channel increment:** larger after the second stage."],
        "t2_cap": "Table 2. Network retrained with changed inputs: skill over the history-only control.",
        "t2_head": ("what the network reads", "skill over the control"),
        "t2_rows": ["all fast channels (reference model)", "fast channels one row (10 ms) late", "BES + CES history",
                    "ECEI + CES history", "Mirnov removed"],
        "t3_cap": "Table 3. Where the increment is (skill over the history-only control).",
        "t3_head": ("rows · discharges", "skill over the control"),
        # '⇥' 는 들여쓰기: T3_INDENT 가 문자열이면 그 폭, 숫자이면 그 cm
        "t3_rows": ["rows 1–3 after a beam blip", "between BES bursts", "discharge temperature thirds:\n⇥coldest",
                    "⇥middle", "⇥hottest"],
        "t3_indent": 0.45,
        "s5_bullets": [
            "**BES of the present frame:** fed one row late, the fast channels give no increment; they are not a 10 ms "
            "precursor.",
            "**Mirnov is not needed;** ECEI alone gives no increment.",
            "**Largest right after a beam blip;** none in the coldest third."],
        "s6_take": "Changes of the co-located BES in the present frame carry information on the pedestal-top CES T_i "
                   "beyond the CES history. The two-stage estimator has a lower error than the causal GP, which uses the "
                   "CES history only, and than the offline GP, which also uses the next value (cut population, Table 1).",
        "t4_cap": "Table 4. Checks.",
        "t4_head": ("check", "value"),
        "t4_rows": ["share of the fast-channel correction that remains in the next, independently fitted frame",
                    "floor of the frame-to-frame scatter of the reported value (RMS)"],
        "s6_checks": ["**Not an artefact of the frame's fit** (Table 4)",
                      "**Most of the remaining error is the scatter floor of the reported value** (Table 4)",
                      "**Computed within a 10 ms frame;** statements registered before each run"],
        "s7_items": [
            ("**Kept in, the values above 3 keV (fit failures) remove the fast-channel increment; the margin over the "
             "causal GP remains (Fig. 4).**",
             "→ test four hypotheses: a few large values dominate the loss · a wider target normalisation · fit failures "
             "coupled to the fast channels · large values in the causal GP's history"),
            ("**No untouched hold-out discharges**",
             "→ a temporal confirmation on the last 97 discharges of the campaign (running)"),
            ("**One chord, one campaign, one device.** The quantity measured is the temperature of the carbon impurity, "
             "and real-time use of BES or CES on KSTAR is not reported", "→ real-time use is a prospect")],
        "fig4_alt": "Skill of the network on the cut and inclusive populations: over the causal GP +0.092 and +0.109, "
                    "over the history-only control +0.053 and -0.002",
        "fig4_cap": "Fig. 4. Skill of the network on the cut\nand on the inclusive population.",
        "refs": ["**References**  " + REFS_LINE,
                 "**Acknowledgement**  The KSTAR data were provided by the KSTAR team. There was no dedicated funding."],
        "doc_title": "Nowcasting the ion temperature at the KSTAR pedestal top every 10 ms (poster)",
    },
    "ko": {
        "out": "KSTAR_CES_포스터_초안.pptx", "lang_tag": "ko-KR",
        "title": "KSTAR pedestal top 이온 온도의 10 ms nowcasting",
        "subtitle": "같은 위치의 빔 방출 신호(BES)를 CES 이력과 함께 읽는 두 단계 estimator",
        "author": "**이승상**  ·  서울대학교 원자핵공학과",
        "anon_author": "발표자 · 소속", "doc_author": "이승상",
        "data_line": "자료: KSTAR 2022 캠페인 639개 방전 (KSTAR team 제공)",
        "sections": ("서론", "자료와 채점", "두 단계 estimator", "결과", "정보는 어디에 있나 (첫 단계 판독)", "결론",
                     "한계와 향후 연구"),
        "s1_paras": [
            "**문제:** CES는 이온 온도 T_i를 10 ms마다 한 값 잰다. pedestal top 측정선에서는 신호가 약해 T_i 값의 일부가 "
            "비어 있고, 빠짐은 무작위가 아니다. BES 수준이 낮을수록 더 자주 빈다(그림 1).",
            "**제약:** 감시·제어는 지금 값을 써야 하므로, 다음 값을 기다리는 보간은 쓸 수 없다.",
            "**착안:** 같은 위치를 BES 9채널과 ECEI 4채널이 빈틈 없이 본다. T_i를 재지는 않지만, 지금 프레임의 플라스마 "
            "변화를 담는다."],
        "question": "질문: 다음 CES 값 없이, CES 이력만 쓸 때보다\n지금 프레임의 T_i를 더 정확히 추정할 수 있는가?",
        "fig1_alt": "BES 수준의 5분위별 T_i 결측 비율: 낮은 5분의 1 18.0 %, 전체 7.7 %, 높은 5분의 1 3.4 %",
        "fig1_cap": "그림 1. BES 수준의 5분위별 T_i 결측 비율 (10 ms 격자 245,925행).",
        "s2_data": "**자료:** KSTAR 2022 캠페인 639개 방전, 10 ms 격자 245,925행. 측정선 R = 2.198 m(자료 제공자가 고른 "
                   "pedestal top). 한 행에는 같은 위치의 BES 9, ECEI 4, Mirnov 2채널 값과 CES 값이 있다.",
        "frames": ["CES 프레임", "직전 프레임\n(hold가 쓰는 값)", "**추정 대상 프레임**\n(t에서 노출 시작)",
                   "blip: 프레임 없음\n(빠진 행)", "CES 프레임"],
        "row_ces": "CES", "row_beam": "빔", "beam_off": "꺼짐 (약 10 ms)",
        "snapshot": "빠른 진단(BES·ECEI·Mirnov) 표본: 시각 t의 순간값",
        "fig2_cap": "그림 2. 한 행이 담는 것. 빠른 진단은 시각 t의 원시 표본 하나이고, CES 값은 t에서 노출이 시작되는 "
                    "프레임이다. 빔이 약 10 ms 꺼지는 blip의 프레임에는 값이 없다.",
        "base_head": ("기준", "내용"),
        "base_rows": [("**hold**", "직전 값 유지"),
                      ("**인과 GP**", "CES 이력만 맞춘 Gaussian process"),
                      ("**offline GP**", "다음 값까지 사용, 실시간으로는 쓸 수 없는 참고 기준"),
                      ("**비교군**", "CES 이력만 읽는 같은 신경망(빠른 채널만 뺌), 그 차이가 빠른 채널의 증분")],
        "skill_note": "skill = 1 − MSE(estimator) / MSE(기준). 0보다 크면 estimator의 평균 제곱 오차가 작다.",
        "s2_read": "**판독:** 방전을 다섯 묶음으로 나눠 번갈아 시험하고 그 나눔을 바꿔 합친다(pooled). 구간은 방전 단위 "
                   "bootstrap 95 %. 3 keV 넘는 값(대부분 fit 실패)은 뺀다. 진술과 판정 규칙은 실행 전에 등록했다.",
        "pipe_fast": ["**빠른 채널 15**", "BES 9 · ECEI 4 · Mirnov 2", "행 t의 값"],
        "pipe_ces": ["**CES 이력 7**", "마지막 값 · 경과 시간 · 관측 여부", "행 t 이전"],
        "pipe_st1": ["**첫 단계: LSTM**", "2층 · 160 유닛", "격자의 모든 행"],
        "pipe_st2": ["**둘째 단계:**", "**gradient boosting**", "첫 단계의 오차 추정"],
        "pipe_fit": "학습 방전의 fold 밖 예측으로 맞춤",
        "pipe_w": (5.95, 3.55, 4.15, 2.05, 1.7),
        "fig3_cap": "그림 3. 두 단계 estimator. 비교군은 빠른 채널만 뺀 같은 구조이고, 그 둘째 단계는 CES 이력만 읽는다.",
        "s3_items": [
            "**모든 행을 순서대로:** 값이 빈 프레임도 격자에 남기고, 값이 있는 행에서만 손실을 센다.",
            "**잔차 보정:** 입력 15개(BES·ECEI 창 8, CES 이력과 첫 단계 출력 7, Mirnov 없음). 학습 방전에서 만든 fold 밖 "
            "예측으로만 맞춘다.",
            "**계산:** CPU 한 스레드에서 한 행이 10 ms 프레임 안에 끝난다(표)."],
        "calc_head": ("한 행 계산 (CPU 한 스레드)", "중앙값"),
        "calc_rows": ["신경망", "인과 GP", "둘째 단계"],
        "t1_cap": "표 1. 기준 대비 skill (pooled, 세 번의 나눔 15 fold). 0보다 크면 오차가 더 낮다.",
        "t1_head": ("기준", "신경망 혼자", "**두 단계 estimator**"),
        "t1_rows": ["hold", "PCHIP 보간 (다음 값 사용)", "인과 GP", "offline GP (다음 값 사용)", "같은 단계의 비교군 (증분)",
                    "RMSE (eV)", "중앙값 절대오차 (eV)", "중앙값 오차 ÷ hold의 중앙값 오차"],
        "t1_c0": 7.9,
        "t1_note": "인과 GP: RMSE 193.3 eV, 중앙값 44.1 eV. hold: 중앙값 45.4 eV.",
        "s4_bullets": [
            "**인과 GP 대비, 다음 값을 쓰는 offline GP 대비:** 두 단계 estimator의 구간이 모두 0 위에 있다",
            "**둘째 단계:** 모든 기준 대비 skill을 높이고, offline GP 대비 구간을 0 아래에서 0 위로 올린다",
            "**중앙값 오차:** hold보다 작다",
            "**빠른 채널의 증분:** 둘째 단계 뒤에 커진다"],
        "t2_cap": "표 2. 입력을 바꿔 다시 학습한 신경망의 비교군 대비 skill.",
        "t2_head": ("신경망이 읽는 것", "비교군 대비"),
        "t2_rows": ["빠른 채널 전부 (기준 모델)", "빠른 채널을 한 행(10 ms) 늦게", "BES + CES 이력", "ECEI + CES 이력",
                    "Mirnov만 뺌"],
        "t3_cap": "표 3. 증분이 있는 곳 (비교군 대비).",
        "t3_head": ("행 · 방전", "비교군 대비"),
        "t3_rows": ["빔 blip 뒤 1–3행", "BES 버스트 사이 구간", "방전 온도 삼등분: 차가운", "⇥중간", "⇥뜨거운"],
        "t3_indent": "방전 온도 삼등분: ",
        "s5_bullets": [
            "**지금 프레임의 BES:** 한 행 늦게 넣으면 증분이 없다. 10 ms 앞선 전조가 아니다.",
            "**Mirnov는 필요 없다.** ECEI만으로는 증분이 없다.",
            "**빔 blip 직후에 가장 크고,** 가장 차가운 3분의 1에는 없다."],
        "s6_take": "같은 위치 BES의 지금 프레임 변화는 pedestal top CES T_i에 대해 CES 이력 너머의 정보를 준다. 두 단계 "
                   "estimator는 CES 이력만 쓰는 인과 GP보다, 다음 값까지 쓰는 offline GP보다도 오차가 낮다(3 keV 컷 "
                   "모집단, 표 1).",
        "t4_cap": "표 4. 확인",
        "t4_head": ("확인", "값"),
        "t4_rows": ["빠른 채널이 만든 보정 가운데 따로 fit되는 다음 프레임에도 남는 몫", "보고값의 프레임별 산포 하한 (RMS)"],
        "s6_checks": ["**fit의 산물이 아니다** (표 4)",
                      "**남은 오차의 대부분은 보고값 산포의 바닥이다** (표 4)",
                      "**10 ms 프레임 안의 계산**, 진술은 실행 전에 등록"],
        "s7_items": [
            ("**3 keV 넘는 값(fit 실패)을 남기면 빠른 채널의 증분이 사라지고, 인과 GP 대비 여유는 남는다(그림 4).**",
             "→ 가설 넷을 가린다: 소수의 큰 값이 손실을 지배 · 정규화 폭 · fit 실패와 빠른 채널의 결합 · 인과 GP 이력"),
            ("**손대지 않은 검증 방전이 없다**", "→ 캠페인 마지막 97개 방전으로 시간순 확증 (진행\u00a0중)"),
            ("**측정선 하나, 캠페인 하나, 장치 하나.** 재는 것은 탄소 불순물 온도이고, KSTAR에서 BES·CES의 실시간 "
             "사용은 보고되지 않았다", "→ 실시간 적용은 전망")],
        "fig4_alt": "3 keV 컷 모집단과 포함 모집단의 신경망 skill: 인과 GP 대비 +0.092와 +0.109, 비교군 대비 +0.053과 "
                    "-0.002",
        "fig4_cap": "그림 4. 3 keV 컷 모집단과 포함 모집단의 skill (신경망).",
        "refs": ["**참고문헌**  " + REFS_LINE, "**사사**  KSTAR 자료는 KSTAR team이 제공하였다. 별도의 연구비 지원은 없었다."],
        "doc_title": "KSTAR pedestal top 이온 온도의 10 ms nowcasting (포스터)",
    },
}

# ================================================================== 인자
ap = argparse.ArgumentParser()
ap.add_argument("--lang", choices=("en", "ko"), default="en")
ap.add_argument("--anon", action="store_true", help="저자 줄을 익명으로 (웹 미리보기용, --out 필요)")
ap.add_argument("--out", type=Path, help="pptx 경로 (기본: docs/presentation/ 의 언어별 이름)")
ap.add_argument("--png", type=Path, help="PNG 경로 (기본: pptx 와 같은 이름)")
ap.add_argument("--png-px", type=int, default=1800, help="PNG 가로 픽셀")
ap.add_argument("--no-png", action="store_true")
args = ap.parse_args()
if args.anon and args.out is None:
    sys.exit("--anon needs --out (the anonymous copy never overwrites the poster)")
LANG = args.lang
T = TEXT[LANG]
OUT = args.out or HERE / T["out"]
PNG = args.png or OUT.with_suffix(".png")
AUTHOR = T["anon_author"] if args.anon else T["author"]

W_CM, H_CM = 45.0, 60.0
FONT = "Pretendard"


def rgb(h):
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


NAVY, INK, BODY, MUTED = rgb("1B2F5E"), rgb("15181E"), rgb("23272F"), rgb("5E6878")
TINT, RULE, WHITE, SUBTLE = rgb("E9EDF4"), rgb("B9C1CE"), rgb("FFFFFF"), rgb("C9D3E6")
BLUE, TINT_B, PANEL = rgb("2A78D6"), rgb("E3EDFA"), rgb("F1F3F6")
FRAME, PREV, BEAM = rgb("E6E9EE"), rgb("D3DAE5"), rgb("9DB7D8")

# ------------------------------------------------------------------ 글자 폭 계측 (Pretendard)
_uf = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"


def _load(name):
    p = _uf / name
    return ImageFont.truetype(str(p if p.exists() else Path(r"C:\Windows\Fonts\malgun.ttf")), 200)


F_REG, F_BOLD = _load("Pretendard-Regular.otf"), _load("Pretendard-Bold.otf")
PT2CM = 2.54 / 72.0
NB = "\u00a0"
SMALL_F = 0.80          # {작은 회색} 글자의 크기 비 (10 pt 아래로는 내리지 않는다)
FIT = 0.995             # 한 줄에 채우는 폭의 비. PowerPoint 의 실제 폭은 PIL 계측의 0.996 배(시험으로 확인)
FRAGILE = []            # 줄바꿈이 폭 끝에 걸린 글(PowerPoint 와 갈릴 수 있음)
TOK = re.compile(r"(\*\*|\{|\}|T\u0302?_i|\^\(1\))")


def small_pt(pt):
    return max(10.0, pt * SMALL_F)


def glue(s):
    """구간 [a, b], 범위 a … b, 숫자와 단위를 한 줄에 묶는다(줄바꿈 없는 공백)."""
    s = re.sub(r"\[[^\]]*\]", lambda m: m.group(0).replace(" ", NB), s)
    s = s.replace(" … ", NB + "…" + NB)
    s = re.sub(r"(?<=\d) (?=%)", NB, s)
    return re.sub(r"(?<=\d) (?=(?:ms|eV|keV|kHz|MHz|µs|m)(?![A-Za-z]))", NB, s)


def parse(s, bold=False):
    """'**굵게**', '{작은 회색}', 'T_i'(아래 첨자), '^(1)'(위 첨자) 표기를 (글자, 굵기, 작음, 첨자) 조각으로."""
    out, b, sm = [], bold, False
    for part in TOK.split(glue(s)):
        if not part:
            continue
        if part == "**":
            b = not b
        elif part == "{":
            sm = True
        elif part == "}":
            sm = False
        elif part.endswith("_i"):
            out += [(part[:-2], b, sm, 0), ("i", b, sm, -1)]
        elif part == "^(1)":
            out.append(("(1)", b, sm, 1))
        else:
            out.append((part, b, sm, 0))
    return out


def _w(s, pt, bold):
    return (F_BOLD if bold else F_REG).getlength(s) / 200.0 * pt * PT2CM


def wrap(s, pt, width, bold=False):
    """낱말 단위 줄바꿈(한글 낱말을 자르지 않음; '\\n' 은 강제 줄바꿈). 줄마다 차지한 폭(cm)을 돌려준다."""
    words, cur, forced = [], 0.0, False
    for txt, b, sm, sh in parse(s, bold):
        size = (small_pt(pt) if sm else pt) * (0.66 if sh else 1.0)
        for atom in re.split(r"( |\n)", txt):
            if atom == "":
                continue
            if atom in (" ", "\n"):
                words.append((cur, forced))
                cur, forced = 0.0, atom == "\n"
            else:
                cur += _w(atom, size, b)
    words.append((cur, forced))
    space = _w(" ", pt, bold)
    lines, x = [], 0.0
    for w, brk in words:
        if brk:
            lines.append(x)
            x = w
        elif x > 0 and x + space + w > width * FIT:
            if x + space + w <= width / 0.99:
                FRAGILE.append(s)
            lines.append(x)
            x = w
        else:
            x = x + (space if x > 0 else 0) + w
    lines.append(x)
    return lines


def text_h(paras, pt, width, pitch, after=0.0, bold=False):
    paras = [paras] if isinstance(paras, str) else paras
    return (sum(len(wrap(s, pt, width, bold)) for s in paras) * pitch + after * (len(paras) - 1)) * PT2CM


def text_w(s, pt, bold=False):
    return max(wrap(s, pt, 1e9, bold))


def nbsp(cm, pt):
    """폭 cm 에 가까운 줄바꿈 없는 공백 줄."""
    return NB * max(1, round(cm / _w(NB, pt, False)))


# ------------------------------------------------------------------ 기본 도형 (DRY 이면 재기만 한다)
prs = Presentation()
prs.slide_width, prs.slide_height = Cm(W_CM), Cm(H_CM)
slide = prs.slides.add_slide(prs.slide_layouts[6])
DRY = False
WARN = []


def _font(run, size, color, bold=False, italic=False, shift=0):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = FONT
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", T["lang_tag"])  # 한글 낱말을 자르지 않는 줄바꿈은 lang = ko-KR 과 latinLnBrk = 0 이 함께 있어야 듣는다
    rpr.set("altLang", "en-US" if LANG == "ko" else "ko-KR")
    latin = rpr.find(qn("a:latin"))
    for tag in ("a:cs", "a:ea"):
        latin.addnext(rpr.makeelement(qn(tag), {"typeface": FONT}))
    if shift:
        rpr.set("baseline", "-25000" if shift < 0 else "30000")


def _fill_paras(tf, paras, size, color, pitch, after, align, bold, italic=False):
    for i, s in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = Pt(pitch)
        p.space_after = Pt(after if i < len(paras) - 1 else 0)
        p._p.get_or_add_pPr().set("latinLnBrk", "0")
        for txt, b, sm, sh in parse(s, bold):
            for k, piece in enumerate(txt.split("\n")):
                if k:
                    p.add_line_break()
                if piece:
                    r = p.add_run()
                    r.text = piece
                    _font(r, small_pt(size) if sm else size, MUTED if sm else color, b, italic, sh)


def text(x, y, w, paras, size, color=BODY, pitch=None, after=0.0, align=PP_ALIGN.LEFT, bold=False, italic=False,
         name=None, check=True):
    """글상자. 높이는 줄 수로 계산해 돌려준다(cm)."""
    paras = [paras] if isinstance(paras, str) else paras
    pitch = pitch or round(size * 1.4, 1)
    need = text_h(paras, size, w, pitch, after, bold)
    if DRY:
        return need
    for s in paras:
        FRAGILE.clear()
        ln = wrap(s, size, w, bold)
        if FRAGILE:
            WARN.append("break at the box edge (%.1f cm box): %s" % (w, s[:50]))
        if check and len(ln) > 1 and ln[-1] < 0.12 * w:
            WARN.append("short last line (%.1f cm of %.1f): %s" % (ln[-1], w, s[:50]))
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(need))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    _fill_paras(tf, paras, size, color, pitch, after, align, bold, italic)
    return need


def box(x, y, w, h, fill=WHITE, line=None, lw=0.75, radius=0.28, name=None, dash=False,
        shape=MSO_SHAPE.RECTANGLE):
    if DRY:
        return None
    s = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = MSO_LINE.DASH
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def label_in(shape, paras, size, color=INK, bold=False, pitch=None, align=PP_ALIGN.CENTER, pad=0.12):
    """도형 안 글자."""
    if shape is None:
        return
    paras = [paras] if isinstance(paras, str) else paras
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Cm(pad))
    _fill_paras(tf, paras, size, color, pitch or round(size * 1.3, 1), 0, align, bold)


def line(x1, y1, x2, y2, color=INK, w=1.0, head=False, dash=False, name=None):
    if DRY:
        return None
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    st = c._element.find(qn("p:style"))      # 기본 선 스타일(그림자 포함)을 떼고 색과 굵기만 준다
    if st is not None:
        c._element.remove(st)
    c.line.color.rgb = color
    c.line.width = Pt(w)
    if dash:
        c.line.dash_style = MSO_LINE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if name:
        c.name = name
    return c


def badge(x, y, d, n, size, name):
    b = box(x, y, d, d, fill=NAVY, shape=MSO_SHAPE.OVAL, name=name)
    label_in(b, str(n), size, color=WHITE, bold=True, pad=0.0)


def fig_size(name):
    """그림은 포스터 위 실제 크기로 그려져 있다: 픽셀 / dpi 로 폭과 높이(cm)."""
    im = Image.open(FIG / name)
    dpi = im.info["dpi"][0]
    return im.size[0] / dpi * 2.54, im.size[1] / dpi * 2.54


def pic(name, x, y, alt):
    """그림을 제 크기로 놓고 높이를 돌려준다."""
    pw, ph = fig_size(name)
    if DRY:
        return ph
    p = slide.shapes.add_picture(str(FIG / name), Cm(x), Cm(y), Cm(pw), Cm(ph))
    p.name = "fig." + name.replace(".png", "")
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return ph


def measure(fn, *a, **k):
    """그리지 않고 높이만 잰다."""
    global DRY
    old, DRY = DRY, True
    try:
        return fn(*a, **k)
    finally:
        DRY = old


# ------------------------------------------------------------------ 격자와 글자 크기
M = 1.5                       # 바깥 여백
GUT = 1.6                     # 단 사이 (가운데 세로선)
COL_W = (W_CM - 2 * M - GUT) / 2      # 두 단의 기본 폭; 실제 폭은 아래에서 SHIFTS 만큼 오른쪽으로 옮겨 맞춘다
SHIFTS = (0.0, 0.4, 0.8, 1.2)          # 오른쪽 단을 넓히는 폭(cm): 두 단의 여백이 고르게 되는 값을 고른다
FOOT_Y = 57.6
Y1 = 57.15                    # 두 단의 아래 끝 (위 끝은 제목 띠 높이에서)
GB, GS = 0.6, 1.2             # 블록 사이, 구역 사이 (cm, 남는 높이를 더 받는다)
CAP_GAP = 0.2                 # 표 설명과 표, 그림과 그림 설명 사이
SIZE_STEPS = (14.5, 14.0, 13.6, 13.2, 12.8, 12.4, 12.0, 11.6, 11.2, 10.8)
MIN_SLACK = -0.05             # 간격마다 더하는 여백(cm)이 두 단 모두 이 이상인 가장 큰 본문 크기를 고른다
                              # (블록 사이 0.55, 구역 사이 1.1 cm 이상)


def set_sizes(b):
    """본문 크기 b 에서 나머지 글자 크기를 정한다. 본문 · 표 설명 · 표의 작은 글자는 10 pt 이상."""
    global BODY_PT, BODY_PITCH, TAKE_PT, TAKE_PITCH, TAB_PT, TAB_PITCH, SIDE_PT, SIDE_PITCH
    global CAP_PT, CAP_PITCH, DIA_PT, HEAD_PT, PARA_AFTER
    BODY_PT, BODY_PITCH = b, round(b * 1.42, 1)
    TAKE_PT = round(b * 1.12, 1)
    TAKE_PITCH = round(TAKE_PT * 1.36, 1)
    TAB_PT = round(b * 0.94, 1)
    TAB_PITCH = round(TAB_PT * 1.34, 1)
    SIDE_PT = max(10.5, round(b * 0.86, 1))           # 나란히 놓인 표 2 · 3
    SIDE_PITCH = round(SIDE_PT * 1.36, 1)
    CAP_PT = max(10.0, round(b * 0.82, 1))
    CAP_PITCH = round(CAP_PT * 1.35, 1)
    DIA_PT = max(10.0, round(b * 0.8, 1))
    HEAD_PT = round(b * 1.85, 1)
    PARA_AFTER = round(b * 0.55, 1)


def body(x, y, w, paras, **kw):
    kw.setdefault("after", PARA_AFTER)
    return text(x, y, w, paras, BODY_PT, pitch=BODY_PITCH, **kw)


def caption(x, y, w, s, align=PP_ALIGN.CENTER, name=None):
    return text(x, y, w, s, CAP_PT, color=MUTED, pitch=CAP_PITCH, italic=True, align=align, name=name)


def heading(x, y, w, n, title, key):
    nw = HEAD_PT * 0.05
    text(x, y, nw, str(n), HEAD_PT, color=NAVY, bold=True, pitch=HEAD_PT * 1.2, name=f"{key}.num")
    text(x + nw, y, w - nw, title, HEAD_PT, color=NAVY, bold=True, pitch=HEAD_PT * 1.2, name=f"{key}.title")
    ry = y + HEAD_PT * 1.2 * PT2CM + 0.2
    line(x, ry, x + w, ry, color=NAVY, w=1.75, name=f"{key}.rule")
    return ry - y + 0.48


def takeaway(x, y, w, s, key):
    bar, padx, pady = 0.24, 0.55, 0.42
    tw = w - bar - 2 * padx
    th = text_h(s, TAKE_PT, tw, TAKE_PITCH, bold=True)
    h = th + 2 * pady
    box(x, y, w, h, fill=TINT, name=f"{key}.box")
    box(x, y, bar, h, fill=NAVY, name=f"{key}.bar")
    text(x + bar + padx, y + pady, tw, s, TAKE_PT, color=NAVY, pitch=TAKE_PITCH, bold=True, name=f"{key}.text")
    return h


def items(x, y, w, paras, glyph, key, gap=0.22, color=NAVY, indent=None):
    """글머리 목록: 기호 + 들여 쓴 문단. 높이를 돌려준다."""
    indent = indent or BODY_PT * 0.042
    yy = y
    for i, s in enumerate(paras):
        text(x, yy, indent, glyph, BODY_PT, color=color, pitch=BODY_PITCH, bold=True, name=f"{key}.g{i}", check=False)
        yy += text(x + indent, yy, w - indent, s, BODY_PT, pitch=BODY_PITCH, name=f"{key}.t{i}")
        if i < len(paras) - 1:
            yy += gap
    return yy - y


def excludes0(cell):
    """칸의 95 % 구간 [a, b]가 0을 제외하는가."""
    m = re.search(r"\[([^,\]]+),\s*([^\]]+)\]", cell.replace(NB, " "))
    lo, hi = (float(v.replace("−", "-")) for v in m.groups())
    return lo > 0 or hi < 0


def booktabs(x, y, cols, header, rows, key, aligns=None, mid_rules=(), fills=(), pt=None, pitch=None, padx=0.16,
             pady=0.11):
    """위아래 굵은 선, 머리 아래 가는 선. rows 의 칸은 '**굵게**', '{작은 회색}' 표기를 쓴다.
    fills 의 행은 연한 청회색 바탕으로 강조한다."""
    pt, pitch = pt or TAB_PT, pitch or TAB_PITCH
    aligns = aligns or [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * (len(cols) - 1)
    xs = [x + sum(cols[:j]) for j in range(len(cols))]
    W = sum(cols)
    yy = y
    line(x, yy, x + W, yy, w=1.6, name=f"{key}.top")
    yy += 0.12

    def row(cells, tag, fill=False):
        nonlocal yy
        hs = [text_h(c, pt, cols[j] - 2 * padx, pitch) for j, c in enumerate(cells)]
        rh = max(hs)
        if fill:
            box(x, yy, W, rh + 2 * pady, fill=TINT, name=f"{key}.{tag}.fill")
        for j, c in enumerate(cells):
            text(xs[j] + padx, yy + pady + (rh - hs[j]) / 2, cols[j] - 2 * padx, c, pt, color=INK, pitch=pitch,
                 align=aligns[j], name=f"{key}.{tag}c{j}", check=False)
        yy += rh + 2 * pady

    row(header, "h")
    yy += 0.02
    line(x, yy, x + W, yy, w=0.75, name=f"{key}.mid")
    yy += 0.08
    for i, cells in enumerate(rows):
        if i in mid_rules:
            yy += 0.04
            line(x, yy, x + W, yy, w=0.75, name=f"{key}.mid{i}")
            yy += 0.08
        row(cells, f"r{i}", fill=i in fills)
    yy += 0.04
    line(x, yy, x + W, yy, w=1.6, name=f"{key}.bottom")
    return yy - y


def col_content(x, w, sections):
    """한 단의 내용 높이(간격 제외)와 간격의 수."""
    hh = [measure(heading, x, 0, w, n, t, k) for n, t, k, _ in sections]
    hs = [[measure(blk, x, 0, w) for blk in blks] for _, _, _, blks in sections]
    return sum(hh) + sum(map(sum, hs)), sum(len(b) - 1 for b in hs), len(sections) - 1


def draw_column(x, w, sections, y0, y1, label):
    """남는 높이를 간격에 나눠(구역 사이는 블록 사이의 두 배) 그린다."""
    content, nb, ns = col_content(x, w, sections)
    free = (y1 - y0) - content - nb * GB - ns * GS
    share = free / (nb + 2 * ns) if (nb + 2 * ns) else 0.0
    print("%s column: content %.2f cm, free %.2f cm -> +%.2f cm per block gap, +%.2f per section gap"
          % (label, content, free, share, 2 * share))
    y = y0
    for i, (n, t, k, blks) in enumerate(sections):
        y += heading(x, y, w, n, t, k)
        for j, blk in enumerate(blks):
            y += blk(x, y, w)
            if j < len(blks) - 1:
                y += GB + share
        if i < ns:
            y += GS + 2 * share
    return free


# ================================================================== 제목 띠
def title_band():
    """제목이 한 줄에 들지 않으면 두 줄로 나누고 띠를 늘린다. 띠의 아래 끝을 돌려준다."""
    tw = W_CM - 2 * M - 1.0
    tpt = next((p for p in (46.0, 44.0, 42.0, 40.0, 38.0, 36.0, 35.0, 34.0) if len(wrap(T["title"], p, tw, True)) == 1),
               42.0)
    t_h = len(wrap(T["title"], tpt, tw, True)) * tpt * 1.15 * PT2CM
    sub_h = text_h(T["subtitle"], 21, tw, 27)
    # 자료 줄은 저자 줄의 오른쪽에 둔다. 저자 줄과 겹치면 그 아래 줄의 오른쪽으로 내린다.
    dw = text_w(T["data_line"], 11) + 0.3
    own_row = W_CM / 2 + text_w(AUTHOR, 17) / 2 + 0.6 > W_CM - M - dw
    band_h = 1.0 + t_h + 0.35 + sub_h + 0.35 + 22 * PT2CM + (0.55 if own_row else 0.0) + 0.75
    box(0, 0, W_CM, band_h, fill=NAVY, name="title.band")
    y = 1.0
    y += text(M + 0.5, y, tw, T["title"], tpt, color=WHITE, bold=True, pitch=tpt * 1.15, align=PP_ALIGN.CENTER,
              name="title") + 0.35
    y += text(M + 0.5, y, tw, T["subtitle"], 21, color=SUBTLE, pitch=27, align=PP_ALIGN.CENTER, name="subtitle") + 0.35
    text(M, y, W_CM - 2 * M, AUTHOR, 17, color=WHITE, pitch=22, align=PP_ALIGN.CENTER, name="author")
    text(W_CM - M - dw, band_h - 0.95, dw, T["data_line"], 11, color=SUBTLE, pitch=14, align=PP_ALIGN.RIGHT,
         name="data.line")
    return band_h


# ================================================================== 왼쪽 단: 1 서론 · 2 자료와 채점 · 3 두 단계 estimator
FIG1 = f"poster_missing_{LANG}.png"
FIG4 = f"poster_inclusive_{LANG}.png"


def s1_intro(x, y, w):
    fw = fig_size(FIG1)[0]
    tw = w - fw - 0.6
    th = body(x, y, tw, T["s1_paras"], name="s1.text")
    fh = pic(FIG1, x + tw + 0.6, y, T["fig1_alt"])
    ch = caption(x + tw + 0.6, y + fh + CAP_GAP, fw, T["fig1_cap"], name="s1.fig1.caption")
    return max(th, fh + CAP_GAP + ch)


def s1_question(x, y, w):
    return takeaway(x, y, w, T["question"], "s1.take")


def s2_data(x, y, w):
    return body(x, y, w, T["s2_data"], name="s2.data")


def s2_timeline(x, y, w):
    """그림 2: 한 행이 담는 것 (CES 프레임, 추정 대상 프레임, blip 으로 빠진 프레임, 빔, 시간 축)."""
    lab_w = 1.35
    sx0 = x + lab_w
    sw = (w - lab_w - 0.7) / 5
    fpt, fpitch, fpad = max(10.0, DIA_PT - 0.5), round(max(10.0, DIA_PT - 0.5) * 1.28, 1), 0.08
    fh = max(text_h(f.split("\n"), fpt, sw - 0.12 - 2 * fpad, fpitch) for f in T["frames"]) + 0.35
    snap_h = DIA_PT * 1.35 * PT2CM
    fy = y + snap_h + 0.7               # 프레임 줄
    by_ = fy + fh + 0.25                # 빔 막대
    ay = by_ + 0.65                     # 시간 축
    h_fig = ay + 0.2 + 0.5 - y
    cap_h = text_h(T["fig2_cap"], CAP_PT, w, CAP_PITCH)
    if DRY:
        return h_fig + CAP_GAP + cap_h
    styles = [(FRAME, None, False, BODY), (PREV, None, False, INK), (WHITE, NAVY, False, INK),
              (None, MUTED, True, MUTED), (FRAME, None, False, BODY)]
    text(x, fy + fh / 2 - 0.3, lab_w, T["row_ces"], DIA_PT, color=MUTED, bold=True, pitch=14, name="sch.label.ces")
    for i, (lab, (fill, ln, dash, col)) in enumerate(zip(T["frames"], styles)):
        b = box(sx0 + i * sw + 0.06, fy, sw - 0.12, fh, fill=fill, line=ln, lw=2.0 if ln is NAVY else 1.0,
                radius=0.12, dash=dash, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"sch.frame{i}")
        label_in(b, lab.split("\n"), fpt, color=col, pitch=fpitch, pad=fpad)
    text(x, by_ - 0.12, lab_w, T["row_beam"], DIA_PT, color=MUTED, bold=True, pitch=14, name="sch.label.beam")
    box(sx0 + 0.06, by_, 3 * sw - 0.12, 0.32, fill=BEAM, radius=0.05, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        name="sch.beam.on1")
    box(sx0 + 4 * sw + 0.06, by_, sw - 0.12, 0.32, fill=BEAM, radius=0.05, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        name="sch.beam.on2")
    text(sx0 + 3 * sw, by_ - 0.07, sw, T["beam_off"], 10.0, color=MUTED, pitch=13, align=PP_ALIGN.CENTER,
         name="sch.beam.off")
    line(sx0 - 0.1, ay, sx0 + 5 * sw + 0.55, ay, color=MUTED, w=1.25, head=True, name="sch.axis")
    for i, lab in enumerate(("t − 20 ms", "t − 10", "t", "t + 10", "t + 20", "t + 30")):
        line(sx0 + i * sw, ay - 0.12, sx0 + i * sw, ay + 0.12, color=MUTED, w=1.25, name=f"sch.tick{i}")
        text(sx0 + i * sw - 1.4, ay + 0.2, 2.8, lab, 10.5, color=INK if lab == "t" else MUTED, bold=(lab == "t"),
             pitch=13, align=PP_ALIGN.CENTER, name=f"sch.ticklabel{i}")
    mx = sx0 + 2 * sw                   # 시각 t 의 빠른 진단 표본
    text(mx - 7.0, y, 14.0, T["snapshot"], DIA_PT, color=NAVY, bold=True, pitch=DIA_PT * 1.35,
         align=PP_ALIGN.CENTER, name="sch.snapshot.label")
    line(mx, y + snap_h + 0.05, mx, fy - 0.05, color=BLUE, w=2.5, head=True, name="sch.snapshot.arrow")
    caption(x, y + h_fig + CAP_GAP, w, T["fig2_cap"], name="s2.fig2.caption")
    return h_fig + CAP_GAP + cap_h


def s2_base(x, y, w):
    c0 = max(text_w(r[0], TAB_PT, True) for r in T["base_rows"]) + 0.6
    th = booktabs(x, y, (c0, w - c0), T["base_head"], T["base_rows"], "tab.base", aligns=(PP_ALIGN.LEFT, PP_ALIGN.LEFT))
    nh = text(x, y + th + 0.2, w, T["skill_note"], 10.0, color=MUTED, pitch=13.5, align=PP_ALIGN.CENTER,
              name="s2.skill.note")
    return th + 0.2 + nh


def s2_read(x, y, w):
    return body(x, y, w, T["s2_read"], name="s2.read")


def s3_pipeline(x, y, w):
    """그림 3: 두 단계 estimator (편집 가능한 도형). 첫 단계 상자는 두 입력의 높이를 다 덮는다."""
    iw, s1w, s2w, o1w, o2w = T["pipe_w"]
    pad, lpitch = 0.12, round(DIA_PT * 1.27, 1)
    lab_h = lambda lab, bw: text_h(lab, DIA_PT, bw - 2 * pad - 0.05, lpitch)
    ih = max(lab_h(T["pipe_fast"], iw), lab_h(T["pipe_ces"], iw)) + 0.45
    igap = 0.4
    a = (w - iw - s1w - o1w - s2w - o2w) / 5          # 화살표 길이 다섯
    full = 2 * ih + igap
    cy = y + full / 2
    s2h = max(2.4, lab_h(T["pipe_st2"], s2w) + 0.45)
    x_s1 = x + iw + a
    x_o1 = x_s1 + s1w + a
    x_s2 = x_o1 + o1w + a
    x_o2 = x_s2 + s2w + a
    fit_x = x_o1 - 0.2
    fit_w = x + w - fit_x
    lab_y = cy + s2h / 2 + 0.7
    fit_h = text_h(T["pipe_fit"], DIA_PT, fit_w, DIA_PT * 1.3)
    h_fig = max(full, lab_y + fit_h - y)
    cap_h = text_h(T["fig3_cap"], CAP_PT, w, CAP_PITCH)
    if a < 0.45:
        WARN.append("pipeline arrows only %.2f cm" % a)
    if DRY:
        return h_fig + 0.25 + cap_h
    nodes = [(x, y, iw, ih, PANEL, RULE, T["pipe_fast"], "arch.fast"),
             (x, y + ih + igap, iw, ih, PANEL, RULE, T["pipe_ces"], "arch.ces"),
             (x_s1, y, s1w, full, TINT_B, BLUE, T["pipe_st1"], "arch.stage1"),
             (x_s2, cy - s2h / 2, s2w, s2h, TINT_B, BLUE, T["pipe_st2"], "arch.stage2")]
    for (bx, by, bw, bh, fill, ln, lab, nm) in nodes:
        b = box(bx, by, bw, bh, fill=fill, line=ln, lw=1.5 if ln is BLUE else 1.0, radius=0.22,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=nm)
        label_in(b, lab, DIA_PT, color=INK, pitch=lpitch, pad=pad)
    for (ox, ow, lab, nm) in ((x_o1, o1w, "T\u0302_i^(1)(t)", "arch.out1"), (x_o2, o2w, "T\u0302_i(t)", "arch.out2")):
        b = box(ox, cy - 0.6, ow, 1.2, fill=WHITE, line=NAVY, lw=1.5, shape=MSO_SHAPE.OVAL, name=nm)
        label_in(b, lab, 13.5, color=NAVY, bold=True, pitch=16, pad=0.02)
    for k, yy in enumerate((y + ih / 2, y + ih + igap + ih / 2)):
        line(x + iw + 0.06, yy, x_s1 - 0.06, yy, color=NAVY, w=1.75, head=True, name=f"arch.in{k}")
    line(x_s1 + s1w + 0.06, cy, x_o1 - 0.06, cy, color=NAVY, w=1.75, head=True, name="arch.a1")
    line(x_o1 + o1w + 0.06, cy, x_s2 - 0.06, cy, color=NAVY, w=1.75, head=True, name="arch.a2")
    line(x_s2 + s2w + 0.06, cy, x_o2 - 0.06, cy, color=NAVY, w=1.75, head=True, name="arch.a3")
    mx2 = x_s2 + s2w / 2
    line(mx2, lab_y - 0.05, mx2, cy + s2h / 2 + 0.06, color=BLUE, w=1.75, head=True, dash=True, name="arch.fit")
    text(fit_x, lab_y, fit_w, T["pipe_fit"], DIA_PT, color=BLUE, pitch=DIA_PT * 1.3, align=PP_ALIGN.CENTER,
         name="arch.fit.label")
    caption(x, y + h_fig + 0.25, w, T["fig3_cap"], name="s3.fig3.caption")
    return h_fig + 0.25 + cap_h


def s3_panel(x, y, w):
    """①②③ 항목(둥근 상자). ③ 아래에 한 행 계산 시간의 작은 표를 항목 글과 같은 들여쓰기로 둔다."""
    pad, bd, gap, tgap = 0.5, 0.78, 0.55, 0.3
    tx = x + pad + bd + 0.45
    tw = x + w - pad - tx
    hs = [text_h(s, BODY_PT, tw, BODY_PITCH) for s in T["s3_items"]]
    cpt = TAB_PT
    c0 = max(text_w(s, cpt) for s in (T["calc_head"][0], *T["calc_rows"])) + 0.5
    c1 = max(text_w(s, cpt) for s in (T["calc_head"][1], *V_CALC)) + 0.6
    rows = list(zip(T["calc_rows"], V_CALC))
    calc = lambda yy: booktabs(tx, yy, (c0, c1), T["calc_head"], rows, "tab.calc",
                               aligns=(PP_ALIGN.LEFT, PP_ALIGN.RIGHT), pady=0.09)
    th = measure(calc, 0.0)
    h = 2 * pad + sum(hs) + gap * (len(hs) - 1) + tgap + th
    box(x, y, w, h, fill=WHITE, line=RULE, lw=1.0, radius=0.35, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="s3.panel")
    yy = y + pad
    for i, (s, hh) in enumerate(zip(T["s3_items"], hs)):
        badge(x + pad, yy + (min(hh, 2 * BODY_PITCH * PT2CM) - bd) / 2, bd, i + 1, 14, f"s3.item{i}.badge")
        text(tx, yy, tw, s, BODY_PT, pitch=BODY_PITCH, name=f"s3.item{i}.text")
        yy += hh
        if i < len(hs) - 1:
            line(x + pad, yy + gap / 2, x + w - pad, yy + gap / 2, color=rgb("E1E5EB"), w=0.75, name=f"s3.sep{i}")
            yy += gap
    calc(yy + tgap)
    return h


# ================================================================== 오른쪽 단: 4 결과 · 5 정보 · 6 결론 · 7 한계
def s4_table(x, y, w):
    ch = caption(x, y, w, T["t1_cap"], name="s4.tab1.caption")
    yy = y + ch + CAP_GAP
    c0 = T["t1_c0"]
    rows = [(lab,) + v for lab, v in zip(T["t1_rows"], V_T1)]
    th = booktabs(x, yy, (c0, (w - c0) / 2, (w - c0) / 2), T["t1_head"], rows, "tab1", mid_rules=(5,), fills=(2, 3))
    yy += th + CAP_GAP
    nh = caption(x, yy, w, T["t1_note"], name="s4.tab1.note")
    return yy + nh - y


def s4_bullets(x, y, w):
    return items(x, y, w, T["s4_bullets"], "•", "s4.b", gap=0.2)


def side_rows(labels, values, indent):
    ind = nbsp(text_w(indent, SIDE_PT) if isinstance(indent, str) else indent, SIDE_PT)
    return [(lab.replace("⇥", ind), v) for lab, v in zip(labels, values)]


def s5_tables(x, y, w):
    gap = 0.6
    tw = (w - gap) / 2
    tabs = ((T["t2_cap"], T["t2_head"], side_rows(T["t2_rows"], V_T2, ""), "tab2"),
            (T["t3_cap"], T["t3_head"], side_rows(T["t3_rows"], V_T3, T["t3_indent"]), "tab3"))
    cap_h = max(measure(caption, 0, 0, tw, cap) for cap, _, _, _ in tabs)
    hs = []
    for j, (cap, head, rows, key) in enumerate(tabs):
        tx = x + j * (tw + gap)
        caption(tx, y, tw, cap, name=f"s5.{key}.caption")
        cv = max(text_w(r[1], SIDE_PT) for r in rows) + 0.3
        hs.append(booktabs(tx, y + cap_h + CAP_GAP, (tw - cv, cv), head, rows, key, pt=SIDE_PT, pitch=SIDE_PITCH,
                           padx=0.1, aligns=(PP_ALIGN.LEFT, PP_ALIGN.RIGHT),
                           fills=[i for i, r in enumerate(rows) if excludes0(r[1])]))
    return cap_h + CAP_GAP + max(hs)


def s5_bullets(x, y, w):
    return items(x, y, w, T["s5_bullets"], "•", "s5.b", gap=0.2)


def s6_take(x, y, w):
    return takeaway(x, y, w, T["s6_take"], "s6.take")


def s6_table(x, y, w):
    ch = caption(x, y, w, T["t4_cap"], name="s6.tab4.caption")
    cv = max(text_w(v, TAB_PT) for v in V_T4) + 0.6
    rows = list(zip(T["t4_rows"], V_T4))
    th = booktabs(x, y + ch + CAP_GAP, (w - cv, cv), T["t4_head"], rows, "tab4", aligns=(PP_ALIGN.LEFT, PP_ALIGN.CENTER))
    return ch + CAP_GAP + th


def s6_checks(x, y, w):
    return items(x, y, w, T["s6_checks"], "✓", "s6.c", gap=0.2)


def s7_limits(x, y, w):
    """번호 항목과 "→" 처방. 그림 4는 ①의 오른쪽에 두고, 그림 아래로 내려간 항목은 단 너비를 다 쓴다."""
    bd, gap, fgap = 0.78, 0.5, 0.6
    fw, fh = fig_size(FIG4)
    cw = fw + fgap - 0.1                      # 설명은 그림 왼쪽 틈까지 써서
    ch = text_h(T["fig4_cap"], CAP_PT, cw, CAP_PITCH)
    fig_bottom = y + fh + CAP_GAP + ch
    yy = y
    for i, (head, fix) in enumerate(T["s7_items"]):
        narrow = yy < fig_bottom - 0.6               # 그림 옆이면 좁게, 그림 끝에 가까우면 그 아래로 내려 넓게
        if not narrow:
            yy = max(yy, fig_bottom + 0.3)
        iw = (w - fw - fgap if narrow else w) - bd - 0.32
        ix = x + bd + 0.32
        badge(x, yy + 0.06, bd, i + 1, 14, f"s7.item{i}.badge")
        yy += text(ix, yy, iw, head, BODY_PT, pitch=BODY_PITCH, name=f"s7.item{i}.text") + 0.06
        yy += text(ix, yy, iw, fix, BODY_PT, color=NAVY, pitch=BODY_PITCH, name=f"s7.item{i}.fix")
        if i < len(T["s7_items"]) - 1:
            yy += gap
    pic(FIG4, x + w - fw, y, T["fig4_alt"])
    caption(x + w - cw, y + fh + CAP_GAP, cw, T["fig4_cap"], name="s7.fig4.caption")
    return max(yy, fig_bottom) - y


S = T["sections"]
LEFT = [(1, S[0], "s1", [s1_intro, s1_question]),
        (2, S[1], "s2", [s2_data, s2_timeline, s2_base, s2_read]),
        (3, S[2], "s3", [s3_pipeline, s3_panel])]
RIGHT = [(4, S[3], "s4", [s4_table, s4_bullets]),
         (5, S[4], "s5", [s5_tables, s5_bullets]),
         (6, S[5], "s6", [s6_take, s6_table, s6_checks]),
         (7, S[6], "s7", [s7_limits])]

# ================================================================== 그리기
band_h = title_band()
Y0 = band_h + 1.15
def col_share(x, w, secs):
    c, nb, ns = col_content(x, w, secs)
    return ((Y1 - Y0) - c - nb * GB - ns * GS) / (nb + 2 * ns)


best = {}
for b_ in SIZE_STEPS:          # 크기마다 두 단의 여백이 가장 고른 폭 나눔을 찾고, 그 작은 쪽 여백을 적는다
    set_sizes(b_)
    opts = []
    for d in SHIFTS:
        wl, wr = COL_W - d / 2, COL_W + d / 2
        sl, sr = col_share(M, wl, LEFT), col_share(M + wl + GUT, wr, RIGHT)
        opts.append((abs(sl - sr), d, min(sl, sr)))
    _, d, mn = min(opts)
    best[b_] = (d, mn)
    if mn >= MIN_SLACK:
        break
pick = b_                      # 조건을 채운 첫 크기, 없으면 가장 작은 크기
set_sizes(pick)
print("size search (body pt: column shift cm, smaller per-gap slack cm): " +
      ", ".join("%.1f: %.1f, %+.2f" % (b_, d_, m_) for b_, (d_, m_) in best.items()))
SHIFT = best[pick][0]
W_L, W_R = COL_W - SHIFT / 2, COL_W + SHIFT / 2
XL, XR = M, M + W_L + GUT
print("lang %s: body %.1f pt, table %.1f, side tables %.1f, captions %.1f, takeaway %.1f, heading %.1f; "
      "columns %.1f | %.1f cm" % (LANG, BODY_PT, TAB_PT, SIDE_PT, CAP_PT, TAKE_PT, HEAD_PT, W_L, W_R))
free_l = draw_column(XL, W_L, LEFT, Y0, Y1, "left")
free_r = draw_column(XR, W_R, RIGHT, Y0, Y1, "right")
line(M + W_L + GUT / 2, Y0 + 0.1, M + W_L + GUT / 2, Y1, color=RULE, w=1.0, name="column.divider")

# ================================================================== 아래: 참고문헌 · 사사
line(M, FOOT_Y, W_CM - M, FOOT_Y, color=RULE, w=1.0, name="footer.rule")
fh_ = text(M, FOOT_Y + 0.3, W_CM - 2 * M, T["refs"], 10.5, color=MUTED, pitch=14, after=3, name="footer.text")
print("footer ends %.2f cm above the sheet edge" % (H_CM - (FOOT_Y + 0.3 + fh_)))
for w_ in dict.fromkeys(WARN):
    print("  WARN", w_)

prs.core_properties.title = T["doc_title"]
prs.core_properties.author = "" if args.anon else T["doc_author"]
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT)


def export_png(pptx, png, px):
    """PowerPoint COM 으로 첫 슬라이드를 PNG 로. 창 없이 열고, 열린 문서가 없을 때만 Quit."""
    import win32com.client
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(Path(pptx).resolve()), True, False, False)  # ReadOnly, Untitled, WithWindow
    try:
        pres.Slides(1).Export(str(Path(png).resolve()), "PNG", px, round(px * H_CM / W_CM))
    finally:
        pres.Close()
        if app.Presentations.Count == 0:
            app.Quit()


if not args.no_png:
    export_png(OUT, PNG, args.png_px)
    print("png", PNG)
