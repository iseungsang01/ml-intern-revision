# -*- coding: utf-8 -*-
"""Nine-slide talk (PPCF poster sections 1-8 + rotation) as an editable PowerPoint file.

    py docs/presentation/build_deck_ppcf.py [template.pptx] [out.pptx]

The background is the layouts of the author's thesis template (`졸논 정리.pptx`: three top bars, navy title band,
navy block on the title slide); the template's own slides are dropped and its document properties cleared.
Text follows the claude.ai deck of 2026-11-19: section names as titles, short factual lines, result numbers only in
tables (blue = 95 % interval excludes 0, shaded rows = emphasis), explanations in the speaker notes.
Figures are the Korean poster figures in figures/poster_ppcf/.
"""
import copy
import io
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "poster_ppcf"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\lss\Desktop\원핵 졸논\졸논 정리.pptx")
DECK_IN = HERE.parents[1] / "data" / "deck_inputs"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1] / "data" / "deck_out" / "KSTAR_CES_발표_PPCF.pptx"

FONT = "맑은 고딕"
NAVY, TEAL, GREY = RGBColor(0x1A, 0x32, 0x60), RGBColor(0x45, 0x90, 0xB8), RGBColor(0x96, 0x9F, 0xA7)
INK, BODY, MUTED = RGBColor(0x2B, 0x2F, 0x36), RGBColor(0x3D, 0x3D, 0x3D), RGBColor(0x5D, 0x64, 0x71)
BLUE, SHADE, HEAD = RGBColor(0x1F, 0x5F, 0xAE), RGBColor(0xE8, 0xF0, 0xFA), RGBColor(0xEB, 0xEB, 0xEB)
WHITE, SOFT = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xD3, 0xDC, 0xEA)


def px(v):
    """1920 x 1080 design pixels -> EMU (12192000 EMU wide)."""
    return Emu(int(round(v * 6350)))


def pt(pxsize):
    return Pt(pxsize / 2.0)  # 1920 px = 960 pt


def style_run(run, size, color, bold=False):
    f = run.font
    f.size, f.bold, f.name = pt(size), bold, FONT
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "ko-KR")
    rpr.set("altLang", "en-US")
    rpr.set("cap", "none")  # the template's title style is all caps
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def fill_tf(tf, lines, size, color, bold=False, align=PP_ALIGN.LEFT, space=6, line=1.25):
    """lines: list of str or list of (text, color, bold) segments."""
    tf.clear()
    tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        p.line_spacing = line
        p._p.get_or_add_pPr().set("latinLnBrk", "0")
        segs = ln if isinstance(ln, list) else [(ln, color, bold)]
        for text, c, b in segs:
            style_run(p.add_run(), size, c, b)
            p.runs[-1].text = text
    return tf


def textbox(slide, x, y, w, h, lines, size=28, color=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, **kw):
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = px(4)
    tf.margin_top = tf.margin_bottom = px(2)
    tf.vertical_anchor = anchor
    fill_tf(tf, lines, size, color, bold, align, **kw)
    return tb


def bullets(slide, x, y, w, h, items, size=28, color=BODY):
    return textbox(slide, x, y, w, h, ["• " + t for t in items], size, color, space=8)


def picture(slide, name, x, y, w):
    path = name if isinstance(name, Path) else FIG / name
    iw, ih = Image.open(path).size
    h = w * ih / iw
    slide.shapes.add_picture(str(path), px(x), px(y), px(w), px(h))
    return y + h


MISS, HELD = RGBColor(0xFB, 0xE3, 0xD2), RGBColor(0xE2, 0xE2, 0xE2)


def table(slide, x, y, w, rows, widths, size=24, rh=46):
    """rows: list of (cells, kind) with kind in {'hd','hi','grp',''}; a cell is str or (str, tone) with tone
    'y' (blue: interval excludes 0), 'n', 'm' (missing value, orange fill) or 'h' (held repeat, grey fill)."""
    nr, nc = len(rows), len(widths)
    gs = slide.shapes.add_table(nr, nc, px(x), px(y), px(w), px(rh * nr))
    tbl = gs.table
    tblpr = gs._element.graphic.graphicData.tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblpr.set(a, "0")
    for j, fr in enumerate(widths):
        tbl.columns[j].width = px(w * fr)
    for i, (cells, kind) in enumerate(rows):
        tbl.rows[i].height = px(rh)
        for j in range(nc):
            cell = tbl.cell(i, j)
            raw = cells[j] if j < len(cells) else ""
            text, tone = (raw, "") if isinstance(raw, str) else raw
            color = BLUE if tone == "y" else (TEAL if kind == "grp" and j == 0 else BODY)
            fill_tf(cell.text_frame, [text], size, color, bold=(kind == "hd"), space=0, line=1.0)
            cell.margin_left = cell.margin_right = px(10)
            cell.margin_top = cell.margin_bottom = px(5)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            row_fill = HEAD if kind == "hd" else (SHADE if kind == "hi" else WHITE)
            cell.fill.fore_color.rgb = MISS if tone == "m" else (HELD if tone == "h" else row_fill)
    return y + rh * nr


def page_number(slide, n, color=GREY):
    textbox(slide, 1662, 937, 165, 57, [str(n)], 24, color, align=PP_ALIGN.RIGHT)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def drop_template_slides(prs):
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)


def content_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = px(110), px(110), px(1700), px(159)
    t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_tf(t.text_frame, [title], 50, WHITE, bold=True, space=0, line=1.0)
    return s


def csv_excerpt(path, t0, t1):
    """Rows of a 10 ms file between t0 and t1 (inclusive): time, CES_TI, CES_VT, first BES channel."""
    import csv
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    h = rows[0]
    it, iti, ivt = h.index("time"), h.index("CES_TI"), h.index("CES_VT")
    ib = [i for i, c in enumerate(h) if c.startswith("BES")][0]
    out, prev_vt = [], None
    for r in rows[1:]:
        t = float(r[it])
        if t0 - 1e-6 <= t <= t1 + 1e-6:
            ti, vt = r[iti].strip(), r[ivt].strip()
            ti_c = (f"{float(ti):.2f}", "n") if ti else (" ", "m")
            if not vt:
                vt_c = (" ", "m")
            else:
                vt_c = (f"{float(vt):.2f}", "h" if vt == prev_vt else "n")
            out.append(([f"{t:.3f}", ti_c, vt_c, f"{float(r[ib]):.4f}"], ""))
        if r[ivt].strip():
            prev_vt = r[ivt].strip()
    return h[ib], out


NOTE = {}
NOTE["cover"] = """제목: 같은 위치의 빠른 진단으로 KSTAR edge 측정선의 전하 교환 이온 온도를 인과적으로 nowcasting하기.
nowcasting: 지금 프레임의 값을 그 프레임의 CES 값 없이 추정하는 것. 인과적: 지금 시각까지의 정보만 쓴다는 뜻."""
NOTE["background"] = """배경: CES는 탄소 C VI 이온의 온도 T_i와 회전 V_rot을 10 ms 프레임마다 보고합니다. 자료 제공자가 pedestal top으로 고른 edge 측정선에서는 값이 자주 빠집니다.
왼쪽 표: 격자 245,925행(첫·끝 행 제외) 기준으로 T_i는 7.7 %, V_rot은 23.5 %가 완전 결측입니다. V_rot은 관측의 54 %가 직전 값의 반복이라 격자의 41.3 %가 유지값이고, 독립 측정은 35.2 %뿐입니다. T_i의 유지값은 226,991개 중 1개입니다. T_i 결측은 BES 세기가 낮은 5분위에서 18.0 % [15.9, 20.0], 높은 5분위에서 3.4 % [2.7, 4.1]로 무작위가 아닙니다.
오른쪽 표: 실제 파일 #30909의 3.47–3.58 s. 주황 칸이 빈 값, 회색 칸이 직전 값의 반복입니다. T_i는 빈 칸이 섞여 있고, V_rot은 16.06이 계속 되풀이됩니다.
두 표적은 서로 독립적으로 빠지므로, 표적마다 따로 가린 손실(masked loss)로 학습합니다.
UFCES(KSTAR 초고속 CES, 100 kHz)는 요동 측정용 설계이고 실시간 분석은 다루지 않습니다.
문제: 감시·제어에는 다음 CES 값 없이 지금 프레임에서 만든 T_i 추정이 필요합니다. 이 발표는 같은 위치 BES의 지금 프레임 값이 CES 이력에 더하는 정보를 잽니다.

용어 풀이. 유지값: 새 측정 없이 직전 값이 다시 적힌 값. 학습·입력·기준에서 모두 뺀다. BES(빔 방출 분광): 가열 빔이 내는 빛의 세기, 밀도와 빔에 의존."""
NOTE["data"] = """자료는 KSTAR 2022 캠페인의 방전 파일 641개, 10 ms 격자 245,925행입니다. 분할에 들어가는 방전은 639개입니다.
그림 1: 빨간 선이 CES 측정선 R = 2.198 m이고, 아래 막대가 BES 시야, BES 5–7열, ECEI 3열입니다. 빗금은 채널 이름에서 추정한 위치입니다.
그림 2: 빠른 채널 값은 그 행 시각의 원시 표본 하나이고, CES 값은 그 시각 근처에서 노출이 시작되는 프레임입니다. 행 간격이 20–40 ms이면 빔 blip이고, 그 프레임은 파일에 없습니다. 행 t의 인과 추정은 t까지의 빠른 표본과 t 이전의 CES 값만 씁니다.
3 keV 넘는 값은 fit 실패로 보고 빠진 값으로 다루고, 직전 값을 되풀이한 유지값은 뺍니다."""
NOTE["layout"] = """왼쪽: KFE 발표 자료(Current status of the KSTAR diagnostics, 2024)의 진단 배치. 위에서 본 배치에서 NBI 1 빔과 그 빔을 보는 Toroidal CES, BES, MSE의 시선, 그리고 ECE 위치를 짚습니다. 오른쪽 단면 배치에서 CES·BES·ECE가 보는 높이(z ≈ 0)를 짚습니다.
오른쪽: 받은 Mirnov 배치도. 토로이달 배열 20개 코일 가운데 파일에 들어 있는 두 코일 MC1T03, MC1T16을 이름의 번호로 표시했습니다. Mirnov 코일은 벽에 있습니다."""
NOTE["positions"] = """받은 ABCD 위치 그림(syndia 보정 전)에 파일의 채널을 표시했습니다. 빨간 원이 BES 9채널(105–107, 205–207, 305–307), 파란 원이 ECEI GT1103–GT1403입니다.
굵은 점선이 CES 측정선 ch18, R = 2.198 m이고, 가는 점선이 이웃 측정선 2.192 m, 2.203 m입니다. BES와 ECEI 채널은 측정선에서 1–2 cm 안에 있습니다.
오른쪽 표가 자료의 요약입니다."""
NOTE["discharges"] = """방전 639개를 한 점씩 그렸습니다. 왼쪽부터 방전 번호에 따른 Ti 중앙값, Vrot 중앙값과 Ti 중앙값, BES 수준과 Ti 결측 비율입니다.
색은 BES 버스트 빈도(초당), 회색은 파일이 짧아 빈도를 정할 수 없는 방전입니다. 빈 네모는 캠페인 마지막 97개 방전으로, 시간순 확증에서 시험 블록으로 씁니다.
Ti 중앙값은 3 keV 넘는 값을 뺀 뒤, Vrot 중앙값은 직전 값의 반복(유지값)을 뺀 뒤의 값입니다. 가운데 패널은 −100–250 km/s 밖의 방전을 생략했습니다(개수는 그림에 표시).
BES 수준이 낮은 방전일수록 Ti 결측이 많습니다."""
NOTE["estimators"] = """그림 3 (a): 첫 단계는 격자의 모든 행을 차례로 읽는 LSTM으로, 지금 행의 빠른 채널과 CES 이력(마지막 값, 경과 시간, 관측 여부)을 읽습니다. 둘째 단계는 첫 단계의 오차를 gradient boosting으로 추정해 더하며, 학습 방전에서 만든 fold 밖 오차로만 맞춥니다.
(b): log T_i의 강건 Kalman filter에 BES 변화 두 가지(공통 성분 c, 반경 방향 차이 g)를 넣은 5-parameter filter. 구동이 없는 것이 KF0, 있는 것이 KFX.
표: offline GP는 다음 값을 쓰므로 실시간으로는 쓸 수 없는 참고 기준입니다. 비교군과의 차이를 빠른 채널의 증분이라 부릅니다.
skill이 0보다 크면 기준보다 평균 제곱 오차가 작습니다. 채점은 감춘 측정 행에서 하고, 방전 단위로 나눠 교차 검증을 반복해 합칩니다."""
NOTE["result"] = """표 2: 행은 기준, 열은 신경망 혼자(첫 단계)와 두 단계 estimator. 파란 값은 95 % 구간이 0(비율이면 1)을 제외.
신경망 혼자: hold, 인과 GP, 비교군 대비 구간이 0 위. offline GP 대비는 구간이 0 아래(offline GP의 오차가 더 낮음).
두 단계 estimator: 모든 기준 대비 skill이 올라가고 offline GP 대비 구간도 0 위. 비교군 대비 증분은 0.048에서 0.097. 등록한 진술 여섯 개가 모두 성립.
아래 칸: RMSE는 인과 GP보다 17.8 eV 낮고, 중앙값 오차는 hold의 0.909배, 20 % 이내 비율도 인과 GP보다 높음.
그림 4: 결측이 많은 두 방전과 전형적인 한 방전의 예. 빨간 선이 신경망, 파란 점선이 인과 GP.
신경망 열은 재분할 8개, 두 단계 열은 재분할 3개(15 fold), 신경망마다 초기화 하나."""
NOTE["information"] = """표 3: 값은 모두 비교군 대비 skill.
첫 줄: 학습된 신경망에 BES만 한 프레임 늦은 값을 넣으면 증분이 0.
다시 학습한 신경망: 빠른 채널 전부를 한 행 늦게 넣으면 증분이 없음 → 10 ms 앞선 전조가 아니라 같은 프레임의 정보. BES와 CES 이력만으로 증분이 있고, 전체 신경망과 BES만 읽는 신경망은 구분되지 않음. ECEI와 이력만으로는 증분 없음. Mirnov를 빼도 손실 없음.
마지막 줄: 5-parameter filter에서 BES 구동이 있는 KFX가 없는 KF0보다 +0.049, 신경망의 증분과 같은 크기.
주의: BES는 가열 빔의 빛이라 밀도와 빔에 함께 의존. 파일에 빔 출력이 없어 빔 변화의 기여는 배제 못 함."""
NOTE["where"] = """그림 5: 프레임 하나가 빠진 빔 blip 1,262건 뒤의 사건 평균(앞선 여섯 행 대비 비). BES는 두 행 안에 회복하고, 보고된 T_i는 빔도 분광기도 쓰지 않는 ECEI처럼 천천히 회복.
표 4 위: 증분은 빠진 프레임 뒤 1–3행에서 가장 큼. 비교군은 늘어난 시간 간격은 보지만 감소의 깊이는 보지 못함. 중간·가장 뜨거운 3분의 1에서는 구간이 0 위, 가장 차가운 3분의 1에서는 0 포함. 이득이 가장 큰 방전 64개(약 10분의 1)를 빼면 구간이 0 포함.
표 4 아래: 다음 프레임은 따로 fit되는데 빠른 채널 보정의 0.810이 남음. 보고값의 프레임별 독립 산포의 바닥은 RMS 135.7 eV, 신경망의 평균 제곱 오차는 그 1.331배."""
NOTE["conclusion"] = """결과는 모두 3 keV 컷 모집단에 대한 것.
왼쪽: 비교군 대비 증분, 같은 크기의 filter 증분, 한 프레임 늦은 BES에서 증분 없음, 두 단계 estimator의 offline GP 대비 skill과 중앙값 오차, 다음 프레임에 남는 보정.
오른쪽: 신경망 혼자일 때의 단서. 이득은 보고값의 산포에 비해 작고, 가장 차가운 3분의 1에서 확인되지 않으며, 이득이 큰 방전 10분의 1을 빼면 구간이 0을 포함하고, 단일 분할에서는 확인되지 않음. 그래서 주 진술은 pooled 판독으로 함."""
NOTE["limits"] = """1. 컷 모집단 한정: 3 keV 넘는 값을 남기면 인과 GP 대비 여유는 남지만 비교군 대비 증분은 구간이 0을 포함(표 5). 프레임별 fit 불확도가 있으면 fit 실패와 플라스마를 가를 수 있음.
2. 측정선 하나, 캠페인 하나, 장치 하나. C VI 불순물 온도, 파일에 빔 출력 없음.
3. 손대지 않은 hold-out 방전 없음 → 캠페인 마지막 97개 방전으로 시간순 확증 진행 중.
4. KSTAR에서 BES·CES 실시간 사용 발표 없음. 지금 프레임의 BES와 두 프레임 이내의 CES 값이 필요.
다음 단계: 3 keV 넘는 값이 증분을 지우는 이유, 가설 넷. CPU로 되는 뒤의 둘을 먼저 잼."""
NOTE["vrot"] = """V_rot은 지운 것이 아니라 보충 자료로 옮긴 것.
V_rot 가지는 CES 이력만 읽고 빠른 채널은 읽지 않음. 빠른 채널 증분과 V_rot 증분의 상관이 |r| 0.044 이하(T_i는 하한 +0.127)라서, 빠른 채널 증분이라는 판독이 V_rot에는 없음.
V_rot 자료: 격자의 23.5 % 결측, 관측의 54 %가 직전 값 반복, 독립 측정은 격자의 35.2 %.
표 6: hold·running mean 대비 구간이 0 위. 인과 GP 대비는 pooled로만 0 위(분할별 1/4). PCHIP 대비 하한이 겨우 0 위, offline GP 대비 0 포함.
표 7 마지막 줄: 빔 blip 뒤 첫 행의 회전은 직전 수준의 0.973배, 구간이 1을 제외(본문에 남긴 회전 판독)."""


def build():
    prs = Presentation(str(TEMPLATE))
    drop_template_slides(prs)
    cp = prs.core_properties
    cp.author = cp.last_modified_by = ""
    cp.title, cp.subject, cp.keywords, cp.comments = "KSTAR CES PPCF 발표", "", "", ""

    # cover
    s = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = px(91), px(150), px(1731), px(300)
    t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_tf(t.text_frame, ["같은 위치의 빠른 진단으로 KSTAR edge 측정선의 전하 교환 이온 온도를 인과적으로 nowcasting하기"],
            54, RGBColor(0x33, 0x33, 0x33), bold=True, align=PP_ALIGN.RIGHT, space=0, line=1.15)
    textbox(s, 130, 840, 1660, 110, ["이승상 · 서울대학교 원자핵공학과", "KSTAR 2022 캠페인 자료 (KSTAR team 제공)"],
            30, WHITE, align=PP_ALIGN.CENTER, space=6)
    notes(s, NOTE["cover"])

    # 1 background and problem
    s = content_slide(prs, "1 배경과 문제")
    textbox(s, 91, 322, 1000, 45, ["배경: 결측 실태 (격자 245,925행 대비)"], 28, NAVY, bold=True)
    y = table(s, 91, 372, 1000, [
        (["항목", "Ti", "Vrot"], "hd"),
        (["완전 결측 (NaN)", "7.7 %", "23.5 %"], ""),
        (["직전 값의 반복 (유지값)", "1개", "41.3 %"], ""),
        (["독립 측정", "92.3 %", "35.2 %"], "hi"),
        (["Ti 결측: BES 세기 최하위 5분위", "18.0 % [15.9, 20.0]", "—"], ""),
        (["Ti 결측: BES 세기 최상위 5분위", "3.4 % [2.7, 4.1]", "—"], ""),
    ], [0.48, 0.30, 0.22], rh=44)
    bullets(s, 91, y + 14, 1000, 90, [
        "Ti와 Vrot이 서로 독립적으로 결측 → 표적별 masked loss",
        "UFCES (KSTAR 초고속 CES, 100 kHz): 요동 측정용 설계, 실시간 분석 없음"], 26)
    textbox(s, 91, y + 134, 1000, 45, ["문제"], 28, NAVY, bold=True)
    bullets(s, 91, y + 182, 1000, 90, [
        "감시·제어: 다음 CES 값 없이 지금 프레임에서 만든 Ti 추정 필요",
        "대상: 같은 위치 BES의 지금 프레임 값이 CES 이력에 더하는 정보"], 26)
    bes_name, ex = csv_excerpt(HERE.parents[1] / "data" / "s30909.csv", 3.472, 3.582)
    textbox(s, 1130, 322, 695, 45, ["원본 파일 예: #30909, 3.47–3.58 s"], 24, MUTED)
    ye = table(s, 1130, 366, 695, [(["time (s)", "CES_TI", "CES_VT", bes_name], "hd")] + ex,
               [0.22, 0.26, 0.24, 0.28], size=22, rh=38)
    textbox(s, 1130, ye + 6, 695, 40, ["주황: 빈 값 · 회색: 직전 값의 반복"], 24, MUTED)
    page_number(s, 2)
    notes(s, NOTE["background"])

    # 2a layout of the diagnostics
    s = content_slide(prs, "2 진단 배치")
    yb = picture(s, DECK_IN / "kfe_layout_p15.png", 91, 330, 1100)
    textbox(s, 91, yb + 8, 1100, 40, ["KSTAR 진단 배치 (J. Ko, Current status of the KSTAR diagnostics, KFE, 2024, 15쪽)"], 22, MUTED)
    ym = picture(s, DECK_IN / "mirnov_marked.png", 1235, 330, 590)
    textbox(s, 1235, ym + 8, 590, 70, ["Mirnov 토로이달 배열 20개 (제공 그림), 파일의 두 코일 표시"], 22, MUTED)
    page_number(s, 3)
    notes(s, NOTE["layout"])

    # 2b channel positions
    s = content_slide(prs, "2 채널 위치")
    yb = picture(s, DECK_IN / "abcd_marked.png", 91, 330, 1240)
    textbox(s, 91, yb + 8, 1240, 70, ["제공된 ABCD 위치 (syndia 보정 전). 원: 파일의 BES 9채널(빨강)과 ECEI GT1103–GT1403(파랑). 굵은 점선: CES ch18, 가는 점선: 이웃 측정선 (2.192, 2.203 m)"], 22, MUTED)
    table(s, 1370, 330, 455, [
        (["항목", "값"], "hd"),
        (["방전 파일", "641개 (분할 639개)"], ""),
        (["격자", "10 ms, 245,925행"], ""),
        (["CES 측정선", "ch18, R = 2.198 m"], ""),
        (["BES", "9채널 (3×3)"], ""),
        (["ECEI", "GT1103–GT1403"], ""),
        (["Mirnov", "MC1T03, MC1T16"], ""),
        (["제외", "3 keV 초과, 유지값"], ""),
    ], [0.38, 0.62], size=22, rh=44)
    page_number(s, 4)
    notes(s, NOTE["positions"])

    # 2c discharges
    s = content_slide(prs, "2 방전 분포")
    yb = picture(s, HERE / "figures" / "deck_ppcf" / "dataset_overview_ko.png", 91, 330, 1734)
    textbox(s, 91, yb + 8, 1734, 70, ["방전마다 한 점. 색: BES 버스트 빈도, 빈 네모: 캠페인 마지막 97개 방전(시간순 시험 블록). Ti 중앙값은 3 keV 컷 뒤, Vrot 중앙값은 유지값을 뺀 뒤의 값"], 22, MUTED)
    page_number(s, 5)
    notes(s, NOTE["discharges"])

    # 3 estimators
    s = content_slide(prs, "3 estimator와 채점")
    yb = picture(s, "poster_schematic_ko.png", 91, 330, 900)
    textbox(s, 91, yb + 10, 900, 40, ["그림 3. (a) 두 단계 estimator. (b) 5-parameter filter (KF0: 구동 없음, KFX: BES 구동)."], 24, MUTED)
    y = table(s, 1035, 330, 790, [
        (["기준", "읽는 것"], "hd"),
        (["hold", "마지막 관측값"], ""),
        (["인과 GP", "최근 CES 값 16개에 맞춘 Matérn-3/2 GP (주 기준)"], ""),
        (["offline GP", "양쪽 이웃 (다음 값 사용, 실시간 불가)"], ""),
        (["비교군", "빠른 채널만 뺀 같은 신경망 (차이 = 빠른 채널의 증분)"], "hi"),
    ], [0.26, 0.74])
    textbox(s, 1035, y + 30, 790, 40, ["skill = 1 − MSE(estimator) / MSE(기준)"], 24, MUTED)
    bullets(s, 1035, y + 85, 790, 220, ["채점: 감춘 측정 행의 오차", "방전 단위 5-fold 교차 검증, 나눔을 바꿔 반복 (pooled)",
                                        "방전 단위 bootstrap 95 % 구간", "진술과 판정 규칙: 실행 전 등록"], 24)
    page_number(s, 6)
    notes(s, NOTE["estimators"])

    # 4 result
    s = content_slide(prs, "4 주 결과")
    textbox(s, 91, 322, 1100, 40, ["표 2. 3 keV 초과 제외 · 신경망: 재분할 8개(r2–r9) · 나머지: 재분할 3개(r2–r4, 15 fold)"], 24, MUTED)
    y = table(s, 91, 366, 1100, [
        (["기준 대비 skill", "신경망 (첫 단계)", "두 단계 estimator"], "hd"),
        (["hold", ("+0.375 [+0.352, +0.394]", "y"), ("+0.436 [+0.418, +0.452]", "y")], ""),
        (["인과 GP", ("+0.089 [+0.064, +0.112]", "y"), ("+0.176 [+0.155, +0.196]", "y")], "hi"),
        (["offline GP (다음 값 사용)", ("−0.037 [−0.068, −0.008]", "y"), ("+0.063 [+0.038, +0.086]", "y")], "hi"),
        (["비교군 (증분)", ("+0.048 [+0.022, +0.073]", "y"), ("+0.097 [+0.077, +0.116]", "y")], ""),
        (["r2–r4 행의 오차", "", ""], "grp"),
        (["RMSE (eV)", "184.2 [174.3, 194.4]", "175.4 [165.9, 185.3]"], ""),
        (["RMSE − 인과 GP (eV)", ("−9.1 [−11.5, −6.6]", "y"), ("−17.8 [−20.2, −15.4]", "y")], ""),
        (["중앙값 |오차| ÷ hold", "1.010 [0.972, 1.062]", ("0.909 [0.889, 0.931]", "y")], ""),
        (["20 % 이내 − 인과 GP", "−0.002 [−0.013, +0.007]", ("+0.021 [+0.018, +0.025]", "y")], ""),
    ], [0.34, 0.33, 0.33])
    textbox(s, 91, y + 8, 1100, 40, ["파란 값: 95 % 구간이 0(비율은 1)을 제외"], 24, MUTED)
    yb = picture(s, "poster_trace_ko.png", 1235, 330, 590)
    textbox(s, 1235, yb + 8, 590, 70, ["그림 4. 시험 방전 세 개의 보고된 Ti와 인과 추정 (음영: 값이 없는 행, 신경망: 첫 단계)"], 24, MUTED)
    page_number(s, 7)
    notes(s, NOTE["result"])

    # 5 information
    s = content_slide(prs, "5 정보를 나르는 입력")
    textbox(s, 91, 322, 1160, 40, ["표 3. 비교군 대비 skill (filter는 KF0 대비)"], 24, MUTED)
    y = table(s, 91, 366, 1160, [
        (["입력", "비교군 대비 skill"], "hd"),
        (["학습된 신경망에 BES만 한 프레임 늦게 (r2–r9)", "+0.000 [−0.024, +0.022]"], "hi"),
        (["입력을 바꿔 다시 학습한 신경망 (r2–r4)", ""], "grp"),
        (["빠른 채널 전부를 한 행 늦게", "−0.006 [−0.025, +0.016]"], ""),
        (["BES와 CES 이력만", ("+0.044 [+0.020, +0.066]", "y")], "hi"),
        (["ECEI와 CES 이력만", "−0.011 [−0.092, +0.022]"], ""),
        (["Mirnov 제거", ("+0.060 [+0.023, +0.085]", "y")], ""),
        (["BES만 읽는 신경망 대비 첫 단계 신경망", "+0.009 [−0.019, +0.031]"], ""),
        (["5-parameter filter: KF0 대비 KFX (BES 구동, r3–r9)", ("+0.049 [+0.040, +0.059]", "y")], "hi"),
    ], [0.64, 0.36])
    textbox(s, 91, y + 8, 1160, 40, ["파란 값: 95 % 구간이 0을 제외"], 24, MUTED)
    textbox(s, 1300, 330, 525, 45, ["주의"], 28, NAVY, bold=True)
    bullets(s, 1300, 380, 525, 260, ["BES: 가열 빔의 빛, 밀도와 빔에 의존", "파일에 빔 출력 없음: 빔 변화의 기여는 배제 못 함"], 26)
    page_number(s, 8)
    notes(s, NOTE["information"])

    # 6 where
    s = content_slide(prs, "6 이득의 위치와 fit 점검")
    yb = picture(s, "poster_blips_ko.png", 91, 330, 780)
    textbox(s, 91, yb + 10, 780, 70, ["그림 5. 빔 blip 뒤 사건 평균 (앞선 여섯 행 대비 비, 음영: 95 % 구간, 점선: 대조 행)"], 24, MUTED)
    textbox(s, 915, 322, 910, 40, ["표 4. 첫 단계, pooled (r2–r9)"], 24, MUTED)
    y = table(s, 915, 366, 910, [
        (["판독", "값"], "hd"),
        (["증분: 비교군 대비 신경망", ""], "grp"),
        (["빠진 프레임 뒤 1–3행", ("+0.151 [+0.093, +0.215]", "y")], "hi"),
        (["가장 차가운 3분의 1 (방전 중앙값 Ti)", "+0.007 [−0.023, +0.031]"], "hi"),
        (["중간 3분의 1", ("+0.042 [+0.009, +0.074]", "y")], ""),
        (["가장 뜨거운 3분의 1", ("+0.092 [+0.040, +0.137]", "y")], ""),
        (["이득이 가장 큰 방전 64개를 뺀 경우", "+0.027 [−0.004, +0.051]"], ""),
        (["fit 점검과 산포 바닥", ""], "grp"),
        (["ρ1: 다음 프레임에도 남은 보정", "0.810 [0.753, 0.866]"], ""),
        (["바닥 s, RMS (eV)", "135.7 [126.4, 145.7]"], ""),
        (["신경망: 바닥 대비 평균 제곱 오차", "1.331 [1.246, 1.427]"], ""),
    ], [0.60, 0.40])
    textbox(s, 915, y + 8, 910, 70, ["파란 값: 95 % 구간이 0을 제외 · ρ1: 한 프레임 뒤 보정 계수 ÷ 같은 프레임의 계수"], 24, MUTED)
    page_number(s, 9)
    notes(s, NOTE["where"])

    # 7 conclusion
    s = content_slide(prs, "7 결론")
    textbox(s, 91, 330, 1000, 45, ["결과 (3 keV 컷 모집단)"], 28, NAVY, bold=True)
    bullets(s, 91, 385, 1000, 520, [
        "비교군 대비 신경망의 증분: 구간이 0 위 (표 2)",
        "BES 구동 filter의 증분: 신경망과 같은 크기 (표 3)",
        "BES만 한 프레임 늦추면 증분 없음 (표 3)",
        "두 단계 estimator: offline GP 대비 skill의 구간이 0 위, hold 대비 중앙값 오차 비의 구간이 1 아래 (표 2)",
        "빠른 채널 보정의 대부분이 따로 fit되는 다음 프레임에도 남음 (표 4)"], 28)
    textbox(s, 1140, 330, 685, 45, ["단서 (신경망 혼자)"], 28, NAVY, bold=True)
    bullets(s, 1140, 385, 685, 520, [
        "이득이 보고값의 프레임 간 산포에 비해 작음",
        "방전 온도의 가장 차가운 3분의 1에서 확인되지 않음",
        "이득이 가장 큰 방전 10분의 1에 의존",
        "파일 단위 단일 분할에서 확인되지 않음"], 28)
    page_number(s, 10)
    notes(s, NOTE["conclusion"])

    # 8 limits
    s = content_slide(prs, "8 한계와 전망")
    items = [
        ("컷 모집단 한정: 3 keV 초과 값을 남기면 증분의 구간이 0을 포함 (표 5)", "→ 프레임별 fit 불확도"),
        ("측정선·캠페인·장치 하나, C VI 불순물 온도, 파일에 빔 출력 없음", "→ 빔 출력, 평형 매핑, 다른 측정선"),
        ("손대지 않은 hold-out 방전 없음", "→ 마지막 97개 방전으로 시간순 확증 (진행 중)"),
        ("KSTAR에서 BES·CES 실시간 사용 발표 없음, 지금 프레임 BES와 두 프레임 이내의 CES 값 필요", "→ beam-off 프레임이 필요 없는 CES 분석"),
    ]
    y = 330
    for k, (a, b) in enumerate(items, 1):
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, px(91), px(y + 2), px(44), px(44))
        dot.fill.solid()
        dot.fill.fore_color.rgb = NAVY
        dot.line.fill.background()
        fill_tf(dot.text_frame, [str(k)], 24, WHITE, bold=True, align=PP_ALIGN.CENTER, space=0, line=1.0)
        dot.text_frame.margin_left = dot.text_frame.margin_right = 0
        lines = 2 if len(a) > 34 else 1
        textbox(s, 150, y, 900, 40 * lines + 44, [a, [(b, TEAL, False)]], 26, BODY, space=2)
        y += 40 * lines + 44 + 24
    textbox(s, 1100, 322, 725, 40, ["표 5. 3 keV 초과 값을 학습과 채점에 남긴 경우 (r2–r4)"], 24, MUTED)
    y = table(s, 1100, 366, 725, [
        (["신경망의 대비 기준", "skill"], "hd"),
        (["인과 GP", ("+0.109 [+0.070, +0.146]", "y")], ""),
        (["비교군", "−0.002 [−0.017, +0.010]"], "hi"),
    ], [0.52, 0.48])
    textbox(s, 1100, y + 8, 725, 40, ["파란 값: 95 % 구간이 0을 제외"], 24, MUTED)
    textbox(s, 1100, y + 70, 725, 45, ["다음: 3 keV 초과 값의 영향"], 28, NAVY, bold=True)
    bullets(s, 1100, y + 120, 725, 260, ["손실을 지배하는 소수의 큰 값", "목표 정규화의 폭", "fit 실패와 빠른 채널의 결합", "인과 GP 이력의 큰 값"], 26)
    page_number(s, 11)
    notes(s, NOTE["limits"])

    # 9 rotation
    s = content_slide(prs, "보충: 회전 Vrot")
    textbox(s, 91, 322, 830, 40, ["표 6. Vrot 가지의 skill (pooled, 컷 모집단)"], 24, MUTED)
    y1 = table(s, 91, 366, 830, [
        (["대비 기준", "skill"], "hd"),
        (["hold", ("+0.265 [+0.198, +0.326]", "y")], ""),
        (["running mean", ("+0.121 [+0.074, +0.187]", "y")], ""),
        (["인과 GP (분할별로는 4개 중 1개)", ("+0.151 [+0.038, +0.290]", "y")], "hi"),
        (["PCHIP (다음 값 사용)", ("+0.092 [+0.0003, +0.170]", "y")], ""),
        (["offline GP (다음 값 사용)", "−0.073 [−0.162, +0.005]"], "hi"),
    ], [0.55, 0.45])
    textbox(s, 91, y1 + 8, 830, 40, ["파란 값: 95 % 구간이 0을 제외"], 24, MUTED)
    textbox(s, 965, 322, 860, 40, ["표 7. Vrot 자료와 빠른 채널"], 24, MUTED)
    y2 = table(s, 965, 366, 860, [
        (["측정", "값"], "hd"),
        (["빠른 채널 증분과 Vrot 증분의 상관 (Ti: 하한 +0.127)", "|r| ≤ 0.044"], "hi"),
        (["Vrot 결측 (격자 대비)", "23.5 %"], ""),
        (["관측 가운데 직전 값의 반복", "54 %"], ""),
        (["독립 측정 (격자 대비)", "35.2 %"], ""),
        (["빔 blip 뒤 첫 행의 회전 ÷ 직전 수준", "0.973 [0.960, 0.989]"], ""),
    ], [0.68, 0.32])
    bullets(s, 91, max(y1, y2) + 70, 1734, 120, ["Vrot 가지: CES 이력만 입력, 빠른 채널 미사용 → 빠른 채널 증분 판독 없음", "결과 위치: 원고 보충 자료"], 28)
    page_number(s, 12)
    notes(s, NOTE["vrot"])

    prs.save(str(OUT))
    print("written", OUT.name, len(prs.slides), "slides")


if __name__ == "__main__":
    build()
