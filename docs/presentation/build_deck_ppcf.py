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
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1] / "data" / "deck_out" / "중간 보고 2_final.pptx"  # the author's name for the final deck

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


def problem_slide(prs):
    """'1 문제': why nowcasting matters, from the verified prior work (numbers in the table only)."""
    s = content_slide(prs, "1 문제")
    textbox(s, 91, 322, 1734, 45, ["선행 연구"], 28, NAVY, bold=True)
    y = table(s, 91, 372, 1734, [
        (["출처", "내용", "시간 · 정확도"], "hd"),
        (["ITER CXRS 요구 (Mazon et al. 2025)", "제어용 중심 Ti 측정", "100 ms, 10 %"], ""),
        (["HL-3 신경망 CXRS (Tian et al. 2026)", "플라스마 제어에 필요한 응답 시간", "약 10 ms"], ""),
        (["JT-60U (Kobayashi et al. 2007)", "Ti 실시간 전송, NBI 되먹임", "10 ms마다"], ""),
        (["DIII-D 실시간 CER (Chrystal et al. 2026)", "기준 분석과 20 % 이내인 Ti 비율", "5 ms 노출, 중심 88 % · 가장자리 79 %"], ""),
        (["KSTAR NN-CES (Lee et al., FED 222)", "beam-off 프레임 대기, R ≳ 2.2 m에서 성능 저하, 실시간 CES는 목표 단계", "최대 1 s 지연"], "hi"),
    ], [0.30, 0.46, 0.24], size=22, rh=46)
    bullets(s, 91, y + 22, 1734, 200, [
        "제어: 약 10 ms 단위의 Ti 필요, 실시간 CX 분석은 가장자리에서 정확도 낮음",
        "KSTAR edge 측정선 (R = 2.198 m): 실시간 CES 없음, 결측과 빔 blip 잦음",
        "문제: 다음 CES 값 없이 지금 프레임의 Ti 추정 (nowcasting)",
        "대상: 같은 위치 BES의 지금 프레임 값이 CES 이력에 더하는 정보"], 26)
    notes(s, NOTE["problem"])
    return s


def estimators_right(s):
    """Right column of '3 estimator와 채점': the references, scoring in one line, and what (b) is."""
    y = table(s, 1035, 326, 790, [
        (["기준", "읽는 것"], "hd"),
        (["hold", "마지막 관측값"], ""),
        (["인과 GP", "CES 이력만 맞춘 GP (주 기준)"], ""),
        (["offline GP", "다음 값까지 사용, 실시간 불가"], ""),
        (["비교군", "빠른 채널만 뺀 같은 신경망"], "hi"),
    ], [0.26, 0.74], size=20, rh=38)
    textbox(s, 1035, y + 8, 790, 70, ["skill = 1 − MSE(estimator) / MSE(기준) · 채점: 감춘 측정 행, 방전 단위 교차 검증 (pooled), bootstrap 95 % 구간"], 20, MUTED)
    textbox(s, 1035, y + 88, 790, 40, ["(b) 5-parameter filter"], 24, NAVY, bold=True)
    bullets(s, 1035, y + 130, 790, 230, [
        "예측: log Ti의 직전 값에 BES 변화 두 가지를 더함 (c: 9채널 공통 성분, g: 7열 − 5열)",
        "갱신: CES 값이 오면 차이(innovation)가 클수록 덜 반영",
        "파라미터 5개: bc, bg, q, r, k",
        "KF0 (구동 없음)과 KFX (BES 구동) 비교: 같은 정보를 신경망 없이 시험 (표 3)"], 22)


NOTE = {}
NOTE["cover"] = """제목: Pedestal top에서의 빠른 진단으로 KSTAR edge 측정선의 전하 교환 이온 온도를 인과적으로 nowcasting하기.
nowcasting: 지금 프레임의 값을 그 프레임의 CES 값 없이 추정하는 것. 인과적: 지금 시각까지의 정보만 쓴다는 뜻."""
NOTE["background"] = """배경: CES는 탄소 C VI 이온의 온도 T_i와 회전 V_rot을 10 ms 프레임마다 보고합니다. 자료 제공자가 pedestal top으로 고른 edge 측정선에서는 값이 자주 빠집니다.
왼쪽 표: 격자 245,925행(첫·끝 행 제외) 기준으로 T_i는 7.7 %, V_rot은 23.5 %가 완전 결측입니다. V_rot은 관측의 54 %가 직전 값의 반복이라 격자의 41.3 %가 유지값이고, 독립 측정은 35.2 %뿐입니다. T_i의 유지값은 226,991개 중 1개입니다. T_i 결측은 BES 세기가 낮은 5분위에서 18.0 % [15.9, 20.0], 높은 5분위에서 3.4 % [2.7, 4.1]로 무작위가 아닙니다.
오른쪽 표: 실제 파일 #30909의 3.47–3.58 s. 주황 칸이 빈 값, 회색 칸이 직전 값의 반복입니다. T_i는 빈 칸이 섞여 있고, V_rot은 16.06이 계속 되풀이됩니다.
두 표적은 서로 독립적으로 빠지므로, 표적마다 따로 가린 손실(masked loss)로 학습합니다.
UFCES(KSTAR 초고속 CES, 100 kHz)는 요동 측정용 설계이고 실시간 분석은 다루지 않습니다.

용어 풀이. 유지값: 새 측정 없이 직전 값이 다시 적힌 값. 학습·입력·기준에서 모두 뺀다. BES(빔 방출 분광): 가열 빔이 내는 빛의 세기, 밀도와 빔에 의존."""
NOTE["data"] = """자료는 KSTAR 2022 캠페인의 방전 파일 641개, 10 ms 격자 245,925행입니다. 분할에 들어가는 방전은 639개입니다.
그림 1: 빨간 선이 CES 측정선 R = 2.198 m이고, 아래 막대가 BES 시야, BES 5–7열, ECEI 3열입니다. 빗금은 채널 이름에서 추정한 위치입니다.
그림 2: 빠른 채널 값은 그 행 시각의 원시 표본 하나이고, CES 값은 그 시각 근처에서 노출이 시작되는 프레임입니다. 행 간격이 20–40 ms이면 빔 blip이고, 그 프레임은 파일에 없습니다. 행 t의 인과 추정은 t까지의 빠른 표본과 t 이전의 CES 값만 씁니다.
3 keV 넘는 값은 fit 실패로 보고 빠진 값으로 다루고, 직전 값을 되풀이한 유지값은 뺍니다."""
NOTE["problem"] = """선행 연구로 본 nowcasting의 필요입니다.
제어 쪽 요구: HL-3의 신경망 CXRS 논문(Tian et al. 2026)은 플라스마 제어에 보통 필요한 응답 시간을 약 10 ms로 적습니다. ITER CXRS 요구(Mazon et al. 2025)는 제어용 중심 Ti에 100 ms, 10 %입니다.
실시간 분석 선례: JT-60U는 Ti를 10 ms마다 보내 NBI 되먹임에 썼고(Kobayashi et al. 2007), DIII-D의 실시간 CER은 5 ms 노출 뒤 약 1 ms에 결과를 내지만 기준 분석과 20 % 안에 드는 Ti 비율이 중심 88 %, 가장자리 79 %로 가장자리에서 낮습니다(Chrystal et al. 2026).
KSTAR: 지금의 CES 분석은 배경을 빼려고 beam-off 프레임을 기다려 최대 1 s 늦고, R ≳ 2.2 m에서 성능이 떨어지며, 실시간 CES는 목표로 제시된 단계입니다(Lee et al., FED 222). 이 연구의 측정선 R = 2.198 m은 그 경계에 있고 결측과 빔 blip이 잦습니다.
그래서 문제를 다음 CES 값 없이 지금 프레임의 Ti를 추정하는 것(nowcasting)으로 둡니다. 재는 것은 같은 위치 BES의 지금 프레임 값이 CES 이력에 더하는 정보입니다."""
NOTE["layout"] = """왼쪽: KFE 발표 자료(Current status of the KSTAR diagnostics, 2024)의 진단 배치. 빨간 표시가 이 연구가 읽는 진단입니다. 위에서 본 배치에서 NBI 1 빔과, 그 빔을 보는 Toroidal CES와 BES의 시선, ECEI가 있는 G 포트를 짚습니다. 단면 배치에서는 CES 측정점 R = 2.198 m, z = 0을 짚습니다.
맨 오른쪽: 받은 Mirnov 배치도. 토로이달 배열 20개 코일 가운데 파일에 들어 있는 두 코일 MC1T03, MC1T16을 이름의 번호로 표시했습니다. Mirnov 코일은 벽에 있습니다."""
NOTE["positions"] = """가운데: 받은 ABCD 위치 그림(syndia 보정 전)을 우리 채널 구역만 잘라 파일의 채널을 표시했습니다. 빨간 원이 BES 9채널(105–107, 205–207, 305–307), 파란 원이 ECEI GT1103–GT1403입니다.
굵은 점선이 CES 측정선 ch18, R = 2.198 m이고, 가는 점선이 이웃 측정선 2.192 m, 2.203 m입니다. BES와 ECEI 채널은 측정선에서 1–2 cm 안에 있습니다.
아래 표가 자료의 요약입니다."""
NOTE["blip"] = """빔 blip 한 건의 예입니다. KSTAR #31923, 5.00 s.
왼쪽 위: 원시 BES(500 kHz, 파일의 9채널 평균). 가열 빔이 약 11 ms 꺼지는 동안 BES가 거의 0으로 떨어집니다(회색). 검은 점은 10 ms 파일에 들어 있는 BES 값입니다.
왼쪽 가운데: 10 ms 파일의 CES Ti. 4.992 s 다음 행이 5.012 s로, 그 사이 프레임의 값이 없습니다(주황). 빔이 꺼지면 전하 교환 빛이 없어서 그 프레임은 분석되지 않습니다. 직후 Ti는 약 414 eV에서 187 eV로 낮아졌다가 수십 ms에 걸쳐 회복합니다.
왼쪽 아래: 원시 ECEI(4채널 평균). 빔과 무관한 신호인데도 blip 직후부터 낮아지고 천천히 회복합니다. 즉 Ti의 감소는 빔 세기가 아니라 플라스마 쪽 변화를 따릅니다(6절 그림 5의 사건 평균과 같은 모습).
오른쪽: 같은 방전을 다룬 Heo et al. (2025) 그림 2. (b)의 PNBI가 약 3.5 s와 5.0 s에 계단처럼 떨어지는 것이 빔 blip이고, (e)의 BES_0206 스펙트로그램에서 같은 시각에 흰 세로줄이 보입니다. BES_0206은 이 연구가 읽는 채널 가운데 하나입니다."""
NOTE["discharges"] = """방전 639개를 한 점씩 그렸습니다. 왼쪽부터 방전 번호에 따른 Ti 중앙값, Vrot 중앙값과 Ti 중앙값, BES 수준과 Ti 결측 비율입니다.
색은 BES 버스트 빈도(초당), 회색은 파일이 짧아 빈도를 정할 수 없는 방전입니다. 빈 네모는 캠페인 마지막 97개 방전으로, 시간순 확증에서 시험 블록으로 씁니다.
Ti 중앙값은 3 keV 넘는 값을 뺀 뒤, Vrot 중앙값은 직전 값의 반복(유지값)을 뺀 뒤의 값입니다. 가운데 패널은 −100–250 km/s 밖의 방전을 생략했습니다(개수는 그림에 표시).
BES 수준이 낮은 방전일수록 Ti 결측이 많습니다."""
NOTE["estimators"] = """그림 3 (a): 첫 단계는 격자의 모든 행을 차례로 읽는 LSTM으로, 지금 행의 빠른 채널과 CES 이력(마지막 값, 경과 시간, 관측 여부)을 읽습니다. 둘째 단계는 첫 단계의 오차를 gradient boosting으로 추정해 더하며, 학습 방전에서 만든 fold 밖 오차로만 맞춥니다.
1단계(LSTM)는 CES 이력만 쓰는 방법들(hold, 인과 GP)이 크게 틀리는 행에서 오차를 줄였다. 그 대신 보통 행의 오차는 줄이지 못했다. 2단계(트리)는 그 행에서 알 수 있는 입력으로 1단계가 남긴 오차를 예측해 빼 준다. 그 결과 평균 제곱 오차와 중앙값 오차가 함께 내려갔다.
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
    fill_tf(t.text_frame, ["Pedestal top에서의 빠른 진단으로 KSTAR edge 측정선의", "전하 교환 이온 온도를 인과적으로 nowcasting하기"],
            54, RGBColor(0x33, 0x33, 0x33), bold=True, align=PP_ALIGN.RIGHT, space=0, line=1.15)
    textbox(s, 130, 840, 1660, 110, ["이승상 · 서울대학교 원자핵공학과", "KSTAR 2022 캠페인 자료 (KSTAR team 제공)"],
            30, WHITE, align=PP_ALIGN.CENTER, space=6)
    notes(s, NOTE["cover"])

    # 1 background and problem
    s = content_slide(prs, "1 배경")
    textbox(s, 91, 322, 1000, 45, ["배경: 결측 실태 (shot 641개, 245,925행)"], 28, NAVY, bold=True)
    y = table(s, 91, 372, 1000, [
        (["항목", "Ti", "Vrot"], "hd"),
        (["완전 결측 (NaN)", "7.7 %", "23.5 %"], ""),
        (["직전 값의 반복 (유지값)", "1개", "41.3 %"], ""),
        (["독립 측정", "92.3 %", "35.2 %"], "hi"),
    ], [0.48, 0.30, 0.22], rh=44)
    bullets(s, 91, y + 60, 1000, 60, ["Ti와 Vrot이 서로 독립적으로 결측 → 표적별로 따로 학습"], 26)
    bes_name, ex = csv_excerpt(HERE.parents[1] / "data" / "s30909.csv", 3.472, 3.582)
    textbox(s, 1130, 322, 695, 45, ["원본 파일 예: #30909, 3.47–3.58 s"], 24, MUTED)
    ye = table(s, 1130, 366, 695, [(["time (s)", "CES_TI", "CES_VT", bes_name], "hd")] + ex,
               [0.22, 0.26, 0.24, 0.28], size=22, rh=38)
    textbox(s, 1130, ye + 6, 695, 40, ["주황: 빈 값 · 회색: 직전 값의 반복"], 24, MUTED)
    page_number(s, 2)
    notes(s, NOTE["background"])

    # 1 problem
    s = problem_slide(prs)
    page_number(s, 3)

    # 2 diagnostics and data, one slide
    s = content_slide(prs, "2 진단과 자료")
    yb = picture(s, DECK_IN / "kfe_layout_p15.png", 91, 326, 860)
    textbox(s, 91, yb + 4, 860, 40, ["KSTAR 진단 배치 (J. Ko, KFE, 2024, 15쪽). 빨강: 이 연구가 읽는 진단"], 22, MUTED)
    ya = picture(s, DECK_IN / "abcd_marked.png", 975, 326, 520)
    textbox(s, 975, ya + 4, 520, 80, ["채널 위치 (제공 ABCD 그림, syndia 보정 전). 원: BES 9채널, ECEI 4채널. 굵은 점선: CES ch18"], 22, MUTED)
    ym = picture(s, DECK_IN / "mirnov_marked.png", 1515, 326, 310)
    textbox(s, 1515, ym + 4, 310, 80, ["Mirnov 토로이달 배열 (제공 그림)"], 22, MUTED)
    table(s, 91, 782, 1734, [
        (["방전 파일", "격자", "CES 측정선", "BES", "ECEI", "Mirnov", "제외"], "hd"),
        (["641개 (분할 639개)", "10 ms, 245,925행", "ch18, R = 2.198 m", "9채널 (105–307)", "GT1103–GT1403", "MC1T03, MC1T16", "3 keV 초과, 유지값"], ""),
    ], [0.165, 0.15, 0.15, 0.13, 0.14, 0.135, 0.13], size=22, rh=44)
    page_number(s, 4)
    notes(s, NOTE["layout"] + chr(10) * 2 + NOTE["positions"])

    # 3 estimators
    s = content_slide(prs, "3 estimator와 채점")
    yb = picture(s, "poster_schematic_ko.png", 91, 330, 900)
    textbox(s, 91, yb + 10, 900, 40, ["그림 3. (a) 두 단계 estimator. (b) 5-parameter filter (KF0: 구동 없음, KFX: BES 구동)."], 24, MUTED)
    estimators_right(s)
    page_number(s, 5)
    notes(s, NOTE["estimators"])

    # 4 result
    s = content_slide(prs, "4 결과")
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
    page_number(s, 6)
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
    page_number(s, 7)
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
    page_number(s, 8)
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
    page_number(s, 9)
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
    page_number(s, 10)
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
    page_number(s, 11)
    notes(s, NOTE["vrot"])

    # appendix: one beam blip
    s = content_slide(prs, "보충: 빔 blip 예 (#31923, 5.00 s)")
    yb = picture(s, HERE / "figures" / "deck_ppcf" / "blip_31923_ko.png", 91, 326, 980)
    textbox(s, 91, yb + 6, 980, 70, ["우리 자료: 원시 BES(500 kHz)가 11 ms 동안 꺼지고(회색), 그 프레임의 CES 값이 없음(주황). 직후 Ti와 ECEI가 함께 낮아졌다가 회복"], 22, MUTED)
    yh = picture(s, DECK_IN / "heo2025_fig2.png", 1120, 326, 420)
    textbox(s, 1120, yh + 6, 530, 130, ["Heo et al., Nucl. Fusion 65, 036003 (2025), 그림 2 (같은 방전 #31923). (b) PNBI가 약 3.5 s와 5.0 s에 떨어짐, (d) Ti,edge, (e) BES_0206 스펙트로그램의 흰 세로줄"], 22, MUTED)
    page_number(s, 12)
    notes(s, NOTE["blip"])

    # appendix: discharges
    s = content_slide(prs, "보충: 방전 분포")
    yb = picture(s, HERE / "figures" / "deck_ppcf" / "dataset_overview_ko.png", 91, 330, 1734)
    textbox(s, 91, yb + 8, 1734, 70, ["방전마다 한 점. 색: BES 버스트 빈도, 빈 네모: 캠페인 마지막 97개 방전(시간순 시험 블록). Ti 중앙값은 3 keV 컷 뒤, Vrot 중앙값은 유지값을 뺀 뒤의 값"], 22, MUTED)
    page_number(s, 13)
    notes(s, NOTE["discharges"])

    prs.save(str(OUT))
    print("written", OUT.name, len(prs.slides), "slides")


if __name__ == "__main__":
    build()
