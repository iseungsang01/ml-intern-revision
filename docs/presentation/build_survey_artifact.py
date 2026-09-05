# -*- coding: utf-8 -*-
"""Build docs/presentation/survey_artifact.html - the published reference page.

One tab per table, and every cell comes from appendix_content.py, so the appendix
slides and this page cannot disagree. After editing the content, re-run this and
republish the artifact to the same URL.

    py docs/presentation/build_survey_artifact.py
"""
import html
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import appendix_content as AP  # noqa: E402

OUT = os.path.join(HERE, "survey_artifact.html")

E = lambda t: html.escape(t, quote=False)

# ---------------------------------------------------------------- row states
TRIED_META = [
    # (group, state, chip class, chip text) in TRIED_WINDOW + SEQ + MISC order
    ("win", "closed", "hold", "대조군으로 강등"),
    ("win", "closed", "closed", "닫힘"),
    ("win", "closed", "closed", "닫힘"),
    ("win", "closed", "closed", "닫힘"),
    ("win", "closed", "hold", "조건부"),
    ("win", "closed", "closed", "비채택"),
    ("seq", "closed", "hold", "부분 성공"),
    ("seq", "adopted", "adopt", "채택 · 백본"),
    ("seq", "closed", "closed", "미승격"),
    ("seq", "adopted", "adopt", "조건부 채택"),
    ("seq", "closed", "closed", "축 닫힘"),
    ("seq", "closed", "closed", "바닥 측정"),
    ("ops", "closed", "closed", "동률"),
    ("ops", "closed", "adopt", "우세"),
    ("ops", "closed", "closed", "비채택"),
    ("ops", "closed", "closed", "축 닫힘"),
    ("ops", "closed", "hold", "상한 설정"),
    ("ops", "closed", "adopt", "사후 보정 채택"),
    ("ops", "closed", "closed", "종결"),
    ("ops", "open", "open", "진행 중"),
]
GROUP_LABEL = {"win": "윈도", "seq": "시퀀스", "ops": "연산자·문맥"}
FUSION_STATE = ["closed", "open", "closed", "closed", "closed", "open",
                "closed", "closed", "closed", "open", "closed", "closed"]
GENERAL_STATE = ["closed", "closed", "closed", "open", "closed", "closed",
                 "open", "open", "open", "open"]
ISO_STATE = ["closed", "closed", "closed", "closed", "closed", "open"]


def tried_rows():
    rows = AP.TRIED_WINDOW + AP.TRIED_SEQ + AP.TRIED_MISC
    out = []
    for meta, r in zip(TRIED_META, rows):
        grp, state, chip_cls, chip_txt = meta
        out.append('        <tr data-state="%s" data-group="%s">' % (state, grp))
        out.append('          <td class="grp">%s</td>' % E(GROUP_LABEL[grp]))
        out.append('          <td class="name">%s<br><span class="chip chip--%s">%s</span></td>'
                   % (E(r[0]), chip_cls, E(chip_txt)))
        out.append('          <td class="year">%s</td>' % E(r[1]))
        for cell in r[2:]:
            out.append('          <td>%s</td>' % E(cell))
        out.append('        </tr>')
    return "\n".join(out)


def simple_rows(rows, states, device_col=None, year_col=None):
    out = []
    for st, r in zip(states, rows):
        out.append('        <tr data-state="%s">' % st)
        for i, cell in enumerate(r):
            cls = ""
            if i == 0:
                cls = ' class="name"'
            elif i == device_col:
                cls = ' class="device"'
            elif i == year_col:
                cls = ' class="year"'
            out.append('          <td%s>%s</td>' % (cls, E(cell)))
        out.append('        </tr>')
    return "\n".join(out)


def priority_rows():
    out = []
    for r in AP.PRIORITY_ROWS:
        cls = ' class="is-vrot"' if r[1].startswith("V_rot") else ""
        out.append('        <tr data-state="open"%s>' % cls)
        out.append('          <td class="rank">%s</td>' % E(r[0]))
        for cell in r[1:]:
            out.append('          <td>%s</td>' % E(cell))
        out.append('        </tr>')
    return "\n".join(out)


def th(items):
    return "".join('<th style="width:%s">%s</th>' % (w, E(t)) for t, w in items)


SRC_GROUPS = [("핵융합", AP.SURVEY_SOURCES[:16]),
              ("시계열", AP.SURVEY_SOURCES[16:31]),
              ("동형", AP.SURVEY_SOURCES[31:])]


def source_lists():
    out = []
    for label, items in SRC_GROUPS:
        out.append('      <div class="srcgroup">')
        out.append('        <h3>%s <span>%d</span></h3>' % (E(label), len(items)))
        out.append('        <ul>')
        for name, url in items:
            out.append('          <li><a href="%s" target="_blank" rel="noopener">%s</a></li>'
                       % (url, E(name)))
        out.append('        </ul>')
        out.append('      </div>')
    return "\n".join(out)


PAGE = """<title>CES 모델 계보와 문헌 지도</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">

<style>
:root{
  --ground:#F4F6FA; --surface:#FFFFFF; --surface-2:#EBF0F6;
  --ink:#132133; --ink-soft:#4A5A70; --ink-faint:#7A889C;
  --rule:#DCE3EC;
  --accent:#1F5FA8; --accent-soft:rgba(31,95,168,.10);
  --teal:#0E7A6B; --amber:#B05F14; --amber-soft:rgba(176,95,20,.12);
  --crimson:#98302B; --crimson-soft:rgba(152,48,43,.10);
  --ok-soft:rgba(14,122,107,.10);
  --head-bg:#132133; --head-ink:#FFFFFF; --head-dim:#9CACC4; --head-bar:#0C1826;
  --shadow:0 1px 2px rgba(19,33,51,.05), 0 6px 18px rgba(19,33,51,.05);
  --sans:'IBM Plex Sans KR','Malgun Gothic','맑은 고딕',system-ui,sans-serif;
  --mono:'IBM Plex Mono',Consolas,ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#101722; --surface:#18212E; --surface-2:#1F2A3A;
    --ink:#E4EAF3; --ink-soft:#9DAABF; --ink-faint:#73819A;
    --rule:#2A3547;
    --accent:#6BA3E8; --accent-soft:rgba(107,163,232,.14);
    --teal:#45BFAC; --amber:#DE9A54; --amber-soft:rgba(222,154,84,.14);
    --crimson:#E0837C; --crimson-soft:rgba(224,131,124,.13);
    --ok-soft:rgba(69,191,172,.12);
    --head-bg:#0A121C; --head-ink:#E4EAF3; --head-dim:#78879F; --head-bar:#070D15;
    --shadow:0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  --ground:#101722; --surface:#18212E; --surface-2:#1F2A3A;
  --ink:#E4EAF3; --ink-soft:#9DAABF; --ink-faint:#73819A;
  --rule:#2A3547;
  --accent:#6BA3E8; --accent-soft:rgba(107,163,232,.14);
  --teal:#45BFAC; --amber:#DE9A54; --amber-soft:rgba(222,154,84,.14);
  --crimson:#E0837C; --crimson-soft:rgba(224,131,124,.13);
  --ok-soft:rgba(69,191,172,.12);
  --head-bg:#0A121C; --head-ink:#E4EAF3; --head-dim:#78879F; --head-bar:#070D15;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.25);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
h1,h2,h3{margin:0;text-wrap:balance}
a{color:var(--accent)}
code,.mono{font-family:var(--mono)}

/* ---------- header + nav ---------- */
header{background:var(--head-bg);color:var(--head-ink);padding:20px 26px 16px}
.kick{font-family:var(--mono);font-size:11.5px;font-weight:500;letter-spacing:.14em;
  color:var(--teal);text-transform:uppercase}
header h1{margin:6px 0 4px;font-size:26px;font-weight:700;letter-spacing:-.02em}
header p{margin:0;color:var(--head-dim);font-size:13.5px;max-width:80ch}
nav{display:flex;background:var(--head-bar);padding:0 18px;overflow-x:auto}
.tab{flex:none;background:none;border:0;border-bottom:3px solid transparent;
  color:var(--head-dim);font-family:var(--sans);font-size:14px;font-weight:600;
  padding:12px 17px;cursor:pointer;white-space:nowrap;transition:color .12s,border-color .12s}
.tab:hover{color:var(--head-ink)}
.tab[aria-selected="true"]{color:var(--head-ink);border-bottom-color:var(--teal)}
.tab:focus-visible{outline:2px solid var(--teal);outline-offset:-4px}
.tab b{font-family:var(--mono);font-size:11px;font-weight:500;margin-left:7px;
  color:var(--head-dim);font-variant-numeric:tabular-nums}
.tab[aria-selected="true"] b{color:var(--teal)}

/* ---------- panes ---------- */
.pane{padding:20px 26px 30px;max-width:1560px;margin:0 auto}
.pane[hidden]{display:none}
.lede{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;margin-bottom:6px}
.lede h2{font-size:19px;font-weight:700;letter-spacing:-.01em}
.lede p{margin:6px 0 0;font-size:13.5px;color:var(--ink-soft);max-width:88ch}
.lede .n{font-family:var(--mono);font-size:11.5px;color:var(--ink-faint);
  font-variant-numeric:tabular-nums;white-space:nowrap;padding-top:5px}

/* ---------- fact strip ---------- */
.factstrip{display:flex;flex-wrap:wrap;margin:16px 0 4px;border:1px solid var(--rule);
  border-radius:9px;background:var(--surface);overflow:hidden;box-shadow:var(--shadow)}
.fact{flex:1 1 150px;padding:11px 15px;border-right:1px solid var(--rule)}
.fact:last-child{border-right:0}
.fact dt{font-family:var(--mono);font-size:10px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-faint);margin:0}
.fact dd{margin:3px 0 0;font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
.fact dd small{font-weight:400;color:var(--ink-faint);font-size:12.5px}

/* ---------- controls ---------- */
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0 0}
#q{flex:1 1 250px;min-width:180px;padding:8px 12px;font:inherit;font-size:13.5px;
  color:var(--ink);background:var(--surface);border:1px solid var(--rule);border-radius:6px}
#q::placeholder{color:var(--ink-faint)}
#q:focus-visible,.pill:focus-visible,.themebtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.pill{font:inherit;font-size:13px;font-weight:500;padding:7px 13px;border-radius:6px;cursor:pointer;
  border:1px solid var(--rule);background:var(--surface);color:var(--ink-soft)}
.pill[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.count{font-family:var(--mono);font-size:12px;color:var(--ink-faint);margin-left:auto;
  font-variant-numeric:tabular-nums}
.themebtn{font:inherit;font-size:12.5px;padding:7px 11px;border-radius:6px;cursor:pointer;
  border:1px solid var(--rule);background:var(--surface);color:var(--ink-soft)}
.hint{margin:10px 0 0;font-size:12.5px;color:var(--ink-faint)}

/* ---------- tables ---------- */
.scroller{overflow-x:auto;border:1px solid var(--rule);border-radius:9px;
  background:var(--surface);margin-top:14px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%}
table.t-tried{min-width:1180px}
table.t-fusion{min-width:1320px}
table.t-general{min-width:1140px}
table.t-iso{min-width:1040px}
table.t-next{min-width:1080px}
thead th{position:sticky;top:0;z-index:5;background:var(--head-bg);color:var(--head-ink);
  text-align:left;font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.05em;
  padding:10px 12px;white-space:nowrap}
tbody td{padding:11px 12px;vertical-align:top;font-size:13.5px;color:var(--ink-soft);
  border-top:1px solid var(--rule)}
tbody tr:nth-child(even) td{background:var(--surface-2)}
tbody td.name{color:var(--ink);font-weight:600}
tbody td.grp{color:var(--ink-faint);font-weight:500;font-size:12.5px;white-space:nowrap;
  font-family:var(--mono)}
td.device{color:var(--teal);font-weight:600;white-space:nowrap;font-size:13px}
td.year{font-family:var(--mono);font-size:12.5px;color:var(--ink-faint);
  white-space:nowrap;font-variant-numeric:tabular-nums}
.chip{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.05em;
  padding:2px 7px;border-radius:4px;white-space:nowrap;margin-top:6px;font-weight:500}
.chip--closed{background:var(--crimson-soft);color:var(--crimson)}
.chip--adopt{background:var(--ok-soft);color:var(--teal)}
.chip--open{background:var(--amber-soft);color:var(--amber)}
.chip--hold{background:var(--surface-2);color:var(--ink-faint)}
tr.is-vrot td{box-shadow:inset 3px 0 0 var(--amber)}
td.rank{font-family:var(--mono);font-size:19px;font-weight:600;color:var(--accent);
  font-variant-numeric:tabular-nums}
tr.is-vrot td.rank{color:var(--amber)}

/* ---------- cards ---------- */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;margin-top:16px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:9px;
  padding:15px 17px;box-shadow:var(--shadow)}
.card.stop{border-left:3px solid var(--crimson)}
.card.open{border-left:3px solid var(--amber)}
.card h3{font-size:14.5px;font-weight:600;margin-bottom:9px}
.card ul{margin:0;padding-left:17px;display:grid;gap:7px}
.card li{font-size:13.5px;color:var(--ink-soft)}
.card li b{color:var(--ink);font-weight:600}
.takeaway{margin:16px 0 0;padding:13px 15px;border-radius:9px;background:var(--accent-soft);
  border-left:3px solid var(--accent);font-size:13.5px;color:var(--ink)}

/* ---------- three theses ---------- */
.thesis{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:18px}
.thesis > div{padding-left:13px;border-left:3px solid var(--accent)}
.thesis > div.open{border-left-color:var(--amber)}
.thesis h3{font-size:14px;font-weight:600;margin-bottom:4px}
.thesis p{margin:0;font-size:13.5px;color:var(--ink-soft)}

/* ---------- sources ---------- */
.srcwrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:16px}
.srcgroup{background:var(--surface);border:1px solid var(--rule);border-radius:9px;
  padding:15px 17px;box-shadow:var(--shadow)}
.srcgroup h3{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-faint);font-weight:500;margin-bottom:10px}
.srcgroup h3 span{color:var(--teal)}
.srcgroup ul{margin:0;padding:0;list-style:none;display:grid;gap:7px}
.srcgroup li{font-size:13px}
.srcgroup a{text-decoration:none;border-bottom:1px solid var(--rule)}
.srcgroup a:hover{border-bottom-color:var(--accent)}

.empty{padding:26px 0;color:var(--ink-faint);font-size:13.5px}
footer{padding:18px 26px 34px;font-size:12.5px;color:var(--ink-faint);
  border-top:1px solid var(--rule)}
footer p{margin:0 0 6px;max-width:90ch}
@media (max-width:820px){.pane{padding:16px 14px 26px}header{padding:16px 16px 14px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<header>
  <div class="kick">KSTAR CES 나우캐스팅 · 조사일 2026-09-05</div>
  <h1>CES 모델 계보와 문헌 지도</h1>
  <p>여덟 달의 통제 실험이 남긴 판정 20건과, 같은 문제를 다루는 분야들이 지금 무엇을 쓰는지를 판별로 나누어 보관한다.
     각 판은 표 하나이며, 문장과 수치는 발표자료 부록 A와 같은 파일에서 온다.</p>
</header>

<nav role="tablist" aria-label="자료 판">
  <button class="tab" role="tab" id="tab-tried" aria-controls="pane-tried" aria-selected="true">시도한 모델<b>20</b></button>
  <button class="tab" role="tab" id="tab-fusion" aria-controls="pane-fusion" aria-selected="false">핵융합 문헌<b>12</b></button>
  <button class="tab" role="tab" id="tab-general" aria-controls="pane-general" aria-selected="false">시계열 · 센서<b>10</b></button>
  <button class="tab" role="tab" id="tab-iso" aria-controls="pane-iso" aria-selected="false">동형 분야<b>6</b></button>
  <button class="tab" role="tab" id="tab-next" aria-controls="pane-next" aria-selected="false">다음 팔<b>6</b></button>
  <button class="tab" role="tab" id="tab-src" aria-controls="pane-src" aria-selected="false">출처<b>36</b></button>
</nav>

<div class="pane" id="pane-tried" role="tabpanel" aria-labelledby="tab-tried">
  <div class="lede">
    <div>
      <h2>시도한 모델과 닫은 이유</h2>
      <p>각 행은 통제 변수가 하나이고, 마지막 열이 그것을 닫은 근거와 절 번호이다.
         음성 결과는 그것을 뒤집을 측정을 함께 지목할 때만 결론으로 인정한다(§8j).</p>
    </div>
    <div class="n">THESIS_RESULTS.md §8 · 20건</div>
  </div>
  <dl class="factstrip">
    <div class="fact"><dt>방전</dt><dd>641 <small>shot 30801–32751</small></dd></div>
    <div class="fact"><dt>격자</dt><dd>10 ms <small>공통 정렬</small></dd></div>
    <div class="fact"><dt>확정 프로토콜</dt><dd>W = 2 <small>held-free · 두 모집단</small></dd></div>
    <div class="fact"><dt>백본</dt><dd>seq_v2 <small>357,570 파라미터</small></dd></div>
    <div class="fact"><dt>조사</dt><dd>36건 <small>핵융합 16 · 시계열 15 · 동형 5</small></dd></div>
  </dl>
  <div class="bar">
    <button class="pill" data-group="all" aria-pressed="true">전체 20</button>
    <button class="pill" data-group="win" aria-pressed="false">윈도 계열 6</button>
    <button class="pill" data-group="seq" aria-pressed="false">시퀀스 계열 6</button>
    <button class="pill" data-group="ops" aria-pressed="false">연산자 · 문맥 · 기준선 8</button>
  </div>
  <p class="hint">윈도 계열의 수치는 W = 4 시대의 잠정값이고, 시퀀스 계열부터가 확정 프로토콜이다.</p>
  <div class="scroller">
    <table class="t-tried">
      <thead><tr>{TRIED_TH}</tr></thead>
      <tbody>
{TRIED_ROWS}
      </tbody>
    </table>
  </div>
  <p class="takeaway">{TAKEAWAY}</p>
</div>

<div class="pane" id="pane-fusion" role="tabpanel" aria-labelledby="tab-fusion" hidden>
  <div class="lede">
    <div>
      <h2>핵융합 분야 최신 (2024 ~ 2026)</h2>
      <p>장치와 연도를 따로 두었다. 진단으로 다른 진단을 추정하는 계열은 활발하지만,
         우리 문제와 가장 가까운 두 편이 가장 단순한 모델을 쓴다는 것이 이 판의 요점이다.</p>
    </div>
    <div class="n">12편</div>
  </div>
  <div class="thesis">
    <div><h3>주류는 아직 기억 없는 구조다</h3>
      <p>Diag2Diag와 COMPASS 모두 시간 이력이 없는 MLP이며, seq_v2는 그 위 단계인 인과 상태추정이다.</p></div>
    <div><h3>사전학습은 프로파일에 안 듣는다</h3>
      <p>TokaMind의 미세조정 이득은 전체 +0.017이지만 프로파일 그룹은 −0.005로 음수이다.</p></div>
    <div class="open"><h3>회전 선례는 입력을 가리킨다</h3>
      <p>EAST XCS는 회전을 실제로 추론하지만 입력이 도플러 분광이다. 우리에게 필요한 것도 입력이다.</p></div>
  </div>
  <div class="scroller">
    <table class="t-fusion">
      <thead><tr>{FUSION_TH}</tr></thead>
      <tbody>
{FUSION_ROWS}
      </tbody>
    </table>
  </div>
</div>

<div class="pane" id="pane-general" role="tabpanel" aria-labelledby="tab-general" hidden>
  <div class="lede">
    <div>
      <h2>일반 시계열 · 센서 예측의 주류와 판정</h2>
      <p>마지막 열은 문헌의 주장이 아니라 이 저장소의 통제 실험 판정이다.
         새 계열을 도입하기 전에 이 열을 먼저 읽는다.</p>
    </div>
    <div class="n">10개 방법군</div>
  </div>
  <div class="scroller">
    <table class="t-general">
      <thead><tr>{GENERAL_TH}</tr></thead>
      <tbody>
{GENERAL_ROWS}
      </tbody>
    </table>
  </div>
</div>

<div class="pane" id="pane-iso" role="tabpanel" aria-labelledby="tab-iso" hidden>
  <div class="lede">
    <div>
      <h2>구조적 동형 분야와 가져올 것</h2>
      <p>문제의 골격이 같은 분야들이다. 서로 다른 어휘로 같은 결론에 도달한다.
         표현력이 아니라 개체별 보정과 상태추정 프레임이 값을 한다는 것이다.</p>
    </div>
    <div class="n">6개 계열</div>
  </div>
  <div class="scroller">
    <table class="t-iso">
      <thead><tr>{ISO_TH}</tr></thead>
      <tbody>
{ISO_ROWS}
      </tbody>
    </table>
  </div>
</div>

<div class="pane" id="pane-next" role="tabpanel" aria-labelledby="tab-next" hidden>
  <div class="lede">
    <div>
      <h2>다음 팔의 우선순위</h2>
      <p>비용 대비 정보량 순이다. 각 행은 통제 변수가 하나이며,
         TEST를 여는 팔은 사전등록 뒤에만 실행한다.</p>
    </div>
    <div class="n">6개 팔</div>
  </div>
  <div class="scroller">
    <table class="t-next">
      <thead><tr>{PRIORITY_TH}</tr></thead>
      <tbody>
{PRIORITY_ROWS}
      </tbody>
    </table>
  </div>
  <div class="cards">
    <div class="card stop">
      <h3>권하지 않는 방향</h3>
      <ul>{NOT_REC}</ul>
    </div>
    <div class="card open">
      <h3>회전은 열린 과제이다</h3>
      <ul>
        <li>§8ar의 항별 감사는 <b>현재 입력이 닿는 항이 없음</b>을 보였을 뿐, 회전이 예측 불가능하다고 말하지 않는다.</li>
        <li>뒤집을 측정이 함께 지목되어 있다. 원시 kHz Mirnov(B.6, shot 집합 동결), NBI 토크 채널, CES 피팅 품질 메타데이터, 도플러 분광 입력이다.</li>
        <li>지금도 이기는 구간이 있다. Δt &gt; 15 ms와 peak 층에서 <b>+0.54 ~ +0.79</b>이며, 승패를 가르는 공변량은 방전 안에서 회전이 얼마나 움직이는가 하나이다(조용 34 % → 변동 55 %).</li>
        <li>회전의 자체 이완 시간은 아직 측정 불가이다. 유지값이 관측의 54 %를 차지해 16 ms와 300 ms 사이에서 정해지지 않는다. 이것부터 데이터로 푼다.</li>
      </ul>
    </div>
  </div>
</div>

<div class="pane" id="pane-src" role="tabpanel" aria-labelledby="tab-src" hidden>
  <div class="lede">
    <div>
      <h2>출처</h2>
      <p>2026-09-05에 원문을 확인한 36건이다. 분야별로 나누어 두었다.</p>
    </div>
    <div class="n">36건</div>
  </div>
  <div class="srcwrap">
{SOURCES}
  </div>
</div>

<div class="pane" id="pane-search" hidden>
  <p class="empty" id="empty">검색어와 맞는 행이 없다.</p>
</div>

<div class="bar" id="searchbar" style="padding:0 26px 18px;max-width:1560px;margin:0 auto">
  <input id="q" type="search" placeholder="이 판 안에서 찾기 — 모델 · 장치 · 연도 · 절 번호 (예: KSTAR, 2026, §8ag, Huber)" aria-label="현재 판 안에서 검색">
  <button class="pill" data-state="open" aria-pressed="false" id="onlyopen">열린 것만</button>
  <span class="count" id="count"></span>
  <button class="themebtn" id="theme" type="button">테마</button>
</div>

<footer>
  <p>판정 수치는 <span class="mono">THESIS_RESULTS.md §8</span>과 동결된 평가 산출물에서 옮겼고,
     문헌 요약은 2026-09-05에 원문을 확인한 것이다. 같은 표가 발표자료 부록 A에도 있으며
     두 곳 모두 <span class="mono">docs/presentation/appendix_content.py</span>를 읽는다.</p>
  <p>W = 4 시대의 수치는 잠정으로 표시했으며 확정 프로토콜의 주장에는 쓰지 않는다.</p>
</footer>

<script>
(function(){
  "use strict";
  var tabs = [].slice.call(document.querySelectorAll(".tab"));
  var panes = {};
  tabs.forEach(function(t){ panes[t.id] = document.getElementById(t.getAttribute("aria-controls")); });

  var q = document.getElementById("q");
  var countEl = document.getElementById("count");
  var emptyEl = document.getElementById("empty");
  var searchPane = document.getElementById("pane-search");
  var searchBar = document.getElementById("searchbar");
  var onlyOpen = document.getElementById("onlyopen");
  var groupPills = [].slice.call(document.querySelectorAll(".pill[data-group]"));

  var active = "tab-tried", group = "all", openOnly = false;

  var rows = [].slice.call(document.querySelectorAll("tbody tr"));
  rows.forEach(function(r){ r.dataset.hay = (r.textContent||"").toLowerCase().replace(/\\s+/g," "); });

  function store(k,v){ try{ localStorage.setItem(k,v); }catch(e){} }
  function load(k){ try{ return localStorage.getItem(k); }catch(e){ return null; } }

  function apply(){
    tabs.forEach(function(t){
      var on = t.id === active;
      t.setAttribute("aria-selected", String(on));
      panes[t.id].hidden = !on;
    });
    var pane = panes[active];
    var isSrc = active === "tab-src";
    searchBar.style.display = isSrc ? "none" : "";
    var term = isSrc ? "" : q.value.trim().toLowerCase();
    var mine = rows.filter(function(r){ return pane.contains(r); });
    var shown = 0;
    mine.forEach(function(r){
      var okGroup = !r.dataset.group || group === "all" || r.dataset.group === group;
      var okOpen = !openOnly || r.dataset.state === "open";
      var okTerm = !term || r.dataset.hay.indexOf(term) !== -1;
      var on = okGroup && okOpen && okTerm;
      r.hidden = !on;
      if (on) shown++;
    });
    searchPane.hidden = isSrc || shown !== 0;
    emptyEl.hidden = searchPane.hidden;
    countEl.textContent = mine.length ? (shown + " / " + mine.length + " 행") : "";
    store("ces-tab", active); store("ces-q", q.value);
    store("ces-group", group); store("ces-open", openOnly ? "1" : "0");
  }

  tabs.forEach(function(t){
    t.addEventListener("click", function(){ active = t.id; apply(); });
    t.addEventListener("keydown", function(ev){
      var i = tabs.indexOf(t), n = tabs.length;
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft"){
        ev.preventDefault();
        var nx = tabs[(i + (ev.key === "ArrowRight" ? 1 : n - 1)) % n];
        active = nx.id; apply(); nx.focus();
      }
    });
  });
  groupPills.forEach(function(p){
    p.addEventListener("click", function(){
      group = p.dataset.group;
      groupPills.forEach(function(o){ o.setAttribute("aria-pressed", String(o === p)); });
      apply();
    });
  });
  onlyOpen.addEventListener("click", function(){
    openOnly = !openOnly;
    onlyOpen.setAttribute("aria-pressed", String(openOnly));
    apply();
  });
  q.addEventListener("input", apply);

  var st = load("ces-tab"); if (st && panes[st]) active = st;
  var sq = load("ces-q"); if (sq) q.value = sq;
  var sg = load("ces-group");
  if (sg){
    var hit = groupPills.filter(function(p){ return p.dataset.group === sg; })[0];
    if (hit){ group = sg; groupPills.forEach(function(o){ o.setAttribute("aria-pressed", String(o === hit)); }); }
  }
  if (load("ces-open") === "1"){ openOnly = true; onlyOpen.setAttribute("aria-pressed", "true"); }
  apply();

  document.getElementById("theme").addEventListener("click", function(){
    var root = document.documentElement, now = root.getAttribute("data-theme");
    var dark = now ? now === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.setAttribute("data-theme", dark ? "light" : "dark");
  });
})();
</script>
"""

page = PAGE
page = page.replace("{TRIED_TH}", th([("계열", "7%"), ("모델 · 시도", "17%"), ("시기", "8%"),
                                      ("무엇을 바꿨나 (통제 변수)", "21%"),
                                      ("결과", "23%"), ("닫은 이유 · 근거", "24%")]))
page = page.replace("{TRIED_ROWS}", tried_rows())
page = page.replace("{TAKEAWAY}", E(AP.TRIED_TAKEAWAY))
page = page.replace("{FUSION_TH}", th([("논문", "13%"), ("장치", "7%"), ("연도", "6%"),
                                       ("문제", "14%"), ("구조", "22%"),
                                       ("데이터", "10%"), ("우리에게 의미", "28%")]))
page = page.replace("{FUSION_ROWS}", simple_rows(AP.FUSION_ROWS, FUSION_STATE,
                                                 device_col=1, year_col=2))
page = page.replace("{GENERAL_TH}", th([("방법군", "12%"), ("연도", "7%"), ("대표", "20%"),
                                        ("문헌의 주장", "33%"), ("우리 데이터 판정", "28%")]))
page = page.replace("{GENERAL_ROWS}", simple_rows(AP.GENERAL_ROWS, GENERAL_STATE, year_col=1))
page = page.replace("{ISO_TH}", th([("분야", "16%"), ("연도", "8%"), ("동형 관계", "18%"),
                                    ("주류", "28%"), ("교훈", "30%")]))
page = page.replace("{ISO_ROWS}", simple_rows(AP.ISO_ROWS, ISO_STATE, year_col=1))
page = page.replace("{PRIORITY_TH}", th([("순위", "5%"), ("실험 팔 (통제 변수 하나)", "33%"),
                                         ("출처", "20%"), ("비용", "14%"),
                                         ("사전등록 판정 지표", "28%")]))
page = page.replace("{PRIORITY_ROWS}", priority_rows())
page = page.replace("{NOT_REC}", "".join("<li>%s</li>" % E(t) for t in AP.NOT_RECOMMENDED))
page = page.replace("{SOURCES}", source_lists())

io.open(OUT, "w", encoding="utf-8").write(page)
print("wrote", OUT, len(page), "chars")
for tag in ("{TRIED", "{FUSION", "{GENERAL", "{ISO", "{PRIORITY", "{NOT_REC", "{SOURCES", "{TAKE"):
    assert tag not in page, tag
print("no placeholders left")
