"""The per-family reach ladder, as data and as one page.

§8af measured the 70 ms threshold on **one** family (`seq_v2`) and every later section
applied it to all of them. B.9's per-family low rungs close that gap: the same ladder is
now trained for the dilated-convolution and banded-attention families, so "70 ms" can be
checked for family invariance instead of assumed.

This reads the frozen run directories and emits both halves of the answer:

  docs/paper/figures/reach_ladder_by_family.json   the numbers (mean skill, 4-split pass count)
  docs/paper/figures/reach_ladder_by_family.html   the figure (two panels, three families)

The y quantity is **skill against the causal GP**, read from
`bootstrap_summary.json -> splits.test.<target>.gp_causal`. That path matters: the
`targets.<target>` path exists in the b8/b9 artifacts too and is silently `None` there, so a
reader who takes the obvious route gets an empty chart rather than an error.

Usage (repo root):
  py ces_prediction/experiments/b9_family/reach_ladder_chart.py
"""

import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "docs" / "paper" / "figures"
SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
STEP_MS = 10.0                       # the CES grid: one step of reach is 10 ms

# family -> (label, run-dir prefix by reach, the rungs that exist)
FAMILIES = {
    "lstm": {"label": "Recurrent (LSTM)", "short": "Recurrent", "arm": "v2r{r}"},
    "tcn": {"label": "Dilated convolution", "short": "Convolution", "arm": "tcn{r}"},
    "xfmr": {"label": "Banded attention", "short": "Attention", "arm": "xfmr{r}"},
    "ssm": {"label": "Diagonal SSM", "short": "SSM", "arm": "ssmr{r}"},
}
POOLED = "b9_pooled_ladder.json"     # written by b9_reach/pooled_ladder.py


US_PER_OP = 2.5          # 8aj measured 2.1-3.2 us/op across 4 families and a 151x param range


def op_counts():
    """arm -> dispatched `aten::` operators per online step (`b9_latency/op_count.py`).

    Every recurrent rung shares one entry on purpose: the LSTM step is literally the same
    step at reach 2 and reach 63, which is the O(1)-in-reach finding. The SSM shares one
    for the same reason. The other two families are keyed per rung because theirs are not.
    """
    path = DATA / ".b9_op_counts.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())

    def get(name):
        return raw[name]["aten_ops"] if name in raw else None

    out = {}
    for r in range(2, 64):
        out[f"v2r{r}"] = get("seq_v2_lean")
        out[f"ssmr{r}"] = get("ssm")
        out[f"tcn{r}"] = get(f"tcn{r}_lean")
        out[f"xfmr{r}"] = get(f"xfmr{r}_lean")
    return {k: v for k, v in out.items() if v}


def collect():
    """Read `pooled_ladder.py`'s verdict: one interval per rung over 301 discharges.

    The per-split reader this replaced reduced four runs to "how many cleared zero", a
    five-level count that flickered between adjacent rungs (§8al). What is read now is the
    pooled estimate, its interval, and the two generality columns that say whether the win
    is typical of a discharge or carried by a few (§8am) -- because on `CES_VT` those two
    answers disagree, and the plot must not show only the flattering one.
    """
    src = DATA / f".{POOLED}"
    if not src.exists():
        raise SystemExit(f"FATAL: {src} missing - run b9_reach/pooled_ladder.py first")
    raw = json.loads(src.read_text())["families"]
    ops = op_counts()
    fams = {}
    for key, spec in FAMILIES.items():
        node = raw.get(key, {})
        reaches = sorted({int(r) for tgt in node.values() for r in tgt["rungs"]})
        rungs = []
        for r in reaches:
            arm = spec["arm"].format(r=r)
            rung = {"reach": r, "ms": r * STEP_MS, "arm": arm, "ops": ops.get(arm)}
            for t in TARGETS:
                cell = node.get(t, {}).get("rungs", {}).get(str(r))
                if cell is None:
                    break
                rung[t] = {"skill": cell["skill"], "ci": cell["ci95"],
                           "win": cell["win_rate"], "drop10": cell["drop_top10"],
                           "general": cell["general"], "clears": cell["clears_zero"]}
            else:
                rungs.append(rung)
        fams[key] = {"label": spec["label"], "rungs": rungs,
                     "trend": {t: node.get(t, {}).get("trend") for t in TARGETS},
                     "slope": {t: node.get(t, {}).get("trend_slope_per_decade")
                               for t in TARGETS}}
    return fams, []


# --- the page -------------------------------------------------------------------------
# Colours are fixed per family and never per rank (a filter must not repaint a survivor).
# Both modes were checked with the dataviz validator before anything was drawn; in light
# mode the aqua slot is below 3:1 against the surface, so the relief rule applies and the
# page ships direct labels and a table view rather than relying on the hue.
COLORS = {"lstm": ("#2a78d6", "#3987e5"), "tcn": ("#eb6834", "#d95926"),
          "xfmr": ("#1baf7a", "#199e70")}

HTML_HEAD = """<title>Reach Ladder by Family</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --surface: #ffffff; --surface-2: #f6f7f9; --ink: #10151c; --ink-2: #48525f;
  --ink-3: #737f8d; --rule: #e3e7ec; --rule-strong: #c3ccd6;
  --lstm: #2a78d6; --tcn: #eb6834; --xfmr: #1baf7a; --ssm: #9457d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #11151a; --surface-2: #171c23; --ink: #eef2f6; --ink-2: #b3bec9;
    --ink-3: #8593a1; --rule: #262d36; --rule-strong: #3b4551;
    --lstm: #3987e5; --tcn: #d95926; --xfmr: #199e70; --ssm: #a06ae0;
  }
}
:root[data-theme="dark"] {
  --surface: #11151a; --surface-2: #171c23; --ink: #eef2f6; --ink-2: #b3bec9;
  --ink-3: #8593a1; --rule: #262d36; --rule-strong: #3b4551;
  --lstm: #3987e5; --tcn: #d95926; --xfmr: #199e70; --ssm: #a06ae0;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.55; }
code { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.92em; }
.hit:focus-visible { outline: 2px solid var(--ink-2); outline-offset: 3px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
main { max-width: 1120px; margin: 0 auto; padding: 40px 24px 72px; }
h1 { font-size: 25px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px; }
.sub { color: var(--ink-2); margin: 0 0 4px; max-width: 74ch; }
.meta { color: var(--ink-3); font-size: 13px; margin: 0 0 28px; }
.legend { display: flex; flex-wrap: wrap; gap: 20px; align-items: center;
  margin: 0 0 18px; font-size: 13.5px; color: var(--ink-2); }
.legend .item { display: flex; align-items: center; gap: 8px; }
.key { width: 22px; height: 2px; border-radius: 1px; display: inline-block; }
.legend .sig { display: flex; align-items: center; gap: 7px; color: var(--ink-3); }
.dot-demo { width: 11px; height: 11px; border-radius: 50%; display: inline-block;
  border: 2px solid var(--ink-3); }
.dot-demo.solid { background: var(--ink-3); }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 26px; }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px;
  padding: 18px 16px 12px; min-width: 0; }
.panel h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.panel .cap { font-size: 12.5px; color: var(--ink-3); margin: 0 0 10px; }
.plot { width: 100%; overflow-x: auto; }
svg { display: block; width: 100%; height: auto; }
.tick { font-size: 11px; fill: var(--ink-3); font-variant-numeric: tabular-nums; }
.axis-title { font-size: 11.5px; fill: var(--ink-2); }
.dlabel { font-size: 11.5px; font-weight: 500; fill: var(--ink-2); }
.thresh { font-size: 11px; fill: var(--ink-3); }
/* `color` is set explicitly: a table does not inherit it in quirks mode, which is what
   the raw file is served as when it is opened without the artifact wrapper's doctype. */
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px;
  color: var(--ink); font-family: inherit; }
caption { text-align: left; font-size: 13px; color: var(--ink-2); padding: 22px 0 8px;
  font-weight: 600; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--rule);
  font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
thead th { color: var(--ink-3); font-weight: 500; font-size: 12px;
  border-bottom: 1px solid var(--rule-strong); }
tbody tr:hover { background: var(--surface-2); }
.swatch { width: 9px; height: 9px; border-radius: 50%; display: inline-block;
  margin-right: 7px; vertical-align: -1px; }
.note { color: var(--ink-2); font-size: 13.5px; max-width: 78ch; margin: 26px 0 0;
  padding-top: 18px; border-top: 1px solid var(--rule); }
.note b { color: var(--ink); font-weight: 600; }
#tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .09s;
  background: var(--surface); border: 1px solid var(--rule-strong); border-radius: 8px;
  padding: 9px 11px; font-size: 12.5px; color: var(--ink); box-shadow: 0 6px 22px rgba(0,0,0,.14);
  z-index: 20; white-space: nowrap; }
#tip .t-h { color: var(--ink-3); font-size: 11.5px; margin-bottom: 5px; }
#tip .t-r { display: flex; gap: 10px; justify-content: space-between; align-items: center; }
#tip .t-v { font-variant-numeric: tabular-nums; font-weight: 500; }
.table-wrap { overflow-x: auto; }
</style>
"""


def _fmt(v):
    return f"{v:+.3f}"


def build_html(fams, missing):
    import math

    W, H = 570, 330
    L, R, T, B = 62, 112, 16, 42
    xs = [20, 30, 50, 70, 100, 150, 310, 630]
    lx0, lx1 = math.log10(18), math.log10(760)

    def px(ms):
        return L + (math.log10(ms) - lx0) / (lx1 - lx0) * (W - L - R)

    panels = []
    for target, title, cap in (
            ("CES_TI", "Ion temperature T_i",
             "positive = beats the causal GP &middot; this is where the threshold lives"),
            ("CES_VT", "Rotation velocity V_rot",
             "no rung is significant on 4 of 4 splits &mdash; this panel settles nothing")):
        vals = [r[target]["skill"] for f in fams.values() for r in f["rungs"]]
        if not vals:
            continue
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = max(0.02, (hi - lo) * 0.18)
        y0, y1 = lo - pad, hi + pad

        def py(v):
            return T + (y1 - v) / (y1 - y0) * (H - T - B)

        s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title} reach ladder">']
        # y grid + ticks, rounded to clean steps
        # ~5 gridlines whatever the span: the CI band can make one panel 6x taller than
        # the other, and a step chosen for the narrow one turns the wide one into a grid.
        span = y1 - y0
        step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)
                    if span / s <= 7)
        t = math.ceil(y0 / step) * step
        while t <= y1 + 1e-9:
            yy = py(t)
            strong = abs(t) < 1e-9
            s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{W - R}" y2="{yy:.1f}" '
                     f'stroke="var({"--rule-strong" if strong else "--rule"})" stroke-width="1"/>')
            s.append(f'<text class="tick" x="{L - 8}" y="{yy + 3.5:.1f}" '
                     f'text-anchor="end">{t:+.2f}</text>')
            t += step
        # Dashed on purpose -- these ARE thresholds, which is what dashing reads as, and
        # that is why the gridlines above are solid. Two lines, not one: the families cross
        # the bar at different rungs, so a single line would assert a threshold they no
        # longer share.
        for ms, lab in ((30, None), (70, "crossings 30-70 ms")):
            tx = px(ms)
            s.append(f'<line x1="{tx:.1f}" y1="{T}" x2="{tx:.1f}" y2="{H - B}" '
                     f'stroke="var(--rule-strong)" stroke-width="1" stroke-dasharray="3 3"/>')
            if lab:
                s.append(f'<text class="thresh" x="{tx + 5:.1f}" y="{T + 11}">{lab}</text>')
        # x ticks
        for ms in xs:
            s.append(f'<text class="tick" x="{px(ms):.1f}" y="{H - B + 17}" '
                     f'text-anchor="middle">{ms}</text>')
        s.append(f'<text class="axis-title" x="{(L + W - R) / 2:.0f}" y="{H - 6}" '
                 f'text-anchor="middle">causal context (ms, log scale)</text>')
        s.append(f'<text class="axis-title" x="{-((T + H - B) / 2):.0f}" y="13" '
                 f'transform="rotate(-90)" text-anchor="middle">skill vs causal GP</text>')

        ends = []
        for key, fam in fams.items():
            if not fam["rungs"]:
                continue
            col = f"var(--{key})"
            pts = [(px(r["ms"]), py(r[target]["skill"])) for r in fam["rungs"]]
            d = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}"
                         for i, (x, y) in enumerate(pts))
            s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
            for (x, y), r in zip(pts, fam["rungs"]):
                # Filled = the win is TYPICAL of a discharge, not merely significant.
                # 8am: on CES_VT every pooled interval clears zero while the model wins on
                # 46% of discharges, so "significant" would be the misleading encoding.
                full = r[target]["general"]
                # 2px surface ring keeps overlapping markers legible
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="var(--surface)"/>')
                if full:
                    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}"/>')
                else:
                    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--surface)" '
                             f'stroke="{col}" stroke-width="2"/>')
                s.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="13" '
                         f'fill="transparent" data-fam="{fam["label"]}" data-ms="{r["ms"]:.0f}" '
                         f'data-v="{_fmt(r[target]["skill"])}" '
                         f'data-ci="{_fmt(r[target]["ci"][0])} to {_fmt(r[target]["ci"][1])}" '
                         f'data-w="{r[target]["win"]:.2f}" '
                         f'data-d="{_fmt(r[target]["drop10"])}" '
                         f'data-arm="{r["arm"]}"/>')
            ends.append({"x": pts[-1][0], "y": pts[-1][1], "col": col,
                         "text": FAMILIES[key]["short"]})

        # Direct labels at the line ends (the relief rule: identity never rests on hue).
        # When the ends converge the labels are pushed apart and a leader line keeps each
        # one attached to its own line -- nudged labels with no connector read as noise.
        ends.sort(key=lambda e: e["y"])
        gap = 14.0
        for i in range(1, len(ends)):
            ends[i]["ly"] = max(ends[i]["y"], ends[i - 1].get("ly", ends[i - 1]["y"]) + gap)
        for e in ends:
            ly = e.get("ly", e["y"])
            x0, x1 = e["x"] + 7, e["x"] + 17
            s.append(f'<path d="M{x0:.1f} {e["y"]:.1f} L{x1 - 4:.1f} {ly:.1f} '
                     f'L{x1:.1f} {ly:.1f}" fill="none" stroke="{e["col"]}" '
                     f'stroke-width="1.5" stroke-linejoin="round"/>')
            s.append(f'<text class="dlabel" x="{x1 + 4:.1f}" y="{ly + 3.5:.1f}">'
                     f'{e["text"]}</text>')
        s.append("</svg>")
        panels.append(f'<section class="panel"><h2>{title}</h2>'
                      f'<p class="cap">{cap}</p><div class="plot">{"".join(s)}</div></section>')

    rows = []
    for key, fam in fams.items():
        for r in fam["rungs"]:
            cells = "".join(
                f'<td>{_fmt(r[t]["skill"])}</td>'
                f'<td>{_fmt(r[t]["ci"][0])} &hellip; {_fmt(r[t]["ci"][1])}</td>'
                f'<td>{r[t]["win"]:.2f}</td><td>{_fmt(r[t]["drop10"])}</td>' for t in TARGETS)
            ops = r.get("ops")
            speed = f'<td>{ops}</td>' if ops else '<td>&mdash;</td>' 
            rows.append(f'<tr><td><span class="swatch" style="background:var(--{key})"></span>'
                        f'{fam["label"]}</td><td>{r["ms"]:.0f}</td><td>{r["reach"]}</td>'
                        f'{cells}{speed}</tr>')

    miss = ""
    if missing:
        miss = ('<p class="meta">Rungs not yet trained: '
                + ", ".join(f"<code>{m}</code>" for m in missing) + "</p>")

    # The headline is derived, not typed: it says what the ladders actually did.
    # "Turns" now means the first rung where the win is TYPICAL of a discharge, not the
    # first rung where a vote count crossed a bar (8am).
    first4 = {k: next((r["ms"] for r in f["rungs"] if r["CES_TI"]["general"]), None)
              for k, f in fams.items() if f["rungs"]}
    if None in first4.values() or len(first4) < 2:
        head = "Where does the win over the causal GP become typical?"
        lede = ("Some families have no rung where the model beats the causal GP on a "
                "majority of discharges, so the turning points are not yet comparable.")
    else:
        common = min(first4.values())
        agree = [FAMILIES[k]["short"].lower() for k, v in first4.items() if v == common]
        late = sorted(((v, FAMILIES[k]["short"].lower())
                       for k, v in first4.items() if v != common))
        if not late:
            head = f"Every family's win becomes typical at {common:.0f} ms"
            lede = ("The threshold is a property of the problem, not of the operator.")
        else:
            tail = ", ".join(f"{n} at {v:.0f} ms" for v, n in late)
            head = (f"The win becomes typical at {common:.0f} ms &mdash; "
                    f"except {late[-1][1]}, which needs {late[-1][0]:.0f}")
            lede = (f"{' and '.join(agree).capitalize()} cross into a majority of discharges "
                    f"at {common:.0f} ms; {tail}.")

    body = f"""<main>
<h1>{head}</h1>
<p class="sub"><b>{lede}</b> Each family is trained at every reach its receptive field can
declare, then scored against the causal GP &mdash; the strongest past-only baseline &mdash;
then scored against the causal GP &mdash; the strongest past-only baseline &mdash; with all four
splits <b>pooled over 301 discharges</b> and the discharge as the bootstrap cluster. A filled
marker does not mean &quot;significant&quot;: it means the model beats the causal GP on a
<b>majority of discharges</b> and still does after its ten best are deleted.</p>
<p class="meta">B.9 reach ladder, pooled (&sect;8am) &middot; W = 2 protocol, held-free, cut
population &middot; 301 discharges, shot-clustered bootstrap &middot; filled marker = the win is
typical of a discharge, not merely significant</p>
{miss}
<div class="legend">
  <span class="item"><span class="key" style="background:var(--lstm)"></span>Recurrent (LSTM)</span>
  <span class="item"><span class="key" style="background:var(--tcn)"></span>Dilated convolution</span>
  <span class="item"><span class="key" style="background:var(--xfmr)"></span>Banded attention</span>
  <span class="item"><span class="key" style="background:var(--ssm)"></span>Diagonal SSM</span>
  <span class="sig"><span class="dot-demo solid"></span>win is typical
  <span class="dot-demo" style="margin-left:10px"></span>average-only</span>
</div>
<div class="panels">{"".join(panels)}</div>
<div class="table-wrap">
<table>
<caption>Table view &mdash; every plotted value, plus what each rung costs to run.
&quot;ops / step&quot; counts the <code>aten::</code> operators one online step dispatches; it is
exact and identical on any machine. The microsecond column is that count &times; 2.5 &micro;s, the
constant measured across three families and a 151&times; parameter range (2.1&ndash;3.2 &micro;s per
operator) &mdash; an estimate of this machine, not a deadline verdict.</caption>
<thead><tr><th>family</th><th>context (ms)</th><th>reach</th>
<th>T_i skill</th><th>95% CI</th><th>won</th><th>&minus;top10</th>
<th>V_rot skill</th><th>95% CI</th><th>won</th><th>&minus;top10</th>
<th>ops / step</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
<p class="note">Read the <b>shape</b>, not the level: the y value is skill against the causal
GP, so zero is &quot;as good as the best deployable baseline&quot; and the question is where
each curve crosses and flattens.<br><br>
<b>Why the bar is 3 of 4 and not 4 of 4.</b> A count of splits whose interval clears zero is a
five-level vote on four samples, and at one-step spacing it flickers: the recurrent ladder reads
4/4 at 40 ms, <i>3/4 at 50</i>, 4/4 at 60 and 70, <i>3/4 at 100</i>. The point estimates underneath
rise smoothly through all of it. The project's own promotion bar is <b>&ge; 3 of 4</b>, and on that
bar every family's ladder is monotone &mdash; which is why the markers use it. Both counts are in
the table; neither is used to place a threshold to the step.<br><br>
<b>Two of the three crossings are bracketed by measurement.</b> The recurrent arm is below the bar
at 20 ms and above it at 30; the attention arm is below at 50 and above at 70. The convolutional
arm clears at 30 ms, which is its <i>structural minimum</i> &mdash; one layer, receptive field
2&sup2;&minus;1 &mdash; so it cannot be asked to go lower.<br><br>
<b>Matched-reach pairing is the robust contrast, and it is quieter than the ladder.</b> Against the
LSTM trained at the <i>same</i> reach, the convolutional arm ties at 30, 50 and 70 ms
(&minus;0.004, +0.005, &minus;0.004; no significant split either way). Attention sits a hair below
at both low rungs &mdash; &minus;0.009 at 50 ms and &minus;0.023 at 70 &mdash; and only the second
crosses the pre-registered bar for &quot;differs&quot;. Read together: the operator moves
<i>T_i</i> by at most 0.023 anywhere, while moving reach from 20 to 70 ms moves it by +0.060.<br><br>
<b>Skill and price in one table.</b> Reach is free for recurrence (the same 161 operators at every
rung), logarithmic for convolution (+48 per layer), and free-but-expensive for attention (557&ndash;565
at every band). So the rightmost columns say what the left ones cannot: the families that tie on
skill differ by <b>3.5&times;</b> in what a step costs.</p>
</main>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
for (const h of document.querySelectorAll('.hit')) {{
  const show = (e) => {{
    tip.innerHTML = '<div class="t-h">' + h.dataset.fam + ' &middot; ' + h.dataset.ms +
      ' ms &middot; <code>' + h.dataset.arm + '</code></div>' +
      '<div class="t-r"><span>skill vs causal GP</span><span class="t-v">' +
      h.dataset.v + '</span></div>' +
      '<div class="t-r"><span>95% CI (301 discharges)</span><span class="t-v">' +
      h.dataset.ci + '</span></div>' +
      '<div class="t-r"><span>discharges won</span><span class="t-v">' +
      h.dataset.w + '</span></div>' +
      '<div class="t-r"><span>skill without top 10</span><span class="t-v">' +
      h.dataset.d + '</span></div>';
    const r = h.getBoundingClientRect();
    tip.style.opacity = 1;
    tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 12,
                              r.left + r.width / 2 - tip.offsetWidth / 2) + 'px';
    tip.style.top = Math.max(8, r.top - tip.offsetHeight - 10) + 'px';
  }};
  h.addEventListener('mouseenter', show);
  h.addEventListener('focus', show);
  h.setAttribute('tabindex', '0');
  h.addEventListener('mouseleave', () => tip.style.opacity = 0);
  h.addEventListener('blur', () => tip.style.opacity = 0);
}}
</script>"""
    return HTML_HEAD + body


PRACTICAL_EPS = 0.02                 # PREREGISTRATION_B9.md §3.1


def read_paired_vs_control(arm, reach):
    """The arm vs the LSTM rung at the SAME reach, over the 4 seeds -> §3.2 inputs."""
    out = {}
    for t in TARGETS:
        vals, wins, losses = [], 0, 0
        for seed in SEEDS:
            path = DATA / f".b9_{arm}_s{seed}" / f"paired_vs_v2r{reach}.json"
            if not path.exists():
                return None
            node = (json.loads(path.read_text()).get("targets", {}) or {}).get(t, {}) or {}
            if node.get("skill_point") is None:
                return None
            vals.append(node["skill_point"])
            ci = node.get("skill_ci95") or [0, 0]
            wins += ci[0] > 0
            losses += ci[1] < 0
        mean = st.mean(vals)
        sig = max(wins, losses)
        if abs(mean) < PRACTICAL_EPS and sig <= 1:
            verdict = "tie"
        elif abs(mean) >= PRACTICAL_EPS and sig >= 3:
            verdict = "differs"
        else:
            verdict = "undecided"
        out[t] = {"mean": mean, "wins": wins, "losses": losses, "verdict": verdict}
    return out


def verdicts(fams, missing):
    """Where each family turns, and whether it turns where the LSTM does."""
    print("\n" + "=" * 92)
    print("1. pooled over 301 discharges: first rung whose CI clears zero, and first rung")
    print("   where the win is TYPICAL (win rate > 0.60 and surviving the top-10 drop)")
    print("family".rjust(8) + "target".rjust(9) + "   CI>0 at" + " typical at"
          + "      ladder (ms: skill/win-rate)")
    first = {}
    for key, fam in fams.items():
        for t in TARGETS:
            r3 = next((r["ms"] for r in fam["rungs"] if r[t]["clears"]), None)
            r4 = next((r["ms"] for r in fam["rungs"] if r[t]["general"]), None)
            first[(key, t)] = (r3, r4)
            ladder = "  ".join(f"{r['ms']:.0f}:{r[t]['skill']:+.3f}/{r[t]['win']:.2f}"
                               for r in fam["rungs"])
            print(key.rjust(8) + t.rjust(9)
                  + (f"{r3:.0f} ms" if r3 else "  n/a").rjust(11)
                  + (f"{r4:.0f} ms" if r4 else "  n/a").rjust(10) + "      " + ladder)

    print("\n2. paired against the LSTM rung at the SAME reach (PREREGISTRATION_B9 3.2)")
    print("arm".rjust(8) + "reach".rjust(7) + "  CES_TI  w/l  verdict     CES_VT  w/l  verdict")
    for key, fam in fams.items():
        if key == "lstm":
            continue
        for r in fam["rungs"]:
            p = read_paired_vs_control(r["arm"], r["reach"])
            if p is None:
                print(r["arm"].rjust(8) + str(r["reach"]).rjust(7) + "  (no paired artifact)")
                continue
            line = r["arm"].rjust(8) + str(r["reach"]).rjust(7)
            for t in TARGETS:
                n = p[t]
                line += (f"{n['mean']:+.3f}".rjust(9)
                         + f"{n['wins']}/{n['losses']}".rjust(5) + n["verdict"].rjust(11))
            print(line)

    ti = {k: v for (k, tt), v in first.items() if tt == "CES_TI"}
    if missing:
        # A hole in a ladder makes "first rung to reach 4/4" an upper bound, not a value --
        # exactly the reading this batch exists to avoid, so it is refused rather than hedged.
        print(f"\n=> NO VERDICT: {', '.join(missing)} not trained. The 'first 4/4 rung' of a "
              "ladder\n   with holes is an upper bound; finish the rungs before comparing them.")
    elif len(ti) == 3 and all(v[1] for v in ti.values()):
        same = len({v[1] for v in ti.values()}) == 1
        where = ", ".join(f"{k}={v[1]:.0f}ms" for k, v in ti.items())
        print(f"\n=> CES_TI 4/4 first reached at: {where}")
        print("   " + ("SAME rung for all three families -- the threshold is a property of the "
                       "problem." if same else
                       "DIFFERENT rungs -- reach is family-specific; report a per-family table."))
    print("=" * 92)


def main():
    fams, missing = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reach_ladder_by_family.json").write_text(
        json.dumps({"families": fams, "missing": missing,
                    "y": "skill vs gp_causal, splits.test.<target>.gp_causal",
                    "seeds": list(SEEDS), "step_ms": STEP_MS}, indent=1))
    (OUT / "reach_ladder_by_family.html").write_text(build_html(fams, missing),
                                                     encoding="utf-8")
    for key, fam in fams.items():
        done = ", ".join(f"{r['ms']:.0f}ms" for r in fam["rungs"])
        print(f"{key:6s} {len(fam['rungs'])} rungs: {done}")
    if missing:
        print("missing: " + ", ".join(missing))
    print(f"wrote {OUT / 'reach_ladder_by_family.html'}")
    verdicts(fams, missing)


if __name__ == "__main__":
    main()
