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
    "lstm": {"label": "Recurrent (LSTM)", "short": "Recurrent",
             "arm": "v2r{r}", "reaches": [2, 3, 7, 15, 31, 63]},
    "tcn": {"label": "Dilated convolution", "short": "Convolution",
            "arm": "tcn{r}", "reaches": [3, 7, 15, 63]},
    "xfmr": {"label": "Banded attention", "short": "Attention",
             "arm": "xfmr{r}", "reaches": [7, 15, 63]},
}


def read_rung(arm):
    """-> {target: {mean, values, n_pass, n}} over the 4 seeds, or None if not run."""
    out = {t: {"values": [], "n_pass": 0} for t in TARGETS}
    for seed in SEEDS:
        path = DATA / f".b9_{arm}_s{seed}" / "bootstrap_summary.json"
        if not path.exists():
            return None
        node = json.loads(path.read_text())["splits"]["test"]
        for t in TARGETS:
            gp = node[t]["gp_causal"]
            out[t]["values"].append(gp["skill_point"])
            # "significant" = the paired CI excludes zero on the winning side.
            if gp["skill_ci95"][0] > 0:
                out[t]["n_pass"] += 1
    for t in TARGETS:
        out[t]["mean"] = st.mean(out[t]["values"])
        out[t]["n"] = len(out[t]["values"])
    return out


def collect():
    fams = {}
    missing = []
    for key, spec in FAMILIES.items():
        rungs = []
        for r in spec["reaches"]:
            arm = spec["arm"].format(r=r)
            rung = read_rung(arm)
            if rung is None:
                missing.append(arm)
                continue
            rungs.append({"reach": r, "ms": r * STEP_MS, "arm": arm,
                          **{t: {"mean": rung[t]["mean"], "n_pass": rung[t]["n_pass"],
                                 "values": rung[t]["values"]} for t in TARGETS}})
        fams[key] = {"label": spec["label"], "rungs": rungs}
    return fams, missing


# --- the page -------------------------------------------------------------------------
# Colours are fixed per family and never per rank (a filter must not repaint a survivor).
# Both modes were checked with the dataviz validator before anything was drawn; in light
# mode the aqua slot is below 3:1 against the surface, so the relief rule applies and the
# page ships direct labels and a table view rather than relying on the hue.
COLORS = {"lstm": ("#2a78d6", "#3987e5"), "tcn": ("#eb6834", "#d95926"),
          "xfmr": ("#1baf7a", "#199e70")}

HTML_HEAD = """<title>Reach Ladder by Family</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --surface: #ffffff; --surface-2: #f6f7f9; --ink: #10151c; --ink-2: #48525f;
  --ink-3: #737f8d; --rule: #e3e7ec; --rule-strong: #c3ccd6;
  --lstm: #2a78d6; --tcn: #eb6834; --xfmr: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #11151a; --surface-2: #171c23; --ink: #eef2f6; --ink-2: #b3bec9;
    --ink-3: #8593a1; --rule: #262d36; --rule-strong: #3b4551;
    --lstm: #3987e5; --tcn: #d95926; --xfmr: #199e70;
  }
}
:root[data-theme="dark"] {
  --surface: #11151a; --surface-2: #171c23; --ink: #eef2f6; --ink-2: #b3bec9;
  --ink-3: #8593a1; --rule: #262d36; --rule-strong: #3b4551;
  --lstm: #3987e5; --tcn: #d95926; --xfmr: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.55; }
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
    xs = [20, 30, 70, 150, 310, 630]
    lx0, lx1 = math.log10(18), math.log10(760)

    def px(ms):
        return L + (math.log10(ms) - lx0) / (lx1 - lx0) * (W - L - R)

    panels = []
    for target, title, cap in (
            ("CES_TI", "Ion temperature T_i",
             "positive = beats the causal GP &middot; this is where the threshold lives"),
            ("CES_VT", "Rotation velocity V_rot",
             "no rung is significant on 4 of 4 splits &mdash; this panel settles nothing")):
        vals = [r[target]["mean"] for f in fams.values() for r in f["rungs"]]
        if not vals:
            continue
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = max(0.02, (hi - lo) * 0.18)
        y0, y1 = lo - pad, hi + pad

        def py(v):
            return T + (y1 - v) / (y1 - y0) * (H - T - B)

        s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title} reach ladder">']
        # y grid + ticks, rounded to clean steps
        step = 0.05 if (y1 - y0) > 0.18 else 0.02
        t = math.ceil(y0 / step) * step
        while t <= y1 + 1e-9:
            yy = py(t)
            strong = abs(t) < 1e-9
            s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{W - R}" y2="{yy:.1f}" '
                     f'stroke="var({"--rule-strong" if strong else "--rule"})" stroke-width="1"/>')
            s.append(f'<text class="tick" x="{L - 8}" y="{yy + 3.5:.1f}" '
                     f'text-anchor="end">{t:+.2f}</text>')
            t += step
        # the threshold: dashed on purpose -- it IS a threshold, which is what
        # dashing reads as, and that is why the gridlines above are solid.
        tx = px(70)
        s.append(f'<line x1="{tx:.1f}" y1="{T}" x2="{tx:.1f}" y2="{H - B}" '
                 f'stroke="var(--rule-strong)" stroke-width="1" stroke-dasharray="3 3"/>')
        s.append(f'<text class="thresh" x="{tx + 5:.1f}" y="{T + 11}">70 ms</text>')
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
            pts = [(px(r["ms"]), py(r[target]["mean"])) for r in fam["rungs"]]
            d = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}"
                         for i, (x, y) in enumerate(pts))
            s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
            for (x, y), r in zip(pts, fam["rungs"]):
                full = r[target]["n_pass"] == 4
                # 2px surface ring keeps overlapping markers legible
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="var(--surface)"/>')
                if full:
                    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}"/>')
                else:
                    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--surface)" '
                             f'stroke="{col}" stroke-width="2"/>')
                s.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="13" '
                         f'fill="transparent" data-fam="{fam["label"]}" data-ms="{r["ms"]:.0f}" '
                         f'data-v="{_fmt(r[target]["mean"])}" data-p="{r[target]["n_pass"]}" '
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
                f'<td>{_fmt(r[t]["mean"])}</td><td>{r[t]["n_pass"]}/4</td>' for t in TARGETS)
            rows.append(f'<tr><td><span class="swatch" style="background:var(--{key})"></span>'
                        f'{fam["label"]}</td><td>{r["ms"]:.0f}</td><td>{r["reach"]}</td>'
                        f'{cells}</tr>')

    miss = ""
    if missing:
        miss = ('<p class="meta">Rungs not yet trained: '
                + ", ".join(f"<code>{m}</code>" for m in missing) + "</p>")

    # The headline is derived, not typed: it says what the ladders actually did.
    first4 = {k: next((r["ms"] for r in f["rungs"] if r["CES_TI"]["n_pass"] >= 4), None)
              for k, f in fams.items() if f["rungs"]}
    if missing or None in first4.values() or len(first4) < 3:
        head = "Does the 70 ms threshold belong to the problem or to the LSTM?"
        lede = ("Ladders are still incomplete, so the turning points below are upper bounds "
                "rather than values.")
    elif len(set(first4.values())) == 1:
        head = f"All three families turn at {list(first4.values())[0]:.0f} ms"
        lede = "The threshold is a property of the problem, not of the operator."
    else:
        common = min(first4.values())
        agree = [FAMILIES[k]["short"].lower() for k, v in first4.items() if v == common]
        late = {k: v for k, v in first4.items() if v != common}
        tail = ", ".join(f"{FAMILIES[k]['short'].lower()} needs {v:.0f} ms"
                         for k, v in late.items())
        head = (f"{'Two' if len(agree) == 2 else str(len(agree))} of three families turn at "
                f"{common:.0f} ms &mdash; {tail}")
        lede = (f"The {' and '.join(agree)} ladders clear the bar on all four splits at "
                f"{common:.0f} ms and tie each other rung for rung; the remaining family "
                f"needs one more rung to get there.")

    body = f"""<main>
<h1>{head}</h1>
<p class="sub"><b>{lede}</b> Each family is trained at every reach its receptive field can
declare, then scored against the causal GP &mdash; the strongest past-only baseline &mdash;
on the same four frozen splits. Where the curves turn together, the threshold belongs to the
measurement problem; where one turns later, that is a fact about the operator.</p>
<p class="meta">B.9 axis A + per-family low rungs &middot; W = 2 protocol, held-free, cut
population &middot; mean of 4 split seeds &middot; filled marker = the paired CI clears zero
on all 4</p>
{miss}
<div class="legend">
  <span class="item"><span class="key" style="background:var(--lstm)"></span>Recurrent (LSTM)</span>
  <span class="item"><span class="key" style="background:var(--tcn)"></span>Dilated convolution</span>
  <span class="item"><span class="key" style="background:var(--xfmr)"></span>Banded attention</span>
  <span class="sig"><span class="dot-demo solid"></span>4/4 significant
  <span class="dot-demo" style="margin-left:10px"></span>fewer than 4</span>
</div>
<div class="panels">{"".join(panels)}</div>
<div class="table-wrap">
<table>
<caption>Table view &mdash; every plotted value, so nothing is gated behind hue or hover</caption>
<thead><tr><th>family</th><th>context (ms)</th><th>reach (steps)</th>
<th>T_i skill</th><th>sig.</th><th>V_rot skill</th><th>sig.</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
<p class="note">Read the <b>shape</b>, not the level: the y value is skill against the causal
GP, so zero is &quot;as good as the best deployable baseline&quot; and the question is where
each curve crosses and flattens.<br><br>
<b>Two bars, and they disagree by one rung.</b> The project's promotion bar is <b>&ge; 3 of 4</b>
splits significant, and <i>all three</i> families meet it at 70 ms. The filled markers above use
the stricter <b>4 of 4</b>, and there the attention arm needs 150 ms. Paired against the LSTM
trained at the <i>same</i> reach, the convolution arm ties at both 30 and 70 ms
(&minus;0.004, no significant split either way), while attention at 70 ms loses by
&minus;0.023 with 3 of 4 significant &mdash; the pre-registered &quot;differs&quot; verdict, and
the only one in this batch. It ties again from 150 ms up.<br><br>
<b>What the attention curve does not show.</b> Its lowest rung <i>is</i> 70 ms &mdash; a shorter
band was never trained &mdash; so its 3-of-4 crossing is an upper bound, not a measured turn.</p>
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
      '<div class="t-r"><span>splits with CI &gt; 0</span><span class="t-v">' +
      h.dataset.p + ' / 4</span></div>';
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
    print("1. first rung reaching the promotion bar (skill vs causal GP, CI>0 on N of 4)")
    print("family".rjust(8) + "target".rjust(9) + "  >=3/4 at" + "   4/4 at"
          + "      ladder (ms: skill, sig)")
    first = {}
    for key, fam in fams.items():
        for t in TARGETS:
            r3 = next((r["ms"] for r in fam["rungs"] if r[t]["n_pass"] >= 3), None)
            r4 = next((r["ms"] for r in fam["rungs"] if r[t]["n_pass"] >= 4), None)
            first[(key, t)] = (r3, r4)
            ladder = "  ".join(f"{r['ms']:.0f}:{r[t]['mean']:+.3f}/{r[t]['n_pass']}"
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
