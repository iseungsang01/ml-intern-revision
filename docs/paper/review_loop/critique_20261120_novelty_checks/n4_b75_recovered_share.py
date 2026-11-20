"""N4 (arithmetic on today's B.75 output, unregistered): the share of the staleness cost that the
fast samples recover, from the point RMSE tables of docs/paper/b75_availability.md (target L5).

share = (MSE(T0 under rule) - MSE(TX under rule)) / (MSE(T0 under rule) - MSE(T0 with a 10 ms history))
For fixed delays the 10 ms reference is the D10 row of the same table (common set); for rule S it is
the row "D10 history on the S set". Points only (no intervals): B.75 registered its own
"recovered share" readout; this re-derivation is for the critique text. ASCII output.
"""
import re

txt = open("docs/paper/b75_availability.md", encoding="utf-8").read()
hdr = "| rule | rows / discharges / clusters | H | M5 | G | T0 | TX8 | TX | X0 | O |"
blocks = re.split(r"\n### ", txt)
seen = set()
for b in blocks:
    title = b.splitlines()[0]
    if hdr not in b or "target L5" not in title or title in seen:
        continue
    seen.add(title)
    rows = {}
    for line in b.splitlines():
        if not line.startswith("| ") or line.startswith("| rule") or line.startswith("|---"):
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) < 10:
            continue
        try:
            rows[c[0]] = {"T0": float(c[5]), "TX": float(c[7]), "O": float(c[9]) if c[9] != "-" else None}
        except ValueError:
            continue
    print(title.encode("ascii", "replace").decode())
    ref = rows.get("D10")
    refS = rows.get("D10 history on the S set")
    for r in ("D50", "D100", "D300", "D650", "D1000", "S (schedule)"):
        if r not in rows:
            continue
        base = refS if r.startswith("S") else ref
        if base is None:
            continue
        t0, tx, t0f = rows[r]["T0"] ** 2, rows[r]["TX"] ** 2, base["T0"] ** 2
        share = (t0 - tx) / (t0 - t0f) if t0 > t0f else float("nan")
        print("  %-13s T0 %.1f  TX %.1f  T0(10 ms) %.1f  recovered share %.2f" % (r, rows[r]["T0"], rows[r]["TX"], base["T0"], share))
