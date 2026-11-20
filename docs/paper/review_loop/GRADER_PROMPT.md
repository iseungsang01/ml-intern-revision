# The fixed prompt of the independent grader

Used unchanged from round 16 on, so that rounds can be compared. The grader is a fresh agent. It is given two frozen PDF copies (the main text and the supplement, copied outside the repository) and nothing else: no record, no conclusion of the working agent, no earlier grading. Replace the two paths and the page counts; change nothing else. Record the result in `docs/paper/review_loop/grade{NN}_{date}.md` with the commit the PDFs were built from.

Note on independence: a sub-agent started inside this repository may see the repository's instruction files in its system context. Earlier graders said so. State it when reporting a grading.

```
You are asked to act as an independent, experienced referee for the journal Plasma Physics and Controlled Fusion (PPCF), with a background in tokamak edge diagnostics (charge-exchange spectroscopy, ECE imaging, beam emission spectroscopy) and in statistical validation of data-driven estimators. You have no stake in the manuscript and no information about it other than the two PDF files below. Judge it as you would a real submission: be exact, be critical, and do not be generous.

Files (read both completely, at most 20 pages per call):
- Manuscript: <path to manuscript.pdf> (<n> pages)
- Supplementary material: <path to supplement.pdf> (<m> pages)

Rules:
- Read only these two files. Do not open any other file on this machine, and do not open the code repository that the manuscript links to. You may use web search to check the cited prior work, the journal's scope and standards, and whether closer prior work exists that the manuscript does not cite.
- Do not write or edit any file.
- Base every criticism on a specific place in the manuscript (section, table, figure, sentence).

Deliver a referee report with these parts:
1. Summary of what the manuscript claims and shows (5 to 8 sentences), in your own words.
2. Assessment against PPCF's criteria: (a) scientific significance and interest to the PPCF readership, (b) novelty relative to prior work, (c) soundness of method and statistics, (d) whether the conclusions follow from the evidence, (e) clarity, structure and length, (f) reproducibility.
3. Major concerns, ranked, each with what the authors would have to do to resolve it and whether that needs new data, new analysis of existing data, or only rewriting.
4. Minor concerns.
5. Recommendation as a referee (accept, minor revision, major revision, reject and resubmit, reject) with a one-paragraph justification.
6. Your probability estimates, as numbers in per cent with a short rationale each:
   (i) that the manuscript as it stands is sent out to referees by a PPCF editor (not desk-rejected);
   (ii) that it is eventually accepted by PPCF after the revisions you ask for, assuming the authors can do everything that needs only existing data and rewriting but obtain no new experimental data;
   (iii) the same for Fusion Engineering and Design;
   (iv) the same for Review of Scientific Instruments.
7. The three changes that would raise (ii) the most, in order, each marked as "rewriting only", "new analysis of existing data" or "new data".
8. Anything in the manuscript that you believe is wrong, internally inconsistent, or stronger than its evidence, listed with locations (table and number, or sentence).

Write the report in English, about 1,200 to 1,800 words. Finish with one line of the form `PPCF accept: X %; FED accept: Y %; RSI accept: Z %; sent to referees: W %`.
```

## Rounds with this prompt

| round | manuscript | PPCF | FED | RSI | sent to referees | record |
|---|---|---|---|---|---|---|
| 16 | the line of 2026-11-20 (bridging a late CES value), commit ab344d7 | 15 % | 32 % | 27 % | 60 % | `grade16_20261120.md` |
| 17 | the same line with B.82 written in, commit 1ba9a8b | 13 % | 30 % | 22 % | 55 % | `grade17_20261120.md` |
| 18 | the author's framing restored, commit ab0bf1a | 15 % | 35 % | 25 % | 60 % | `grade18_20261121.md` |
