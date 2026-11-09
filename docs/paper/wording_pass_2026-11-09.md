# Interval-wording pass on docs/paper/main.tex and main_ko.tex

Rule: every result sentence with an evaluative verb (beats / wins / ties / ... ; 이긴다 / 우위 / 동률 / 열세 ...)
restated as point estimate + 95 % interval (or the count of splits whose interval excludes 0 / 1).
No number, macro, label, citation or equation changed. Rules, definitions and captions keep their verbs.

## main.tex (65 edits)

### main.tex #1
Source of any new number: causal-GP range from tab:headline / sec:headline bullet

Before:
```latex
hold in both---the nowcaster beats future-using PCHIP on \Ti{} with skill $+0.17$
to $+0.26$ (spike-cut) and $+0.23$ to $+0.32$ (spike-inclusive), 95\% CIs
excluding zero on \emph{four independent held-out splits in both populations},
and beats the causal GP on all eight; a pre-registered replicate
```
After:
```latex
hold in both---the nowcaster's skill over future-using PCHIP on \Ti{} is $+0.17$
to $+0.26$ (spike-cut) and $+0.23$ to $+0.32$ (spike-inclusive), 95\% CIs
excluding zero on \emph{four independent held-out splits in both populations},
and its skill over the causal GP is $+0.08$ to $+0.17$ with the CI excluding zero on
all eight; a pre-registered replicate
```

### main.tex #2
Source of any new number: median ratio 0.94--1.00 and 1/4 significant from tab:dual / sec:physfilter

Before:
```latex
bootstrap over eight fresh partitions it beats PCHIP, persistence and the causal
GP in mean squared error in both populations ($[+0.07, +0.70]$ cut vs.\ PCHIP)---a
tail-carried win; in the median it ties persistence at the 10\,ms photon floor.
```
After:
```latex
bootstrap over eight fresh partitions its mean-squared-error skill over PCHIP,
persistence and the causal GP has a 95\% interval excluding zero in both populations
($[+0.07, +0.70]$ cut vs.\ PCHIP)---a tail-carried effect; in the median its
error ratio to persistence is 0.94--1.00 (physics-filter median), with the interval
including 1 on 3/4 splits, at the 10\,ms photon floor.
```

### main.tex #3
Source of any new number: tab:mnar / tab:stress ranges; tab:campaign ranges

Before:
```latex
the \emph{genuinely missing} ones keeps the advantage over every causal method on
all splits and over PCHIP on 2/4 (cut) and 4/4 (inclusive); re-splitting strictly
in time across campaigns---which removes the windowed control's advantage over
interpolation---leaves the sequence nowcaster ahead of PCHIP and of the causal GP
on 4/4 initialisations in both populations.
```
After:
```latex
the \emph{genuinely missing} ones keeps the skill over persistence, the causal
reference, at $+0.28$ to $+0.44$ with CIs above zero on all splits and over PCHIP at
$+0.14$ to $+0.28$ with the CI above zero on 2/4 (cut) and 4/4 (inclusive);
re-splitting strictly in time across campaigns---which removes the windowed
control's advantage over interpolation---leaves the sequence nowcaster at $+0.17$ to
$+0.20$ over PCHIP and $+0.11$ to $+0.16$ over the causal GP, CIs above zero on 4/4
initialisations in both populations.
```

### main.tex #4
Source of any new number: sec:deploy (Winkler scores; no interval exists in the record)

Before:
```latex
PASS); calibrated conformal intervals beat both baselines' in every cell; a
```
After:
```latex
PASS); calibrated conformal intervals have a lower (better) Winkler score than
both baselines' in every cell (32/32); a
```

### main.tex #5
Source of any new number: tab:dual

Before:
```latex
cells and beats the backbone's median for \Vrot{} too; because every output that must
```
After:
```latex
cells and its median skill over the backbone for \Vrot{} is $+0.09$ to $+0.27$ with
CIs above zero on 4/4 splits; because every output that must
```

### main.tex #6
Source of any new number: sec:context (point estimates; paired CIs include zero for tcn, attention -0.023 on 3/4)

Before:
```latex
attention tie within 0.023---so the architecture is chosen on cost (dispatched
```
After:
```latex
attention differ by at most 0.023---so the architecture is chosen on cost (dispatched
```

### main.tex #7
Source of any new number: tab:headline; offline-GP range and 1/8 from sec:headline bullet

Before:
```latex
        causal nowcaster beats future-using PCHIP with $\skillp = +0.17$ to $+0.26$
        (cut) and $+0.23$ to $+0.32$ (inclusive), CIs excluding zero on all four
        held-out splits in both populations, and beats the causal GP on all eight
        split--population cells; it ties the offline GP, the strongest
        future-using smoother.
```
After:
```latex
        causal nowcaster's skill over future-using PCHIP is $\skillp = +0.17$ to $+0.26$
        (cut) and $+0.23$ to $+0.32$ (inclusive), CIs excluding zero on all four
        held-out splits in both populations, and its skill over the causal GP is
        $+0.08$ to $+0.17$ with CIs excluding zero on all eight split--population
        cells; against the offline GP, the strongest future-using smoother, the paired
        skill is $-0.05$ to $+0.11$ with the CI including zero on 7/8 cells.
```

### main.tex #8
Source of any new number: tab:dual

Before:
```latex
        alone would give) and was first recorded as a tie; the repeated-5-fold
        replicate, read by its registered two-level bootstrap over eight fresh
        partitions, shows that the mean-squared-error win over PCHIP, persistence
        and the causal GP is real in both populations ($[+0.07, +0.70]$ cut and
        $[+0.05, +0.68]$ inclusive vs.\ PCHIP) and tail-carried, while in the
        median \Vrot{} ties persistence at the 10\,ms photon floor
        (\S\ref{sec:physfilter}). The win comes from history alone:
```
After:
```latex
        alone would give) and was first recorded as an interval including zero; the
        repeated-5-fold replicate, read by its registered two-level bootstrap over
        eight fresh partitions, puts the mean-squared-error skill over PCHIP,
        persistence and the causal GP above zero in both populations ($[+0.07,
        +0.70]$ cut and $[+0.05, +0.68]$ inclusive vs.\ PCHIP), tail-carried, while
        in the median the error ratio to persistence is 0.94--1.00 with the interval
        including 1 on 3/4 splits, at the 10\,ms photon floor
        (\S\ref{sec:physfilter}). The skill comes from history alone:
```

### main.tex #9
Source of any new number: tab:mnar

Before:
```latex
        (\S\ref{sec:mnar}, \S\ref{sec:campaign}). Reweighting the scored points
        onto the covariate distribution of the genuinely missing ones keeps the
        \Ti{} advantage over every causal method on 4/4 splits in both
        populations and over PCHIP on 2/4 (cut) and 4/4 (inclusive).
```
After:
```latex
        (\S\ref{sec:mnar}, \S\ref{sec:campaign}). Reweighting the scored points
        onto the covariate distribution of the genuinely missing ones keeps the
        \Ti{} skill over persistence, the causal reference, at $+0.28$ to $+0.44$
        with CIs above zero on 4/4 splits in both populations and over PCHIP at
        $+0.14$ to $+0.28$ with the CI above zero on 2/4 (cut) and 4/4 (inclusive).
```

### main.tex #10
Source of any new number: tab:campaign

Before:
```latex
        standardizes each discharge's inputs by its own statistics, stays ahead of
        PCHIP and of the causal GP on 4/4 initialisations in both populations.
```
After:
```latex
        standardizes each discharge's inputs by its own statistics, keeps $+0.17$ to
        $+0.20$ over PCHIP and $+0.11$ to $+0.16$ over the causal GP with CIs above
        zero on 4/4 initialisations in both populations.
```

### main.tex #11
Source of any new number: tab:ladder2 (mean -0.194, 4/4*)

Before:
```latex
        (paired mean $+0.002$)---and loses $-0.19$ without it, because a bounded
```
After:
```latex
        (paired mean $+0.002$)---and sits $-0.19$ below it without the cut (CI below
        zero on 4/4), because a bounded
```

### main.tex #12
Source of any new number: tab:family deltas and k=0 row; sec:family ensemble paragraph

Before:
```latex
        aggregate cost (cut population). A shot-encoding neural process, a
        transformer pretrained on another tokamak, and a 4-init ensemble all tie;
        with no same-shot label, no learned arm beats the training mean on \Vrot{}.
```
After:
```latex
        aggregate cost (cut population). A shot-encoding neural process, a
        transformer pretrained on another tokamak, and a 4-init ensemble all change
        \Ti{} skill by $-0.09$ to $+0.03$ against their control, none clearing the
        0.02 band in both populations (ensemble $+0.020$ cut, $+0.007$ inclusive);
        with no same-shot label, every learned arm's \Vrot{} skill against the
        training mean is negative ($-0.003$ to $-0.040$).
```

### main.tex #13
Source of any new number: tab:dual

Before:
```latex
        and beats the backbone's median for both targets on 4/4 splits; three ways of
```
After:
```latex
        and its median skill over the backbone is $+0.06$ to $+0.39$ (\Ti{}) and
        $+0.09$ to $+0.27$ (\Vrot{}) with CIs above zero on 4/4 splits; three ways of
```

### main.tex #14
Source of any new number: sec:context

Before:
```latex
        dilated convolution and attention ties within 0.023. Context makes the win
        typical;
```
After:
```latex
        dilated convolution and attention differs by at most 0.023. Context makes the
        positive per-discharge skill typical;
```

### main.tex #15
Source of any new number: sec:deploy

Before:
```latex
        conformal intervals that beat both baselines' intervals in all 32
```
After:
```latex
        conformal intervals whose Winkler score is below both baselines' in all 32
```

### main.tex #16
Source of any new number: numbers already in sentence

Before:
```latex
beats the increment-form equation on every split (paired $+0.034$ to $+0.042$, CIs above
zero), and learns
```
After:
```latex
sits above the increment-form equation on every split by a paired $+0.034$ to $+0.042$
(CIs above zero), and learns
```

### main.tex #17
Source of any new number: tab:ladder; offline-GP paired skill from sec:headline bullet

Before:
```latex
\subsection{The nowcaster dominates every causal baseline}
Table~\ref{tab:ladder} shows the physical-unit RMSE ladder on the canonical test
split. \seqv{} has the lowest RMSE for both targets among all methods that do not
read the future, beats the causal GP---the strongest deployable arm---by 4\% (\Ti)
and 18\% (\Vrot) in RMSE, and edges the future-using interpolants; the offline GP,
which averages noisy anchors rather than passing through them, is the one arm it
ties (\Ti{} 153.8 vs.\ 157.8). This is the most robust result in the paper: for any
online setting, where future CES is by definition unavailable, the sequence
nowcaster is the clear winner.
```
After:
```latex
\subsection{The nowcaster has the lowest RMSE of every causal baseline}
Table~\ref{tab:ladder} shows the physical-unit RMSE ladder on the canonical test
split. \seqv{} has the lowest RMSE for both targets among all methods that do not
read the future, 4\% (\Ti) and 18\% (\Vrot) below the causal GP---the strongest
deployable arm---and below the future-using interpolants; the offline GP, which
averages noisy anchors rather than passing through them, is the one arm with a lower
\Ti{} RMSE (153.8 vs.\ 157.8; paired skill $-0.05$ to $+0.11$ with the CI including
zero on 7/8 cells, \S\ref{sec:headline}). This is the most robust result in the
paper: for any online setting, where future CES is by definition unavailable, the
sequence nowcaster has the lowest error.
```

### main.tex #18
Source of any new number: tab:headline

Before:
```latex
\subsection{Headline: \Ti{} beats future-using interpolation on four independent
splits, in both populations}
```
After:
```latex
\subsection{Headline: \Ti{} skill over future-using interpolation excludes zero on
four independent splits, in both populations}
```

### main.tex #19
Source of any new number: numbers already in sentence

Before:
```latex
        every shot-clustered CI excludes zero, and every one of the eight
        population--split cells also beats the \emph{causal GP} ($+0.08$ to
        $+0.17$) and persistence ($+0.36$ to $+0.46$). This is the unqualified
        claim of the paper. Against the offline GP the backbone \emph{ties}
        (point estimates $-0.05$ to $+0.11$, significant on 1/8 cells), so the
        observed-population claim reads ``beats the pre-registered interpolants
        and every deployable causal method, and matches a future-using,
        per-sample-tuned smoother''.
```
After:
```latex
        every shot-clustered CI excludes zero, and in every one of the eight
        population--split cells the skill over the \emph{causal GP} ($+0.08$ to
        $+0.17$) and over persistence ($+0.36$ to $+0.46$) also has its CI above
        zero. This is the unqualified claim of the paper. Against the offline GP
        the backbone's paired skill is $-0.05$ to $+0.11$ with the CI including
        zero on 7/8 cells, so the observed-population claim reads ``skill above
        zero over the pre-registered interpolants and every deployable causal
        method, and an interval including zero against a future-using,
        per-sample-tuned smoother''.
```

### main.tex #20
Source of any new number: tab:headline / tab:replicate

Before:
```latex
  \item \textbf{\Vrot{}: a tie on the four-split count, a registered win on the
        replicate.}
```
After:
```latex
  \item \textbf{\Vrot{}: intervals including zero on the four-split count, a
        registered interval above zero on the replicate.}
```

### main.tex #21
Source of any new number: numbers already in sentence

Before:
```latex
        discharge by discharge across eight fresh partitions, \Vrot{} beats PCHIP
        ($[+0.07, +0.70]$ cut, $[+0.05, +0.68]$ inclusive), persistence
        ($[+0.28, +0.76]$, $[+0.14, +0.75]$) and the causal GP ($[+0.05, +0.70]$,
        $[+0.02, +0.68]$) under the registered two-level bootstrap, every one of
```
After:
```latex
        discharge by discharge across eight fresh partitions, the \Vrot{} skill over
        PCHIP is $[+0.07, +0.70]$ (cut) and $[+0.05, +0.68]$ (inclusive), over
        persistence $[+0.28, +0.76]$ and $[+0.14, +0.75]$, and over the causal GP
        $[+0.05, +0.70]$ and $[+0.02, +0.68]$ under the registered two-level
        bootstrap, every interval above zero, every one of
```

### main.tex #22
Source of any new number: tab:dual

Before:
```latex
        fold's ratio is set by a few heavy-error discharges. The win is a
        mean-squared-error, tail-carried one (\S\ref{sec:bulk}); in the median
        \Vrot{} ties persistence at the 10\,ms photon floor
        (\S\ref{sec:physfilter}), and the two statements are made together.
```
After:
```latex
        fold's ratio is set by a few heavy-error discharges. The effect is a
        mean-squared-error, tail-carried one (\S\ref{sec:bulk}); in the median the
        \Vrot{} error ratio to persistence is 0.94--1.00, with the interval
        including 1 on 3/4 splits, at the 10\,ms photon floor
        (\S\ref{sec:physfilter}), and the two statements are made together.
```

### main.tex #23
Source of any new number: numbers already in sentence

Before:
```latex
significant \Vrot{} deficit (8/16 significant wins). All four conditions held.
```
After:
```latex
significant \Vrot{} deficit (8/16 with the CI above zero). All four conditions held.
```

### main.tex #24
Source of any new number: +0.041 (1/4) from sec:context 'Why the windowed family could not reach this axis' and THESIS sec. 8am; backbone range tab:headline

Before:
```latex
The gate also settles \emph{what} the framing buys. The windowed control ties the
causal GP (significant on 1/4 splits, cut) and the sequence backbone beats it on
4/4 in both populations: the reach
```
After:
```latex
The gate also settles \emph{what} the framing buys. The windowed control's skill
over the causal GP is $+0.041$ with the CI including zero on 3/4 splits (cut), and
the sequence backbone's is $+0.08$ to $+0.17$ with CIs above zero on 4/4 in both
populations: the reach
```

### main.tex #25
Source of any new number: PCHIP CIs from tab:gap; persistence CIs from paper_numbers.json largegap_pooled (cut Ti >15: [0.342,0.452]; incl Ti >15: [0.353,0.519]; cut Vrot <=15: [0.206,0.455])

Before:
```latex
\paragraph{The \Ti{} win is not confined to adjacent history.} Beyond 15\,ms the
backbone beats future-using PCHIP in both populations ($+0.268$ and $+0.206$, both
CIs above zero) and beats persistence there by $+0.40$ and $+0.43$. Beyond
45\,ms it still wins under the cut ($+0.267$) and \emph{ties} in the inclusive
population ($-0.004$, CI $[-0.30, +0.25]$): 429 rows over 101 shots is exactly
where a handful of spike-anchored rows can dominate a stratum, and we report the
tie as such.

\paragraph{\Vrot{} beats interpolation in the one place it is hardest for
interpolation.} The global \Vrot{} tie is a tie in the adjacent regime; over
$\Delta t > 15$\,ms the backbone beats PCHIP in both populations ($+0.418$ and
$+0.432$, 130 shots)---the single positive unconditional \Vrot{} verdict in the
paper---and beats persistence at $\leq 15$\,ms ($+0.39$). The widest bin has 14
rows and is not scored.
```
After:
```latex
\paragraph{The \Ti{} skill is not confined to adjacent history.} Beyond 15\,ms the
backbone's skill over future-using PCHIP is $+0.268$ and $+0.206$ in the two
populations (both CIs above zero) and its skill over persistence there is $+0.40$
and $+0.43$ (CIs $[+0.34, +0.45]$ and $[+0.35, +0.52]$). Beyond 45\,ms the skill
over PCHIP is $+0.267$ $[+0.09, +0.41]$ under the cut and $-0.004$ with CI
$[-0.30, +0.25]$ in the inclusive population: 429 rows over 101 shots is exactly
where a handful of spike-anchored rows can dominate a stratum, and we report the
included-zero interval as such.

\paragraph{\Vrot{} skill over interpolation excludes zero in the one place it is
hardest for interpolation.} The global \Vrot{} interval that includes zero is the
adjacent regime's; over $\Delta t > 15$\,ms the backbone's skill over PCHIP is
$+0.418$ $[+0.10, +0.68]$ and $+0.432$ $[+0.13, +0.70]$ in the two populations (130
shots)---the single unconditional \Vrot{} interval above zero in the paper---and
its skill over persistence at $\leq 15$\,ms is $+0.39$ $[+0.21, +0.45]$. The widest
bin has 14 rows and is not scored.
```

### main.tex #26
Source of any new number: control range +0.09..+0.18 from tab:campaign last column

Before:
```latex
\textbf{the sequence nowcaster does not collapse}. It beats PCHIP on 4/4
initialisations in both populations ($+0.17$ to $+0.20$), beats the causal GP on
4/4 in both ($+0.11$ to $+0.16$), and beats the windowed control on 8/8; its
\Vrot{} beats persistence on 4/4 in both populations ($+0.31$ to $+0.34$) where
the windowed control's does on 0/4, and its \Vrot{} margin over the control is
significant on 8/8 ($+0.06$ to $+0.18$).
```
After:
```latex
\textbf{the sequence nowcaster does not collapse}. Its skill over PCHIP is $+0.17$
to $+0.20$ with the CI above zero on 4/4 initialisations in both populations, its
skill over the causal GP $+0.11$ to $+0.16$ (CI above zero on 4/4 in both), and its
paired skill over the windowed control $+0.09$ to $+0.18$ (CI above zero on 8/8); its
\Vrot{} skill over persistence is $+0.31$ to $+0.34$ with the CI above zero on 4/4
in both populations where the windowed control's interval includes zero on 4/4, and
its \Vrot{} margin over the control is $+0.06$ to $+0.18$ with the CI above zero on
8/8.
```

### main.tex #27
Source of any new number: tab:stress

Before:
```latex
Table~\ref{tab:stress} puts them side by side for the adopted model. The
advantage over the baselines a real-time system can actually run survives both
stress tests in both populations. The advantage over \emph{future-using}
interpolation survives the temporal shift in both populations and the MNAR
reweighting with population-conditional significance.
```
After:
```latex
Table~\ref{tab:stress} puts them side by side for the adopted model. The skill
over the baselines a real-time system can actually run survives both stress tests
in both populations with CIs above zero ($+0.28$ to $+0.44$ over persistence under
reweighting, $+0.11$ to $+0.16$ over the causal GP under the campaign split). The
skill over \emph{future-using} interpolation survives the temporal shift in both
populations ($+0.17$ to $+0.20$, CIs above zero) and the MNAR reweighting with
population-conditional significance ($+0.14$ to $+0.28$, CI above zero on 2/4 cut
and 4/4 inclusive).
```

### main.tex #28
Source of any new number: numbers already in sentence (no per-split CI for this cell in the record)

Before:
```latex
  \item \textbf{In the inclusive population the same history-only model still
        beats PCHIP by $+0.15$ to $+0.23$}, and the fast channels add only
```
After:
```latex
  \item \textbf{In the inclusive population the same history-only model still
        has skill $+0.15$ to $+0.23$ over PCHIP}, and the fast channels add only
```

### main.tex #29
Source of any new number: wording only

Before:
```latex
it; the \Vrot{} non-win is a finding about diagnostic information content, not a
```
After:
```latex
it; the \Vrot{} null is a finding about diagnostic information content, not a
```

### main.tex #30
Source of any new number: anchor range tab:ladder2; vs causal GP from paper_numbers.json ladder/b3k8 vs_gp_causal_TI 0.069/0.129/0.161/0.101 (all pass)

Before:
```latex
        compression to eight bounded numbers plus persistence.} \bthree{} beats
        the retrained anchor on 4/4 splits with no \Vrot{} deficit, and its paired
        \Ti{} skill against the backbone is $-0.009$, $-0.005$, $+0.026$, $-0.004$
        (mean $+0.002$, every CI containing zero); it passes PR4 vs.\ PCHIP 4/4 and
        beats the causal GP 4/4. Linear probes
```
After:
```latex
        compression to eight bounded numbers plus persistence.} \bthree{}'s paired
        skill over the retrained anchor is $+0.35$ to $+0.42$ with the CI above zero
        on 4/4 splits and no \Vrot{} deficit, and its paired \Ti{} skill against the
        backbone is $-0.009$, $-0.005$, $+0.026$, $-0.004$ (mean $+0.002$, every CI
        containing zero); it passes PR4 vs.\ PCHIP 4/4 and its skill over the causal
        GP is $+0.07$ to $+0.16$ with the CI above zero on 4/4. Linear probes
```

### main.tex #31
Source of any new number: tab:headline PR4 counts

Before:
```latex
        count \Vrot{} ties interpolation globally (the replicate's pooled readout
        later registers the mean win, \S\ref{sec:headline}), but in high-activity
```
After:
```latex
        count the global \Vrot{} interval vs.\ interpolation includes zero on 3/4
        and 2/4 splits (the replicate's pooled readout later puts the mean skill
        above zero, \S\ref{sec:headline}), but in high-activity
```

### main.tex #32
Source of any new number: numbers already in sentence; single discharge has no interval

Before:
```latex
spread is the per-discharge covariate that predicts where the model wins (Spearman
$\rho = +0.40$ with the per-discharge skill against the causal GP, pooled over the
splits). Where \Ti{} moves
(spread 220\,eV) the nowcast beats PCHIP by $+0.42$ in MSE and ties the causal GP
($+0.03$). Where it does not (spread 34\,eV, \#31089) the nowcast follows the
photon noise of the last sample and loses to both ($-1.06$, $-0.73$), which is the
median-row cost \S\ref{sec:bulk} measures on every split.
```
After:
```latex
spread is the per-discharge covariate that predicts where the model's per-discharge
skill is positive (Spearman $\rho = +0.40$ with the per-discharge skill against the
causal GP, pooled over the splits). Where \Ti{} moves (spread 220\,eV) the nowcast's
MSE skill is $+0.42$ over PCHIP and $+0.03$ over the causal GP (single-discharge
point estimates, no interval). Where it does not (spread 34\,eV, \#31089) the nowcast
follows the photon noise of the last sample and its skill is $-1.06$ and $-0.73$
against the two, which is the median-row cost \S\ref{sec:bulk} measures on every
split.
```

### main.tex #33
Source of any new number: wording only

Before:
```latex
therefore not, by itself, a statement that the model beats interpolation on a
\emph{typical} row.
```
After:
```latex
therefore not, by itself, a statement that the model's error is below interpolation's on a
\emph{typical} row.
```

### main.tex #34
Source of any new number: win rate 0.48 from tab:bulk

Before:
```latex
$-0.562$). On half of the rows the model ties or trails
PCHIP; the quantile ladder of the error ratio model/PCHIP falls from 1.20 at q50
through 1.01 at q70 and 0.96 at q75 to 0.78 at q99---the win begins in the upper
error quartile
```
After:
```latex
$-0.562$). On half of the rows the model's error is at or above PCHIP's (win rate
0.48); the quantile ladder of the error ratio model/PCHIP falls from 1.20 at q50
through 1.01 at q70 and 0.96 at q75 to 0.78 at q99---the ratio drops below 1 in the upper
error quartile
```

### main.tex #35
Source of any new number: numbers already in sentence

Before:
```latex
(4.3 vs 3.7\,km/s). The aggregate tie is a bulk loss offset by a tail win
(peak-stratum $+0.54$ to $+0.79$, \S\ref{sec:peak})
```
After:
```latex
(4.3 vs 3.7\,km/s). The aggregate interval that includes zero is a bulk deficit
offset by a tail gain (peak-stratum $+0.54$ to $+0.79$, \S\ref{sec:peak})
```

### main.tex #36
Source of any new number: Student-t MSE -0.018 [-0.033, -0.001], 1/4 from THESIS_RESULTS.md B.14 table (line ~4496)

Before:
```latex
3/4) while the aggregate MSE skill ties (paired $-0.004$ / $-0.007$, significant
deficits 0/4). It is the first arm whose median beats persistence ($+0.058$, 3/4)
and ties the causal GP, and it halves the median gap to PCHIP from $-0.198$ to
$-0.101$. The Student-$t$ arm is behind Huber on every metric and loses \Ti{} MSE
significantly on one split: a redescending
```
After:
```latex
3/4) while the aggregate MSE skill is paired $-0.004$ / $-0.007$ with the interval
including zero (significant deficits 0/4). It is the first arm whose median skill
over persistence is $+0.058$ with the CI above zero on 3/4 splits and whose median
interval against the causal GP includes zero, and it halves the median gap to PCHIP
from $-0.198$ to $-0.101$. The Student-$t$ arm sits below Huber on every metric and
its \Ti{} MSE against the backbone is $-0.018$ $[-0.033, -0.001]$, with the
individual CI below zero on 1/4 splits: a redescending
```

### main.tex #37
Source of any new number: CI from paper_numbers.json b14_select/cut_huber_TI/m1_vs_gp_causal_ci

Before:
```latex
strengthens the rung: Huber \Ti{} $M_1 = +0.128$ $[+0.114, +0.141]$ (4/4) at an MSE
tie (pooled $+0.007$, significant deficits 0/4), the \Ti{} median at 0.894$\times$
persistence---the band the physics filter reaches (\S\ref{sec:physfilter})---and a
median that now beats the causal GP ($+0.073$, CI $>0$).
```
After:
```latex
strengthens the rung: Huber \Ti{} $M_1 = +0.128$ $[+0.114, +0.141]$ (4/4) at an MSE
difference of $+0.007$ pooled with the interval including zero (significant deficits
0/4), the \Ti{} median at 0.894$\times$ persistence---the band the physics filter
reaches (\S\ref{sec:physfilter})---and a median skill over the causal GP of $+0.073$
$[+0.043, +0.099]$ (CI $>0$).
```

### main.tex #38
Source of any new number: tab:dual

Before:
```latex
\Ti{} the physics filter is the median estimator \textbf{unconditionally}: it beats
the backbone and persistence on 4/4 splits in both populations, at 0.906--0.920$\times$
persistence with a split spread of $\pm0.007$, against $\pm0.1$ or more for every
trained arm. For \Vrot{} it beats the backbone in the median on 4/4 splits in both
populations and ties persistence (1/4 significant, 0.94--1.00$\times$ by split). The
tie survives fitting the filter on TRAIN.
```
After:
```latex
\Ti{} the physics filter is the median estimator \textbf{unconditionally}: its median
skill is $+0.06$ to $+0.39$ over the backbone and $+0.08$ to $+0.09$ over persistence,
CIs above zero on 4/4 splits in both populations, at 0.906--0.920$\times$
persistence with a split spread of $\pm0.007$, against $\pm0.1$ or more for every
trained arm. For \Vrot{} its median skill over the backbone is $+0.09$ to $+0.27$ with
CIs above zero on 4/4 splits in both populations, and its median-error ratio to
persistence is 0.94--1.00$\times$ by split with the interval including 1 on 3/4 splits.
That interval still includes 1 after fitting the filter on TRAIN.
```

### main.tex #39
Source of any new number: numbers already in sentence

Before:
```latex
persistence beaten on at most 2/4 splits. The reason is measured.
```
After:
```latex
the interval excluding 1 on at most 2/4 splits. The reason is measured.
```

### main.tex #40
Source of any new number: numbers already in sentence

Before:
```latex
last value. The tie is therefore a limit of the 10\,ms CES measurement, not of the
estimator. The
filter loses the aggregate MSE to the backbone ($-0.06$ to $-0.52$) because event
```
After:
```latex
last value. The included-1 interval is therefore a limit of the 10\,ms CES measurement, not of the
estimator. The
filter's aggregate MSE skill against the backbone is $-0.06$ to $-0.52$ because event
```

### main.tex #41
Source of any new number: -0.03..-0.09 from THESIS_RESULTS.md sec. 8bf (line ~5265)

Before:
```latex
$|v| > 1000$\,km\,s$^{-1}$ fit failures and its mean loses MSE to the backbone on
3/4 cut splits.
```
After:
```latex
$|v| > 1000$\,km\,s$^{-1}$ fit failures and its mean's MSE skill against the backbone
is $-0.03$ to $-0.09$, with the CI below zero on 3/4 cut splits.
```

### main.tex #42
Source of any new number: numbers already in sentence

Before:
```latex
\paragraph{The named arms are calibrated.} In the cut population the Student-$t$ head
ties the mixture (CRPS difference $+0.009$, below the 0.02 tie width) and is named for
having fewer parameters. It beats the reference in CRPS by $+0.229$ (\Ti{}) and
$+0.363$ (\Vrot{}), significant on every split,
```
After:
```latex
\paragraph{The named arms are calibrated.} In the cut population the Student-$t$ head's
CRPS difference to the mixture is $+0.009$, inside the 0.02 tie width, and it is named for
having fewer parameters. Its CRPS gain over the reference is $+0.229$ (\Ti{}) and
$+0.363$ (\Vrot{}), with the CI above zero on every split,
```

### main.tex #43
Source of any new number: deltas and k=0 points from tab:family; k=0 CIs from THESIS_RESULTS.md family-card table (line ~4662)

Before:
```latex
Three things are read off Table~\ref{tab:family}. \emph{First}, on \Vrot{} the family
ties: the two cut-population edge cells have CI lows of $0.000$ and $0.001$, all
three arms tie in the inclusive population, and no stratum reproduces either cell.
\emph{Second}, the neural process's hypothesis is rejected exactly where it was
built to hold: with no same-shot \Vrot{} label \emph{no} arm beats the training mean
and \texttt{np} is the worst,
```
After:
```latex
Three things are read off Table~\ref{tab:family}. \emph{First}, on \Vrot{} the family's
$\delta$ against the control is $-0.016$ to $+0.021$: the two cut-population edge
cells have CI lows of $0.000$ and $0.001$, the three inclusive-population cells sit at
$-0.003$ to $-0.016$ inside the 0.02 band, and no stratum reproduces either cell.
\emph{Second}, the neural process's hypothesis is rejected exactly where it was
built to hold: with no same-shot \Vrot{} label \emph{every} arm's skill against the
training mean is negative ($-0.003$ to $-0.040$; \texttt{np} $-0.040$ $[-0.07, -0.02]$,
control $-0.025$ $[-0.06, +0.01]$) and \texttt{np} is the worst,
```

### main.tex #44
Source of any new number: tab:family

Before:
```latex
$+0.026$ clears the rule in the cut population only and ties in the inclusive one,
so it is recorded
```
After:
```latex
$+0.026$ clears the rule in the cut population only and is $-0.006$, inside the band,
in the inclusive one, so it is recorded
```

### main.tex #45
Source of any new number: numbers already in sentence (no interval in the record)

Before:
```latex
The card's one positive result is the gap-length profile. On \Vrot{} all four
learned arms lose to PCHIP in one-step gaps ($-0.25$) and win by $+0.4$ to $+0.6$
in gaps of two steps or more
```
After:
```latex
The card's one positive result is the gap-length profile. On \Vrot{} all four
learned arms have skill $-0.25$ against PCHIP in one-step gaps and $+0.4$ to $+0.6$
in gaps of two steps or more
```

### main.tex #46
Source of any new number: wording only

Before:
```latex
say, is where the \Vrot{} tie of \S\ref{sec:asym} lives.
```
After:
```latex
say, is where the \Vrot{} null of \S\ref{sec:asym} lives.
```

### main.tex #47
Source of any new number: numbers already in sentence

Before:
```latex
discharges gives Table~\ref{tab:pooled}, and \textbf{the model beats the causal GP at
20\,ms as well} ($+0.057$, 95\% CI $[+0.027, +0.085]$).
```
After:
```latex
discharges gives Table~\ref{tab:pooled}, and \textbf{the model's skill over the causal
GP is above zero at 20\,ms as well} ($+0.057$, 95\% CI $[+0.027, +0.085]$).
```

### main.tex #48
Source of any new number: tab:pooled; +0.356 has no interval in the record (THESIS sec. 8al)

Before:
```latex
\rightarrow 0.66$, then flat from about 70\,ms. At 20\,ms the model wins on average but on
barely half the discharges; by 70\,ms it wins on two thirds. \textbf{Context does not raise
the average so much as it makes the win typical.} That is the statement a deployment
audience needs, and the threshold must not be described as "the context needed to be
useful": already at 20\,ms the model beats persistence---what a deployed system actually
displaces---by $+0.356$.
```
After:
```latex
\rightarrow 0.66$, then flat from about 70\,ms. At 20\,ms the pooled skill is $+0.057$ with
the CI above zero but the per-discharge win rate is 0.52; by 70\,ms it is 0.66.
\textbf{Context does not raise the average so much as it raises the win rate.} That is the
statement a deployment audience needs, and the threshold must not be described as "the
context needed to be useful": already at 20\,ms the model's skill over persistence---what
a deployed system actually displaces---is $+0.356$.
```

### main.tex #49
Source of any new number: numbers already in sentence

Before:
```latex
causal convolution (3{,}238) and banded attention (295{,}746), all three beat the causal
GP 4/4 at $+0.113$, $+0.119$ and $+0.119$.
```
After:
```latex
causal convolution (3{,}238) and banded attention (295{,}746), all three have skill over
the causal GP of $+0.113$, $+0.119$ and $+0.119$ with the CI above zero on 4/4 splits.
```

### main.tex #50
Source of any new number: numbers already in sentence

Before:
```latex
contrast---the differences are quieter still: \texttt{tcn3/5/7} tie the same-reach LSTM
($-0.004$, $+0.005$, $-0.004$, with no significant split either way)
```
After:
```latex
contrast---the differences are quieter still: \texttt{tcn3/5/7} differ from the same-reach
LSTM by $-0.004$, $+0.005$, $-0.004$, with the CI including zero on every split
```

### main.tex #51
Source of any new number: +0.119 already in paragraph

Before:
```latex
\texttt{tcn3k} matches the 357{,}570-parameter backbone on \Ti{} and beats the causal GP
4/4,
```
After:
```latex
\texttt{tcn3k} matches the 357{,}570-parameter backbone on \Ti{} with skill $+0.119$ over
the causal GP and the CI above zero on 4/4,
```

### main.tex #52
Source of any new number: wording only

Before:
```latex
\paragraph{The model wins on discharges where the target moves.}
```
After:
```latex
\paragraph{The model's per-discharge skill is positive where the target moves.}
```

### main.tex #53
Source of any new number: numbers already in sentence

Before:
```latex
\textbf{A tight interval is not the same as a general effect.} \Ti{} is the opposite: it
wins on 70\% of discharges and its single largest contributor accounts for only 9--16\%.
```
After:
```latex
\textbf{A tight interval is not the same as a general effect.} \Ti{} is the opposite: its
win rate is 0.70 and its single largest contributor accounts for only 9--16\%.
```

### main.tex #54
Source of any new number: numbers already in sentence

Before:
```latex
Of eleven covariates tested for predicting the win, one survives, and it is \emph{how much
the target moves within a discharge}. Split into terciles of that spread, \Ti{} wins
42\% / 83\% / 85\% and \Vrot{} wins 34\% / 48\% / 55\%.
```
After:
```latex
Of eleven covariates tested for predicting a positive per-discharge skill, one survives,
and it is \emph{how much the target moves within a discharge}. Split into terciles of that
spread, the \Ti{} win rate is 42\% / 83\% / 85\% and the \Vrot{} win rate 34\% / 48\% / 55\%.
```

### main.tex #55
Source of any new number: numbers already in sentence (no interval in the record)

Before:
```latex
  \item \textbf{The model's intervals beat both baselines' in every cell} (32/32
        population $\times$ target $\times$ variant $\times$ split) by the Winkler
        interval score, which penalises width and misses jointly:
```
After:
```latex
  \item \textbf{The model's intervals have a lower (better) Winkler score than both
        baselines' in every cell} (32/32 population $\times$ target $\times$ variant
        $\times$ split); the score penalises width and misses jointly:
```

### main.tex #56
Source of any new number: wording only

Before:
```latex
every aggregate skill and does not claim to beat interpolation on a typical row.
```
After:
```latex
every aggregate skill and does not claim a lower error than interpolation on a typical row.
```

### main.tex #57
Source of any new number: -0.040 from sec:bulk

Before:
```latex
one split with a significant MSE deficit that a longer cap does not remove
```
After:
```latex
one split with an MSE difference of $-0.040$ whose CI is below zero, which a longer cap does not remove
```

### main.tex #58
Source of any new number: tab:dual

Before:
```latex
reported beside, not inside, the aggregate model; for \Vrot{} that filter ties
persistence in the median at the photon floor of the 10\,ms line fit.
```
After:
```latex
reported beside, not inside, the aggregate model; for \Vrot{} that filter's median-error
ratio to persistence is 0.94--1.00 with the interval including 1 on 3/4 splits, at the
photon floor of the 10\,ms line fit.
```

### main.tex #59
Source of any new number: RMSE from tab:ladder; paired skill from sec:headline; causal GP from tab:headline

Before:
```latex
\textbf{The offline claim is bounded above by a GP tie.} The offline GP beats
PCHIP and the model only ties it (1/8 cells significant), so ``beats future-using
interpolation'' means ``beats the pre-registered interpolants''; it does not extend
to every offline method. It does extend to the causal GP, the strongest deployable
one.
```
After:
```latex
\textbf{The offline claim is bounded above by the offline GP.} The offline GP's
\Ti{} RMSE is below PCHIP's (153.8 vs.\ 173.6) and the model's paired skill against
it is $-0.05$ to $+0.11$ with the CI including zero on 7/8 cells, so ``skill over
future-using interpolation'' means ``skill over the pre-registered interpolants'';
it does not extend to every offline method. It does extend to the causal GP, the
strongest deployable one ($+0.08$ to $+0.17$, CIs above zero on 8/8).
```

### main.tex #60
Source of any new number: tab:headline, tab:gap, tab:dual, sec:headline

Before:
```latex
sees the future, and a causal GP that is the strongest method that does not. On the
observed population the model beats future-using PCHIP for ion temperature with
significant, replicated skill ($+0.17$ to $+0.32$ across four independent splits
and both populations; $[+0.19, +0.25]$ and $[+0.19, +0.31]$ under a pre-registered
repeated 5-fold replicate that tests every discharge once), ties the strongest
offline smoother, and beats every causal baseline for both targets, including the
causal GP on all eight cells and including the non-adjacent regime
($\Delta t > 15$\,ms). For rotation the replicate turns the four-split tie into a
registered mean-squared-error win over PCHIP, persistence and the causal GP in
both populations ($[+0.07, +0.70]$ cut vs.\ PCHIP), carried by the tail; in the
median \Vrot{} still ties persistence at the photon floor.
```
After:
```latex
sees the future, and a causal GP that is the strongest method that does not. On the
observed population the model's skill over future-using PCHIP for ion temperature is
$+0.17$ to $+0.32$ across four independent splits and both populations with every
CI excluding zero ($[+0.19, +0.25]$ and $[+0.19, +0.31]$ under a pre-registered
repeated 5-fold replicate that tests every discharge once), its paired skill against
the strongest offline smoother is $-0.05$ to $+0.11$ with the CI including zero on
7/8 cells, and its skill over every causal baseline is above zero for both targets
(ion temperature: CI above zero on all eight cells against the causal GP, $+0.08$ to
$+0.17$, and in the non-adjacent regime, $\Delta t > 15$\,ms, $+0.27$ and $+0.21$).
For rotation the replicate turns the four-split intervals that included zero into a
registered mean-squared-error interval above zero over PCHIP, persistence and the
causal GP in both populations ($[+0.07, +0.70]$ cut vs.\ PCHIP), carried by the
tail; in the median the \Vrot{} error ratio to persistence is still 0.94--1.00 with
the interval including 1 on 3/4 splits, at the photon floor.
```

### main.tex #61
Source of any new number: tab:mnar, tab:campaign

Before:
```latex
genuinely missing, in-domain timesteps the advantage over every causal method
survives reweighting on 8/8 population--split cells; across a campaign boundary
the sequence nowcaster stays ahead of PCHIP and of the causal GP on 4/4
initialisations in both populations, where
```
After:
```latex
genuinely missing, in-domain timesteps the skill over persistence, the causal
reference, survives reweighting at $+0.28$ to $+0.44$ with CIs above zero on 8/8
population--split cells; across a campaign boundary the sequence nowcaster keeps
$+0.17$ to $+0.20$ over PCHIP and $+0.11$ to $+0.16$ over the causal GP with CIs
above zero on 4/4 initialisations in both populations, where
```

### main.tex #62
Source of any new number: sec:deploy

Before:
```latex
points where it matters. It emits calibrated intervals that beat both baselines'
by a proper scoring rule in every cell.
```
After:
```latex
points where it matters. It emits calibrated intervals whose proper-scoring-rule
(Winkler) score is below both baselines' in every cell.
```

### main.tex #63
Source of any new number: tab:replicate, tab:dual

Before:
```latex
We also report where it does not work and why. Rotation beats interpolation
only in the mean, ties it in the median, and its skill lives in transients; the ion-temperature win is itself a
tail effect
```
After:
```latex
We also report where it does not work and why. Rotation's interval over
interpolation excludes zero only in the mean ($[+0.07, +0.70]$), its median-error
ratio to persistence is 0.94--1.00 with the interval including 1, and its skill lives
in transients; the ion-temperature skill is itself a
tail effect
```

### main.tex #64
Source of any new number: tab:gap

Before:
```latex
45\,ms the inclusive population is a tie. None of these is a ceiling argument, and
```
After:
```latex
45\,ms the inclusive-population interval includes zero ($-0.004$, $[-0.30, +0.25]$).
None of these is a ceiling argument, and
```

### main.tex #65
Source of any new number: tab:family, sec:family

Before:
```latex
a 26$\times$ width sweep is flat. So is the model-family axis: a shot-encoding
neural process, a transformer pretrained on another tokamak and a 4-init ensemble
all tie, and with no same-shot label no arm beats the training mean. Alongside these, the work contributes a
```
After:
```latex
a 26$\times$ width sweep is flat. So is the model-family axis: a shot-encoding
neural process, a transformer pretrained on another tokamak and a 4-init ensemble
all change \Ti{} skill by $-0.09$ to $+0.03$ with none clearing the 0.02 band in both
populations, and with no same-shot label every arm's skill against the training mean
is negative ($-0.003$ to $-0.040$). Alongside these, the work contributes a
```

## main_ko.tex (73 edits)

### main_ko.tex #1
Source of any new number: tab:headline

Before:
```latex
\item[주 결과] 나우캐스터는 \Ti{}에 대해 미래를 쓰는 PCHIP을
$\skillp = +0.17\sim+0.26$(스파이크 컷)과 $+0.23\sim+0.32$(스파이크 포함)로
이기며, \emph{두 모집단 각각의 독립적인 4개 held-out 분할 모두}에서 95\%
신뢰구간이 0을 제외하고, 8개 셀 전부에서 인과 GP도 이긴다. 모든 방전을 한 번씩
```
After:
```latex
\item[주 결과] 나우캐스터의 \Ti{} skill은 미래를 쓰는 PCHIP 대비
$\skillp = +0.17\sim+0.26$(스파이크 컷)과 $+0.23\sim+0.32$(스파이크 포함)로,
\emph{두 모집단 각각의 독립적인 4개 held-out 분할 모두}에서 95\%
신뢰구간이 0을 제외하고, 인과 GP 대비 skill은 $+0.08\sim+0.17$로 8개 셀 전부에서
신뢰구간이 0을 제외한다. 모든 방전을 한 번씩
```

### main_ko.tex #2
Source of any new number: sec:headline bullet

Before:
```latex
$[+0.19, +0.25]$(컷)·$[+0.19, +0.31]$(포함)이다. 가장 강한 미래 사용
평활기인 오프라인 GP와는 동률이다(\S\ref{sec:results}).
```
After:
```latex
$[+0.19, +0.25]$(컷)·$[+0.19, +0.31]$(포함)이다. 가장 강한 미래 사용
평활기인 오프라인 GP 대비 짝지은 skill은 $-0.05\sim+0.11$로 8개 셀 중 7개에서
신뢰구간이 0을 포함한다(\S\ref{sec:results}).
```

### main_ko.tex #3
Source of any new number: tab:dual

Before:
```latex
\item[회전] \Vrot{}은 네 분할의 관문을 1/4·2/4만 통과해 처음엔 동률로 적었으나,
재현의 등록된 두 단계(모델$\times$방전) bootstrap(새 분할 8벌)에서는 두 모집단
모두 평균 제곱 오차로 PCHIP·persistence·인과 GP를 이긴다(PCHIP 대비 컷
$[+0.07, +0.70]$)---꼬리가 실어 나르는 승리이며, 중앙값에서는 10\,ms 광자
바닥에서 persistence와 동률이다. 고속 진단을
```
After:
```latex
\item[회전] \Vrot{}은 네 분할의 관문을 1/4·2/4만 통과해 처음엔 구간이 0을 포함하는
것으로 적었으나, 재현의 등록된 두 단계(모델$\times$방전) bootstrap(새 분할 8벌)에서는
두 모집단 모두 PCHIP·persistence·인과 GP 대비 평균 제곱 오차 skill의 95\% 구간이
0을 제외한다(PCHIP 대비 컷 $[+0.07, +0.70]$)---꼬리가 실어 나르는 효과이며,
중앙값에서는 persistence 대비 오차 비가 0.94--1.00(물리 필터 중앙값)으로 3/4
분할에서 구간이 1을 포함한다(10\,ms 광자 바닥). 고속 진단을
```

### main_ko.tex #4
Source of any new number: tab:mnar, tab:campaign

Before:
```latex
\emph{진짜 결측} 지점의 공변량 분포로 재가중해도 모든 인과 방법 대비 우위는 전
분할에서 유지되고 PCHIP 대비 우위는 2/4(컷)와 4/4(포함)에서 유지된다. 캠페인을
가로질러 시간으로 엄격히 재분할하면---이는 윈도 대조군의 보간 대비 우위를
제거한다---시퀀스 나우캐스터는 두 모집단 모두에서 4/4 초기화에 대해 PCHIP과 인과
GP를 앞선다(\S\ref{sec:mnar}, \S\ref{sec:campaign}).
```
After:
```latex
\emph{진짜 결측} 지점의 공변량 분포로 재가중해도 인과 기준인 persistence 대비 skill은
$+0.28\sim+0.44$로 전 분할에서 신뢰구간이 0을 제외하고, PCHIP 대비 skill은
$+0.14\sim+0.28$로 2/4(컷)와 4/4(포함)에서 신뢰구간이 0을 제외한다. 캠페인을
가로질러 시간으로 엄격히 재분할하면---이는 윈도 대조군의 보간 대비 우위를
제거한다---시퀀스 나우캐스터는 두 모집단 모두에서 PCHIP 대비 $+0.17\sim+0.20$, 인과
GP 대비 $+0.11\sim+0.16$으로 4/4 초기화에서 신뢰구간이 0을 제외한다(\S\ref{sec:mnar},
\S\ref{sec:campaign}).
```

### main_ko.tex #5
Source of any new number: sec:deploy

Before:
```latex
$+0.45\sim+0.72$, 8/8 PASS), 보정된 conformal 구간은 32개 모집단--타겟--변형--분할
셀 전부에서 두 기준선의 구간을 이긴다(\S\ref{sec:peak}, \S\ref{sec:deploy}).
```
After:
```latex
$+0.45\sim+0.72$, 8/8 PASS), 보정된 conformal 구간은 32개 모집단--타겟--변형--분할
셀 전부에서 두 기준선보다 낮은(좋은) Winkler 점수를 갖는다(\S\ref{sec:peak}, \S\ref{sec:deploy}).
```

### main_ko.tex #6
Source of any new number: tab:dual

Before:
```latex
분할--모집단 칸 모두 persistence의 0.91$\times$에 앉고 \Vrot{} 중앙값에서도
백본을 이긴다. 집계도 이겨야 하는 출력은
```
After:
```latex
분할--모집단 칸 모두 persistence의 0.91$\times$에 앉고 \Vrot{} 중앙값에서도
백본 대비 skill이 $+0.09\sim+0.27$로 4/4 분할에서 신뢰구간이 0을 제외한다. 집계도
이겨야 하는 출력은
```

### main_ko.tex #7
Source of any new number: sec:context

Before:
```latex
비율이 0.52에서 0.66으로 오르는 반면, 같은 도달 범위에서 순환·확장
합성곱·attention은 0.023 이내로 동률이다---아키텍처는
```
After:
```latex
비율이 0.52에서 0.66으로 오르는 반면, 같은 도달 범위에서 순환·확장
합성곱·attention의 차이는 최대 0.023이다---아키텍처는
```

### main_ko.tex #8
Source of any new number: tab:headline, tab:replicate, tab:dual

Before:
```latex
\textbf{답 셋.} \emph{하나}, \Ti{}는 이긴다---미래를 쓰는 PCHIP 대비 skill
$+0.17\sim+0.32$, 두 모집단의 4개 분할 여덟 셀 전부에서 신뢰구간이 0을 제외하고,
배치 가능한 가장 강한 기준선인 인과 GP도 8/8로 이긴다(\S\ref{sec:results}).
\emph{둘}, \Vrot{}은 네 분할 관문을 1/4·2/4만 통과해 동률로 적었고, 모든 방전을
한 번씩 채점하는 사전등록 재현에서야 평균 제곱 오차의 우위가 등록된다(PCHIP 대비
컷 $[+0.07, +0.70]$; 중앙값은 광자 바닥에서 동률). 승리든 동률이든 튜닝 부족이
아니다.
```
After:
```latex
\textbf{답 셋.} \emph{하나}, \Ti{}의 skill은 미래를 쓰는 PCHIP 대비
$+0.17\sim+0.32$로 두 모집단의 4개 분할 여덟 셀 전부에서 신뢰구간이 0을 제외하고,
배치 가능한 가장 강한 기준선인 인과 GP 대비로도 $+0.08\sim+0.17$로 8/8에서
신뢰구간이 0을 제외한다(\S\ref{sec:results}).
\emph{둘}, \Vrot{}은 네 분할 관문을 1/4·2/4만 통과해 구간이 0을 포함하는 것으로
적었고, 모든 방전을 한 번씩 채점하는 사전등록 재현에서야 평균 제곱 오차 skill의
구간이 0을 제외한다(PCHIP 대비 컷 $[+0.07, +0.70]$; 중앙값은 persistence 대비 오차
비 0.94--1.00으로 구간이 1을 포함). 어느 쪽이든 튜닝 부족이
아니다.
```

### main_ko.tex #9
Source of any new number: tab:family, sec:bulk

Before:
```latex
사전학습 transformer·4-초기화 앙상블로 바꿔도 동률이다(\S\ref{sec:family}).
다만 그 \Ti{} 우위는 꼬리에서 온다: 오차 상위 사분위에서 이기고 중앙값 행에서는
persistence와 같다(\S\ref{sec:bulk}).
```
After:
```latex
사전학습 transformer·4-초기화 앙상블로 바꿔도 \Ti{} skill 변화는 $-0.09\sim+0.03$으로
어느 팔도 두 모집단 모두에서 0.02 띠를 넘지 못한다(\S\ref{sec:family}).
다만 그 \Ti{} skill은 꼬리에서 온다: 오차 상위 사분위에서 오차 비가 1 아래로
내려가고 중앙값 행에서는 persistence와 같다(\S\ref{sec:bulk}).
```

### main_ko.tex #10
Source of any new number: sec:headline, tab:dual

Before:
```latex
(표~\ref{tab:falsify}), 그중 둘은 실제로 걸렸다---회전은 네 분할 관문을 통과하지 못해
동률로 강등되었고(재현에서 평균 우위가 등록된 뒤에도 중앙값 동률은 남는다), 오프라인 GP와의 동률 때문에 ``미래를 쓰는 보간을 이긴다''는
``모든 오프라인 방법을 이긴다''로 넓혀 적지 않는다.
```
After:
```latex
(표~\ref{tab:falsify}), 그중 둘은 실제로 걸렸다---회전은 네 분할 관문을 통과하지 못해
구간이 0을 포함하는 것으로 강등되었고(재현에서 평균 skill의 구간이 0을 제외한 뒤에도
중앙값의 persistence 대비 오차 비 0.94--1.00은 남는다), 오프라인 GP 대비 짝지은 skill
$-0.05\sim+0.11$의 구간이 8셀 중 7셀에서 0을 포함하므로 ``미래를 쓰는 보간 대비
skill이 0을 넘는다''를 ``모든 오프라인 방법 대비 0을 넘는다''로 넓혀 적지 않는다.
```

### main_ko.tex #11
Source of any new number: tab:replicate, tab:dual

Before:
```latex
   인과 GP 8/8. \quad \Vrot: 1/4·2/4 --- \textbf{동률}; 반복 5겹 재현에서 평균
   우위 등록, 중앙값 동률.};
```
After:
```latex
   인과 GP 8/8. \quad \Vrot: 1/4·2/4 --- \textbf{구간이 0 포함}; 반복 5겹 재현에서
   평균 skill $[+0.07, +0.70]$(0 제외), 중앙값 오차 비 0.94--1.00(1 포함).};
```

### main_ko.tex #12
Source of any new number: wording only

Before:
```latex
   ① 관측 지점에서만 잰 skill(MNAR) ② 캠페인 이동 ③ 오프라인 GP 동률 ④ 추정기
```
After:
```latex
   ① 관측 지점에서만 잰 skill(MNAR) ② 캠페인 이동 ③ 오프라인 GP 대비 구간 0 포함 ④ 추정기
```

### main_ko.tex #13
Source of any new number: tab:replicate, tab:dual

Before:
```latex
  & \textbf{걸렸다.} 1/4·2/4 $\rightarrow$ 동률; 재현(새 분할 8벌)에서 평균 우위 등록, 중앙값 동률. 고속 진단 0에도 출력 비트 동일 \\
```
After:
```latex
  & \textbf{걸렸다.} 1/4·2/4 $\rightarrow$ 구간이 0 포함; 재현(새 분할 8벌)에서 평균 skill $[+0.07, +0.70]$(0 제외), 중앙값 오차 비 0.94--1.00(1 포함). 고속 진단 0에도 출력 비트 동일 \\
```

### main_ko.tex #14
Source of any new number: sec:headline

Before:
```latex
  & \textbf{걸렸다.} 오프라인 GP와 동률(8셀 중 1개 유의) $\rightarrow$ 주장 축소 \\
```
After:
```latex
  & \textbf{걸렸다.} 오프라인 GP 대비 $-0.05\sim+0.11$, 8셀 중 7셀에서 구간이 0 포함 $\rightarrow$ 주장 축소 \\
```

### main_ko.tex #15
Source of any new number: tab:headline, sec:headline

Before:
```latex
  \item \textbf{\Ti{}에 대한 강건한 긍정 결과} (\S\ref{sec:results}): 인과
        나우캐스터가 미래를 쓰는 PCHIP을 $\skillp = +0.17\sim+0.26$(컷)과
        $+0.23\sim+0.32$(포함)로 이기며, 두 모집단의 4개 held-out 분할 전부에서
        CI가 0을 제외하고, 8개 분할--모집단 셀 전부에서 인과 GP도 이긴다. 가장
        강한 미래 사용 평활기인 오프라인 GP와는 동률이다. 모든 방전을 한 번씩
```
After:
```latex
  \item \textbf{\Ti{}에 대한 강건한 긍정 결과} (\S\ref{sec:results}): 인과
        나우캐스터의 미래를 쓰는 PCHIP 대비 skill은 $\skillp = +0.17\sim+0.26$(컷)과
        $+0.23\sim+0.32$(포함)로 두 모집단의 4개 held-out 분할 전부에서
        CI가 0을 제외하고, 인과 GP 대비 skill은 $+0.08\sim+0.17$로 8개 분할--모집단
        셀 전부에서 CI가 0을 제외한다. 가장 강한 미래 사용 평활기인 오프라인 GP
        대비 짝지은 skill은 $-0.05\sim+0.11$로 8셀 중 7셀에서 CI가 0을 포함한다.
        모든 방전을 한 번씩
```

### main_ko.tex #16
Source of any new number: tab:dual

Before:
```latex
        \Vrot{}은 관문을 1/4·2/4만 통과해(잡음만으로도 나올 수준) 처음엔 동률로
        기록했다. 반복 5겹 재현을 등록된 두 단계 bootstrap(새 분할 8벌)으로 읽으면
        PCHIP·persistence·인과 GP 대비 평균 제곱 오차의 우위가 두 모집단 모두에서
        실재하고(PCHIP 대비 컷 $[+0.07, +0.70]$, 포함 $[+0.05, +0.68]$) 꼬리가
        실어 나르며, 중앙값에서는 10\,ms 광자 바닥에서 persistence와
        동률이다(\S\ref{sec:physfilter}). 이 우위는 이력만의 것이다:
```
After:
```latex
        \Vrot{}은 관문을 1/4·2/4만 통과해(잡음만으로도 나올 수준) 처음엔 구간이
        0을 포함하는 것으로 기록했다. 반복 5겹 재현을 등록된 두 단계 bootstrap(새
        분할 8벌)으로 읽으면 PCHIP·persistence·인과 GP 대비 평균 제곱 오차 skill의
        구간이 두 모집단 모두에서 0을 제외하고(PCHIP 대비 컷 $[+0.07, +0.70]$, 포함
        $[+0.05, +0.68]$) 꼬리가 실어 나르며, 중앙값에서는 persistence 대비 오차 비가
        0.94--1.00으로 3/4 분할에서 구간이 1을 포함한다---10\,ms 광자
        바닥이다(\S\ref{sec:physfilter}). 이 skill은 이력만의 것이다:
```

### main_ko.tex #17
Source of any new number: tab:mnar

Before:
```latex
        (\S\ref{sec:mnar}, \S\ref{sec:campaign}). 채점 지점을 진짜 결측 지점의
        공변량 분포로 재가중해도 \Ti{} 우위는 두 모집단 모두에서 모든 인과 방법
        대비 4/4 분할, PCHIP 대비 2/4(컷)와 4/4(포함)로 유지된다. 캠페인을
```
After:
```latex
        (\S\ref{sec:mnar}, \S\ref{sec:campaign}). 채점 지점을 진짜 결측 지점의
        공변량 분포로 재가중해도 \Ti{}의 인과 기준 persistence 대비 skill은
        $+0.28\sim+0.44$로 두 모집단 모두 4/4 분할에서 CI가 0을 제외하고, PCHIP 대비
        skill은 $+0.14\sim+0.28$로 2/4(컷)와 4/4(포함)에서 CI가 0을 제외한다. 캠페인을
```

### main_ko.tex #18
Source of any new number: tab:campaign

Before:
```latex
        통계로 표준화하는 시퀀스 나우캐스터는 두 모집단 모두에서 4/4 초기화에
        대해 PCHIP과 인과 GP를 앞선다.
```
After:
```latex
        통계로 표준화하는 시퀀스 나우캐스터는 두 모집단 모두에서 PCHIP 대비
        $+0.17\sim+0.20$, 인과 GP 대비 $+0.11\sim+0.16$으로 4/4 초기화에서 CI가 0을
        제외한다.
```

### main_ko.tex #19
Source of any new number: tab:ladder2

Before:
```latex
        skill과 대등하고(짝지은 평균 $+0.002$)---컷이 없으면 $-0.19$를 잃는데,
```
After:
```latex
        skill과 대등하고(짝지은 평균 $+0.002$)---컷이 없으면 $-0.19$ 아래에
        놓이는데(4/4에서 CI가 0 미만),
```

### main_ko.tex #20
Source of any new number: sec:context

Before:
```latex
        이기는 방전의 비율이 0.52에서 0.66으로 오르는 반면, 도달 범위를 맞추면
        순환·확장 합성곱·attention의 110$\times$ 파라미터 범위가 0.023 이내로
        동률이다. 문맥은 승리를 전형적으로 만들고,
```
After:
```latex
        이기는 방전의 비율이 0.52에서 0.66으로 오르는 반면, 도달 범위를 맞추면
        순환·확장 합성곱·attention의 110$\times$ 파라미터 범위가 최대 0.023 차이에
        그친다. 문맥은 양의 방전별 skill을 전형적으로 만들고,
```

### main_ko.tex #21
Source of any new number: sec:deploy

Before:
```latex
        않고 32개 모집단--타겟--변형--분할 셀 전부에서 두 기준선의 구간을 이기는
        분포 무가정 conformal 구간,
```
After:
```latex
        않고 32개 모집단--타겟--변형--분할 셀 전부에서 두 기준선보다 낮은 Winkler
        점수를 내는 분포 무가정 conformal 구간,
```

### main_ko.tex #22
Source of any new number: numbers already in sentence

Before:
```latex
72--78\%, persistence 대비 skill의 87--90\%를 유지하고, 증분 형태의 식을 모든
분할에서 이기며(paired $+0.034 \sim +0.042$, CI가 0 위),
```
After:
```latex
72--78\%, persistence 대비 skill의 87--90\%를 유지하고, 증분 형태의 식보다 모든
분할에서 paired $+0.034 \sim +0.042$만큼 위에 있으며(CI가 0 위),
```

### main_ko.tex #23
Source of any new number: tab:ladder, sec:headline

Before:
```latex
\subsection{나우캐스터는 모든 인과 기준선을 압도한다}
표~\ref{tab:ladder}는 표준 test 분할에서의 물리 단위 RMSE 사다리를 보여준다.
\seqv{}는 미래를 읽지 않는 모든 방법 가운데 두 타겟 모두에서 최저 RMSE를 가지며,
배치 가능한 가장 강한 팔인 인과 GP를 RMSE 기준 4\%(\Ti{})와 18\%(\Vrot{}) 차로
이기고, 미래를 쓰는 보간들도 앞선다. 유일하게 동률인 팔은 잡음 앵커를 통과하는
대신 평균해 내는 오프라인 GP다(\Ti{} 153.8 대 157.8). 이는 본 논문에서 가장
강건한 결과다: 미래 CES가 정의상 없는 모든 온라인 환경에서 시퀀스 나우캐스터는
명백한 승자다.
```
After:
```latex
\subsection{나우캐스터는 모든 인과 기준선 가운데 RMSE가 가장 낮다}
표~\ref{tab:ladder}는 표준 test 분할에서의 물리 단위 RMSE 사다리를 보여준다.
\seqv{}는 미래를 읽지 않는 모든 방법 가운데 두 타겟 모두에서 최저 RMSE를 가지며,
배치 가능한 가장 강한 팔인 인과 GP보다 RMSE가 4\%(\Ti{})와 18\%(\Vrot{}) 낮고,
미래를 쓰는 보간들보다도 낮다. \Ti{} RMSE가 더 낮은 유일한 팔은 잡음 앵커를
통과하는 대신 평균해 내는 오프라인 GP다(\Ti{} 153.8 대 157.8; 짝지은 skill
$-0.05\sim+0.11$, 8셀 중 7셀에서 CI가 0을 포함, \S\ref{sec:headline}). 이는 본
논문에서 가장 강건한 결과다: 미래 CES가 정의상 없는 모든 온라인 환경에서 시퀀스
나우캐스터의 오차가 가장 낮다.
```

### main_ko.tex #24
Source of any new number: tab:headline

Before:
```latex
\subsection{헤드라인: \Ti{}는 두 모집단 모두에서 4개 독립 분할에 걸쳐 미래를 쓰는
보간을 이긴다}
```
After:
```latex
\subsection{헤드라인: \Ti{}의 미래를 쓰는 보간 대비 skill은 두 모집단 모두에서 4개
독립 분할에 걸쳐 CI가 0을 제외한다}
```

### main_ko.tex #25
Source of any new number: numbers already in sentence

Before:
```latex
        shot 군집 CI가 0을 제외한다. 8개 모집단--분할 셀 전부가 \emph{인과}
        GP($+0.08\sim+0.17$)와 persistence($+0.36\sim+0.46$)도 이긴다. 이것이
        본 논문의 무조건적 주장이다. 오프라인 GP를 상대로는 백본이
        \emph{동률}이므로(점추정 $-0.05\sim+0.11$, 8개 중 1개 셀에서 유의),
        관측 모집단에 대한 주장은 ``사전등록된 보간들과 배치 가능한 모든 인과
        방법을 이기고, 미래를 쓰며 표본별로 튜닝된 평활기와는 대등하다''로
        읽힌다.
```
After:
```latex
        shot 군집 CI가 0을 제외한다. 8개 모집단--분할 셀 전부에서 \emph{인과}
        GP 대비($+0.08\sim+0.17$)와 persistence 대비($+0.36\sim+0.46$) skill도 CI가
        0을 넘는다. 이것이 본 논문의 무조건적 주장이다. 오프라인 GP를 상대로는
        백본의 짝지은 skill이 $-0.05\sim+0.11$로 8개 중 7개 셀에서 CI가 0을
        포함하므로, 관측 모집단에 대한 주장은 ``사전등록된 보간들과 배치 가능한
        모든 인과 방법 대비 skill이 0을 넘고, 미래를 쓰며 표본별로 튜닝된
        평활기 대비로는 구간이 0을 포함한다''로 읽힌다.
```

### main_ko.tex #26
Source of any new number: tab:headline / tab:replicate

Before:
```latex
  \item \textbf{\Vrot{}: 네 분할 셈에서는 동률, 재현에서는 등록된 승리.} 두
```
After:
```latex
  \item \textbf{\Vrot{}: 네 분할 셈에서는 구간이 0을 포함, 재현에서는 등록된
        구간이 0을 제외.} 두
```

### main_ko.tex #27
Source of any new number: numbers already in sentence

Before:
```latex
        방전 단위로 채점하면 \Vrot{}은 등록된 두 단계 bootstrap 아래 PCHIP(컷
        $[+0.07, +0.70]$, 포함 $[+0.05, +0.68]$)·persistence($[+0.28, +0.76]$,
        $[+0.14, +0.75]$)·인과 GP($[+0.05, +0.70]$, $[+0.02, +0.68]$)를 이기고, 새
        모델 8벌이 전부 양수다
```
After:
```latex
        방전 단위로 채점하면 등록된 두 단계 bootstrap 아래 \Vrot{}의 skill은 PCHIP
        대비 $[+0.07, +0.70]$(컷)·$[+0.05, +0.68]$(포함), persistence 대비
        $[+0.28, +0.76]$·$[+0.14, +0.75]$, 인과 GP 대비 $[+0.05, +0.70]$·$[+0.02,
        +0.68]$로 모든 구간이 0을 제외하고, 새 모델 8벌이 전부 양수다
```

### main_ko.tex #28
Source of any new number: tab:dual

Before:
```latex
        좌우되기 때문이다. 이 승리는 평균 제곱 오차의, 꼬리가 실어 나르는
        승리이며(\S\ref{sec:bulk}), 중앙값에서 \Vrot{}은 10\,ms 광자 바닥에서
        persistence와 동률이다(\S\ref{sec:physfilter}). 두 진술을 함께 적는다.
```
After:
```latex
        좌우되기 때문이다. 이 효과는 평균 제곱 오차의, 꼬리가 실어 나르는
        효과이며(\S\ref{sec:bulk}), 중앙값에서 \Vrot{}의 persistence 대비 오차 비는
        0.94--1.00으로 3/4 분할에서 구간이 1을 포함한다---10\,ms 광자
        바닥이다(\S\ref{sec:physfilter}). 두 진술을 함께 적는다.
```

### main_ko.tex #29
Source of any new number: numbers already in sentence

Before:
```latex
\Vrot{} 손실을 보이지 않았다(8/16은 유의한 승리). 네 조건이 모두 성립했다.
```
After:
```latex
\Vrot{} 손실을 보이지 않았다(8/16은 CI가 0을 넘는다). 네 조건이 모두 성립했다.
```

### main_ko.tex #30
Source of any new number: +0.041 (1/4) from sec:context / THESIS sec. 8am; tab:headline

Before:
```latex
이 관문은 프레이밍이 \emph{무엇을} 사는지도 확정한다. 윈도 대조군은 인과 GP와
동률인 반면(컷 기준 1/4 분할 유의) 시퀀스 백본은 두 모집단 모두에서 4/4로
이긴다: 세그먼트의
```
After:
```latex
이 관문은 프레이밍이 \emph{무엇을} 사는지도 확정한다. 윈도 대조군의 인과 GP 대비
skill은 $+0.041$로 컷 기준 3/4 분할에서 CI가 0을 포함하는 반면, 시퀀스 백본은
$+0.08\sim+0.17$로 두 모집단 모두 4/4에서 CI가 0을 넘는다: 세그먼트의
```

### main_ko.tex #31
Source of any new number: tab:gap; persistence CIs from paper_numbers.json largegap_pooled

Before:
```latex
\paragraph{\Ti{}의 승리는 인접 이력에 국한되지 않는다.} 15\,ms를 넘어서도 백본은
두 모집단 모두에서 미래를 쓰는 PCHIP을 이기고($+0.268$과 $+0.206$, 두 CI 모두 0
위), 거기서 persistence는 $+0.40$과 $+0.43$으로 이긴다. 45\,ms를 넘어서면 컷에서는
여전히 이기지만($+0.267$) 포함 모집단에서는 \emph{동률}이다($-0.004$, CI
$[-0.30, +0.25]$): 101개 shot의 429행은 소수의 스파이크 앵커 행이 한 계층을
지배할 수 있는 바로 그 지점이며, 우리는 이를 동률로 보고한다.

\paragraph{\Vrot{}은 보간에게 가장 어려운 그 한 곳에서 보간을 이긴다.} 전역
\Vrot{} 동률은 인접 영역에서의 동률이다. $\Delta t > 15$\,ms에서는 백본이 두
모집단 모두에서 PCHIP을 이기며($+0.418$과 $+0.432$, 130개 shot) --- 본 논문에서
유일한 무조건적 \Vrot{} 긍정 판정이다 --- $\leq 15$\,ms에서는 persistence를
이긴다($+0.39$). 가장 넓은 구간은 14행뿐이라 채점하지 않는다.
```
After:
```latex
\paragraph{\Ti{}의 skill은 인접 이력에 국한되지 않는다.} 15\,ms를 넘어서도 백본의
미래를 쓰는 PCHIP 대비 skill은 두 모집단에서 $+0.268$과 $+0.206$이고(두 CI 모두 0
위), 거기서 persistence 대비 skill은 $+0.40$과 $+0.43$이다(CI $[+0.34, +0.45]$,
$[+0.35, +0.52]$). 45\,ms를 넘어서면 PCHIP 대비 skill은 컷에서 $+0.267$
$[+0.09, +0.41]$, 포함 모집단에서 $-0.004$(CI $[-0.30, +0.25]$)다: 101개 shot의
429행은 소수의 스파이크 앵커 행이 한 계층을 지배할 수 있는 바로 그 지점이며,
우리는 이 0을 포함하는 구간을 그대로 보고한다.

\paragraph{\Vrot{}의 보간 대비 skill은 보간에게 가장 어려운 그 한 곳에서 0을
제외한다.} 0을 포함하는 전역 \Vrot{} 구간은 인접 영역의 것이다. $\Delta t >
15$\,ms에서는 백본의 PCHIP 대비 skill이 두 모집단에서 $+0.418$ $[+0.10, +0.68]$과
$+0.432$ $[+0.13, +0.70]$이고(130개 shot) --- 본 논문에서 유일하게 0을 제외하는
무조건적 \Vrot{} 구간이다 --- $\leq 15$\,ms에서는 persistence 대비 skill이 $+0.39$
$[+0.21, +0.45]$다. 가장 넓은 구간은 14행뿐이라 채점하지 않는다.
```

### main_ko.tex #32
Source of any new number: tab:campaign

Before:
```latex
그리고 이것이 이 절의 결과다: \textbf{시퀀스 나우캐스터는 붕괴하지 않는다}. 두
모집단 모두에서 4/4 초기화에 대해 PCHIP을 이기고($+0.17\sim+0.20$), 두 모집단
모두에서 4/4로 인과 GP를 이기며($+0.11\sim+0.16$), 8/8에서 윈도 대조군을 이긴다.
\Vrot{}은 윈도 대조군이 0/4인 자리에서 두 모집단 모두 4/4로 persistence를
이기고($+0.31\sim+0.34$), 대조군 대비 \Vrot{} 마진은 8/8에서
유의하다($+0.06\sim+0.18$).
```
After:
```latex
그리고 이것이 이 절의 결과다: \textbf{시퀀스 나우캐스터는 붕괴하지 않는다}.
PCHIP 대비 skill은 $+0.17\sim+0.20$으로 두 모집단 모두 4/4 초기화에서 CI가 0을
넘고, 인과 GP 대비 skill은 $+0.11\sim+0.16$(두 모집단 모두 4/4에서 CI가 0을 넘음),
윈도 대조군 대비 짝지은 skill은 $+0.09\sim+0.18$(8/8에서 CI가 0을 넘음)이다.
\Vrot{}의 persistence 대비 skill은 $+0.31\sim+0.34$로 두 모집단 모두 4/4에서 CI가
0을 넘는데, 윈도 대조군의 구간은 4/4에서 0을 포함한다. 대조군 대비 \Vrot{} 마진은
$+0.06\sim+0.18$로 8/8에서 CI가 0을 넘는다.
```

### main_ko.tex #33
Source of any new number: tab:stress

Before:
```latex
표~\ref{tab:stress}는 채택된 모델에 대해 둘을 나란히 놓는다. 실시간 시스템이
실제로 돌릴 수 있는 기준선들에 대한 우위는 두 모집단 모두에서 두 스트레스
테스트를 모두 견딘다. \emph{미래를 쓰는} 보간에 대한 우위는 두 모집단 모두에서
시간 이동을 견디고, MNAR 재가중은 모집단 조건부 유의성으로 견딘다.
```
After:
```latex
표~\ref{tab:stress}는 채택된 모델에 대해 둘을 나란히 놓는다. 실시간 시스템이
실제로 돌릴 수 있는 기준선들에 대한 skill은 두 모집단 모두에서 두 스트레스
테스트를 CI가 0을 넘는 채로 견딘다(재가중에서 persistence 대비 $+0.28\sim+0.44$,
캠페인 분할에서 인과 GP 대비 $+0.11\sim+0.16$). \emph{미래를 쓰는} 보간에 대한
skill은 두 모집단 모두에서 시간 이동을 견디고($+0.17\sim+0.20$, CI가 0을 넘음),
MNAR 재가중은 모집단 조건부 유의성으로 견딘다($+0.14\sim+0.28$, 컷 2/4·포함
4/4에서 CI가 0을 넘음).
```

### main_ko.tex #34
Source of any new number: numbers already in sentence

Before:
```latex
  \item \textbf{포함 모집단에서는 같은 이력 전용 모델이 여전히 PCHIP을
        $+0.15\sim+0.23$로 이기며}, 고속 채널이
```
After:
```latex
  \item \textbf{포함 모집단에서는 같은 이력 전용 모델의 PCHIP 대비 skill이
        여전히 $+0.15\sim+0.23$이며}, 고속 채널이
```

### main_ko.tex #35
Source of any new number: wording only

Before:
```latex
비승리는 모델의 실패가 아니라 진단 정보량에 관한 발견이며,
```
After:
```latex
영결과는 모델의 실패가 아니라 진단 정보량에 관한 발견이며,
```

### main_ko.tex #36
Source of any new number: tab:ladder2; paper_numbers.json ladder/b3k8

Before:
```latex
        persistence로의 압축을 견딘다.} \bthree{}는 \Vrot{} 손실 없이 4/4
        분할에서 재학습된 앵커를 이기고, 백본에 대한 짝지은 \Ti{} skill은
        $-0.009$, $-0.005$, $+0.026$, $-0.004$이다(평균 $+0.002$, 모든 CI가 0을
        포함). PCHIP 대비 PR4를 4/4로 통과하고 인과 GP도 4/4로 이긴다. TEST
```
After:
```latex
        persistence로의 압축을 견딘다.} \bthree{}의 재학습된 앵커 대비 짝지은
        skill은 $+0.35\sim+0.42$로 4/4 분할에서 CI가 0을 넘고 \Vrot{} 손실이 없으며,
        백본에 대한 짝지은 \Ti{} skill은 $-0.009$, $-0.005$, $+0.026$, $-0.004$이다
        (평균 $+0.002$, 모든 CI가 0을 포함). PCHIP 대비 PR4를 4/4로 통과하고 인과 GP
        대비 skill은 $+0.07\sim+0.16$으로 4/4에서 CI가 0을 넘는다. TEST
```

### main_ko.tex #37
Source of any new number: tab:headline

Before:
```latex
        \emph{지역적}이다: 네 분할 셈으로는 전역적으로 \Vrot{}이 보간과
        동률이지만(재현의 합산 판독은 뒤에 평균 우위를 등록한다,
        \S\ref{sec:headline}), 매끄러운
```
After:
```latex
        \emph{지역적}이다: 네 분할 셈으로는 전역 \Vrot{} 구간이 3/4·2/4 분할에서
        0을 포함하지만(재현의 합산 판독은 뒤에 평균 skill을 0 위에 둔다,
        \S\ref{sec:headline}), 매끄러운
```

### main_ko.tex #38
Source of any new number: numbers already in sentence

Before:
```latex
\Ti{} 산포가 가장 작은 TEST 방전이다. 이 산포는 모델이 이기는 방전을 예측하는
방전별 공변량이다(분할 합산, 인과 GP 대비 방전별 skill과의 Spearman $\rho = +0.40$). \Ti{}가 움직이는 방전(산포 220\,eV)에서 나우캐스트는 PCHIP을
MSE $+0.42$로 이기고 인과 GP와는 동률이다($+0.03$). 움직이지 않는 방전(산포
34\,eV, \#31089)에서는 마지막 표본의 광자 잡음을 따라가 둘 모두에 진다($-1.06$,
$-0.73$). 이것이
```
After:
```latex
\Ti{} 산포가 가장 작은 TEST 방전이다. 이 산포는 모델의 방전별 skill이 양수인 방전을
예측하는 방전별 공변량이다(분할 합산, 인과 GP 대비 방전별 skill과의 Spearman $\rho = +0.40$). \Ti{}가 움직이는 방전(산포 220\,eV)에서 나우캐스트의 MSE skill은
PCHIP 대비 $+0.42$, 인과 GP 대비 $+0.03$이다(방전 하나의 점추정, 구간 없음). 움직이지
않는 방전(산포 34\,eV, \#31089)에서는 마지막 표본의 광자 잡음을 따라가 두 기준선 대비
skill이 $-1.06$, $-0.73$이다. 이것이
```

### main_ko.tex #39
Source of any new number: wording only

Before:
```latex
$+0.24$''는 그 자체로 \emph{전형적인} 행에서 모델이 보간보다 낫다는 뜻이 아니다.
```
After:
```latex
$+0.24$''는 그 자체로 \emph{전형적인} 행에서 모델의 오차가 보간보다 작다는 뜻이 아니다.
```

### main_ko.tex #40
Source of any new number: wording only

Before:
```latex
\paragraph{\Ti{} 우위는 꼬리에서 온다.} PCHIP 대비 $M_1$은 모집단 $\times$ 타겟
```
After:
```latex
\paragraph{\Ti{} skill은 꼬리에서 온다.} PCHIP 대비 $M_1$은 모집단 $\times$ 타겟
```

### main_ko.tex #41
Source of any new number: tab:bulk

Before:
```latex
$-0.562$). 행의 절반에서 모델은 PCHIP와 동률이거나 못하며, 오차 비율
모델/PCHIP의 분위수 사다리는 q50에서 1.20, q70에서 1.01, q75에서 0.96, q99에서
0.78로 내려간다 --- 우위는 오차 상위 사분위, 즉 피팅 실패·천이·ELM 행에서
시작한다.
```
After:
```latex
$-0.562$). 행의 절반에서 모델의 오차는 PCHIP 이상이며(승률 0.48), 오차 비율
모델/PCHIP의 분위수 사다리는 q50에서 1.20, q70에서 1.01, q75에서 0.96, q99에서
0.78로 내려간다 --- 오차 비가 1 아래로 내려가는 것은 오차 상위 사분위, 즉 피팅
실패·천이·ELM 행에서 시작한다.
```

### main_ko.tex #42
Source of any new number: numbers already in sentence

Before:
```latex
음수이고, 중앙값 행에서는 persistence보다도 나쁘다(4.3 대 3.7\,km/s). 집계
동률은 본류의 손실을 꼬리의 이득(peak 계층 $+0.54\sim+0.79$, \S\ref{sec:peak})이
상쇄한 결과이며,
```
After:
```latex
음수이고, 중앙값 행에서는 persistence보다도 나쁘다(4.3 대 3.7\,km/s). 0을
포함하는 집계 구간은 본류의 손실을 꼬리의 이득(peak 계층 $+0.54\sim+0.79$,
\S\ref{sec:peak})이 상쇄한 결과이며,
```

### main_ko.tex #43
Source of any new number: Student-t MSE from THESIS_RESULTS.md B.14 table

Before:
```latex
3/4) 내리면서 집계 MSE skill은 동률이다(짝지은 $-0.004$ / $-0.007$, 유의 열세
0/4). 중앙값에서 persistence를 이기고($+0.058$, 3/4) 인과 GP와 동률인 첫 팔이며,
PCHIP와의 중앙값 격차는 $-0.198$에서 $-0.101$로 반감한다. Student-$t$는 모든
지표에서 Huber 뒤에 있고 \Ti{} MSE를 한 분할에서 유의하게 잃는다: 재하강
```
After:
```latex
3/4) 내리면서 집계 MSE skill은 짝지은 $-0.004$ / $-0.007$로 구간이 0을
포함한다(유의 열세 0/4). 중앙값의 persistence 대비 skill이 $+0.058$로 3/4 분할에서
CI가 0을 넘고 인과 GP 대비 중앙값 구간이 0을 포함하는 첫 팔이며, PCHIP와의 중앙값
격차는 $-0.198$에서 $-0.101$로 반감한다. Student-$t$는 모든 지표에서 Huber 뒤에
있고 백본 대비 \Ti{} MSE는 $-0.018$ $[-0.033, -0.001]$로 1/4 분할에서 개별 CI가
0 아래다: 재하강
```

### main_ko.tex #44
Source of any new number: paper_numbers.json b14_select

Before:
```latex
칸을 강화한다: Huber \Ti{} $M_1 = +0.128$ $[+0.114, +0.141]$(4/4), MSE는 동률(풀링
$+0.007$, 유의 열세 0/4), \Ti{} 중앙값은 persistence의 0.894$\times$로 물리 필터가
닿는 띠에 들고(\S\ref{sec:physfilter}), 중앙값에서 인과 GP를 이긴다($+0.073$, CI
$>0$).
```
After:
```latex
칸을 강화한다: Huber \Ti{} $M_1 = +0.128$ $[+0.114, +0.141]$(4/4), MSE는 풀링
$+0.007$로 구간이 0을 포함(유의 열세 0/4), \Ti{} 중앙값은 persistence의 0.894$\times$로
물리 필터가 닿는 띠에 들고(\S\ref{sec:physfilter}), 인과 GP 대비 중앙값 skill은
$+0.073$ $[+0.043, +0.099]$(CI $>0$)이다.
```

### main_ko.tex #45
Source of any new number: tab:dual

Before:
```latex
\textbf{무조건부로} 중앙값 추정량이다: 두 모집단 모두 4/4 분할에서 백본과
persistence를 이기며, persistence의 0.906--0.920$\times$에 분할 간 폭 $\pm0.007$로
앉는다(학습된 팔은 모두 $\pm0.1$ 이상). \Vrot{}에서는 두 모집단 4/4 분할에서
중앙값으로 백본을 이기고 persistence와는 동률이다(유의 1/4, 분할별
0.94--1.00$\times$). 이 동률은 필터를 TRAIN에서 맞춰도 남는다.
```
After:
```latex
\textbf{무조건부로} 중앙값 추정량이다: 중앙값 skill이 백본 대비 $+0.06\sim+0.39$,
persistence 대비 $+0.08\sim+0.09$로 두 모집단 모두 4/4 분할에서 CI가 0을 넘고,
persistence의 0.906--0.920$\times$에 분할 간 폭 $\pm0.007$로
앉는다(학습된 팔은 모두 $\pm0.1$ 이상). \Vrot{}에서는 백본 대비 중앙값 skill이
$+0.09\sim+0.27$로 두 모집단 4/4 분할에서 CI가 0을 넘고, persistence 대비 중앙값
오차 비는 분할별 0.94--1.00$\times$로 3/4 분할에서 구간이 1을 포함한다. 이 구간은
필터를 TRAIN에서 맞춰도 1을 포함한 채 남는다.
```

### main_ko.tex #46
Source of any new number: numbers already in sentence

Before:
```latex
persistence를 이기는 분할은 많아야 2/4이다. 그 이유는 측정되어 있다.
```
After:
```latex
구간이 1을 제외하는 분할은 많아야 2/4이다. 그 이유는 측정되어 있다.
```

### main_ko.tex #47
Source of any new number: numbers already in sentence

Before:
```latex
따라서 이 동률은 추정기가 아니라 10\,ms CES 측정의 한계다.
필터는 사건 행에서 늦기 때문에 집계 MSE를 백본에 잃으며($-0.06$ --
$-0.52$), 바로 그래서
```
After:
```latex
따라서 1을 포함하는 이 구간은 추정기가 아니라 10\,ms CES 측정의 한계다.
필터는 사건 행에서 늦기 때문에 백본 대비 집계 MSE skill이 $-0.06$ --
$-0.52$이며, 바로 그래서
```

### main_ko.tex #48
Source of any new number: THESIS_RESULTS.md sec. 8bf

Before:
```latex
\Vrot{}에서는 우도 가중이 $|v| > 1000$\,km\,s$^{-1}$의 맞춤 실패 행에 끌려 사후
평균이 컷 분할 3/4에서 백본에 MSE를 잃는다.
```
After:
```latex
\Vrot{}에서는 우도 가중이 $|v| > 1000$\,km\,s$^{-1}$의 맞춤 실패 행에 끌려 사후
평균의 백본 대비 MSE skill이 $-0.03\sim-0.09$로 컷 분할 3/4에서 CI가 0 아래다.
```

### main_ko.tex #49
Source of any new number: numbers already in sentence

Before:
```latex
\paragraph{지명된 팔은 보정되어 있다.} 컷 모집단에서 Student-$t$ head는 혼합과
동률이며(CRPS 차 $+0.009$, 동률 폭 0.02 미만) 모수가 적어 지명되었다. 기준 대비
CRPS는 $+0.229$(\Ti{}), $+0.363$(\Vrot{})이고 모든 분할에서 유의하며,
```
After:
```latex
\paragraph{지명된 팔은 보정되어 있다.} 컷 모집단에서 Student-$t$ head의 혼합 대비
CRPS 차는 $+0.009$로 동률 폭 0.02 안에 있으며 모수가 적어 지명되었다. 기준 대비
CRPS 이득은 $+0.229$(\Ti{}), $+0.363$(\Vrot{})이고 모든 분할에서 CI가 0을 넘으며,
```

### main_ko.tex #50
Source of any new number: tab:family; k=0 CIs from THESIS_RESULTS.md

Before:
```latex
세 가지가 읽힌다(표~\ref{tab:family}). \emph{첫째}, \Vrot{}에서 모델군은 동률이다:
컷의 두 경계 칸은 CI 하한이 $0.000$과 $0.001$이고 포함 모집단에서는 세 팔 모두
동률이며 어떤 계층도 그 칸을 재현하지 않는다. \emph{둘째}, 신경 과정의 가설은
그것이 설계된 바로 그 자리에서 기각된다: 같은 shot의 \Vrot{} 라벨이 하나도 없으면
\emph{어떤} 팔도 학습 평균을 이기지 못하고 \texttt{np}가 가장 나쁘며, 라벨
```
After:
```latex
세 가지가 읽힌다(표~\ref{tab:family}). \emph{첫째}, \Vrot{}에서 모델군의 대조군
대비 $\delta$는 $-0.016\sim+0.021$이다: 컷의 두 경계 칸은 CI 하한이 $0.000$과
$0.001$이고 포함 모집단의 세 칸은 $-0.003\sim-0.016$으로 0.02 띠 안에 있으며 어떤
계층도 그 칸을 재현하지 않는다. \emph{둘째}, 신경 과정의 가설은
그것이 설계된 바로 그 자리에서 기각된다: 같은 shot의 \Vrot{} 라벨이 하나도 없으면
\emph{모든} 팔의 학습 평균 대비 skill이 음수이고($-0.003\sim-0.040$; \texttt{np}
$-0.040$ $[-0.07, -0.02]$, 대조군 $-0.025$ $[-0.06, +0.01]$) \texttt{np}가 가장
나쁘며, 라벨
```

### main_ko.tex #51
Source of any new number: tab:family

Before:
```latex
\texttt{np}의 \Ti{} $+0.026$은 컷에서만 규칙을
넘고 포함에서는 동률이므로 앙상블 이득과
```
After:
```latex
\texttt{np}의 \Ti{} $+0.026$은 컷에서만 규칙을
넘고 포함에서는 $-0.006$으로 띠 안에 있으므로 앙상블 이득과
```

### main_ko.tex #52
Source of any new number: numbers already in sentence

Before:
```latex
카드의 양성 결과 하나는 간극 길이 프로파일이다. \Vrot{}에서 학습된 네 팔 모두
한 스텝짜리 간극에서는 PCHIP에 지고($-0.25$), 두 스텝 이상에서는 $+0.4\sim+0.6$으로
이긴다(비인과 GP는 $+0.02\sim+0.08$).
```
After:
```latex
카드의 양성 결과 하나는 간극 길이 프로파일이다. \Vrot{}에서 학습된 네 팔 모두
PCHIP 대비 skill이 한 스텝짜리 간극에서는 $-0.25$, 두 스텝 이상에서는
$+0.4\sim+0.6$이다(비인과 GP는 $+0.02\sim+0.08$).
```

### main_ko.tex #53
Source of any new number: wording only

Before:
```latex
척도가 \S\ref{sec:asym}의 \Vrot{} 동률이 사는 자리다.
```
After:
```latex
척도가 \S\ref{sec:asym}의 \Vrot{} 영결과가 사는 자리다.
```

### main_ko.tex #54
Source of any new number: numbers already in sentence

Before:
```latex
표~\ref{tab:pooled}가 되고, \textbf{모델은 20\,ms에서도 인과 GP를 이긴다}
($+0.057$, 95\,\% CI $[+0.027, +0.085]$).
```
After:
```latex
표~\ref{tab:pooled}가 되고, \textbf{모델의 인과 GP 대비 skill은 20\,ms에서도 0을
넘는다}($+0.057$, 95\,\% CI $[+0.027, +0.085]$).
```

### main_ko.tex #55
Source of any new number: tab:pooled

Before:
```latex
0.66$으로 오른 뒤 약 70\,ms부터 평평하다. 20\,ms에서 모델은 평균으로는 이기지만
방전의 절반에서만 이기고, 70\,ms에서는 3분의 2에서 이긴다. \textbf{문맥은 평균을
올린다기보다 승리를 전형적인 것으로 만든다.} 이것이
```
After:
```latex
0.66$으로 오른 뒤 약 70\,ms부터 평평하다. 20\,ms에서 통합 skill은 $+0.057$로 CI가
0을 넘지만 방전별 승률은 0.52이고, 70\,ms에서는 0.66이다. \textbf{문맥은 평균을
올린다기보다 승률을 올린다.} 이것이
```

### main_ko.tex #56
Source of any new number: numbers already in sentence (no interval in the record)

Before:
```latex
20\,ms에서도 모델은 배치된 시스템이 실제로 대체하는 persistence를 $+0.356$으로
앞선다.
```
After:
```latex
20\,ms에서도 모델의 persistence---배치된 시스템이 실제로 대체하는 기준선---대비
skill은 $+0.356$이다.
```

### main_ko.tex #57
Source of any new number: numbers already in sentence

Before:
```latex
띠 attention(295{,}746)을 비교하면 셋 다 인과 GP를 4/4로 이기고 그 값은
$+0.113$ / $+0.119$ / $+0.119$이다.
```
After:
```latex
띠 attention(295{,}746)을 비교하면 셋 다 인과 GP 대비 skill이 $+0.113$ / $+0.119$ /
$+0.119$로 4/4 분할에서 CI가 0을 넘는다.
```

### main_ko.tex #58
Source of any new number: numbers already in sentence

Before:
```latex
\texttt{tcn3/5/7}은
같은 도달 범위의 LSTM과 동률이고($-0.004$, $+0.005$, $-0.004$; 어느 쪽으로도
유의한 분할 없음),
```
After:
```latex
\texttt{tcn3/5/7}은
같은 도달 범위의 LSTM과 $-0.004$, $+0.005$, $-0.004$ 차이이고(모든 분할에서 CI가
0을 포함),
```

### main_ko.tex #59
Source of any new number: +0.119 already in paragraph

Before:
```latex
\texttt{tcn3k}는 357{,}570 파라미터 백본과
\Ti{}에서 동급이고 인과 GP를 4/4로 이기며,
```
After:
```latex
\texttt{tcn3k}는 357{,}570 파라미터 백본과
\Ti{}에서 동급이고 인과 GP 대비 skill $+0.119$가 4/4에서 CI 0을 넘으며,
```

### main_ko.tex #60
Source of any new number: wording only

Before:
```latex
\paragraph{모델은 타겟이 움직이는 방전에서 이긴다.}
```
After:
```latex
\paragraph{모델의 방전별 skill은 타겟이 움직이는 방전에서 양수다.}
```

### main_ko.tex #61
Source of any new number: numbers already in sentence

Before:
```latex
\Ti{}는 반대다: 방전의 70\%에서 이기고 상위 1개 방전의 기여 비중이 9--16\%에
그친다.
```
After:
```latex
\Ti{}는 반대다: 승률이 0.70이고 상위 1개 방전의 기여 비중이 9--16\%에
그친다.
```

### main_ko.tex #62
Source of any new number: wording only

Before:
```latex
승패를 예측하는 공변량을 11개 검사한 결과 살아남은 것은 하나뿐이며,
```
After:
```latex
방전별 skill의 부호를 예측하는 공변량을 11개 검사한 결과 살아남은 것은 하나뿐이며,
```

### main_ko.tex #63
Source of any new number: numbers already in sentence

Before:
```latex
  \item \textbf{모델의 구간이 모든 셀에서 두 기준선의 구간을 이긴다}(32/32
        모집단 $\times$ 타겟 $\times$ 변형 $\times$ 분할). 폭과 빗나감을 함께
        벌하는 Winkler 구간 점수 기준으로
```
After:
```latex
  \item \textbf{모델의 구간은 모든 셀에서 두 기준선보다 낮은(좋은) Winkler 점수를
        갖는다}(32/32 모집단 $\times$ 타겟 $\times$ 변형 $\times$ 분할). 폭과
        빗나감을 함께 벌하는 그 구간 점수 기준으로
```

### main_ko.tex #64
Source of any new number: tab:ladder, sec:headline, tab:headline

Before:
```latex
4 & \textbf{오프라인 주장의 상한은 GP 동률이다.} 오프라인 GP는 PCHIP을 이기고
    모델은 그것과 동률일 뿐이다(8개 셀 중 1개 유의)
  & 중간
  & ``미래를 쓰는 보간을 이긴다''를 ``사전등록된 보간들을 이긴다''로 좁혀 적는다.
    배치 가능한 최강인 인과 GP로는 확장된다
```
After:
```latex
4 & \textbf{오프라인 주장의 상한은 오프라인 GP다.} 오프라인 GP의 \Ti{} RMSE는
    PCHIP보다 낮고(153.8 대 173.6) 모델의 그에 대한 짝지은 skill은 $-0.05\sim+0.11$로
    8개 셀 중 7개에서 CI가 0을 포함한다
  & 중간
  & ``미래를 쓰는 보간 대비 skill''을 ``사전등록된 보간들 대비 skill''로 좁혀 적는다.
    배치 가능한 최강인 인과 GP로는 확장된다($+0.08\sim+0.17$, 8/8에서 CI가 0을 넘음)
```

### main_ko.tex #65
Source of any new number: sec:deploy

Before:
```latex
  & 분포 무가정 conformal 구간이 32개 셀 전부에서 두 기준선의 구간을 이긴다는
    사실만 주장한다(\S\ref{sec:deploy})
```
After:
```latex
  & 분포 무가정 conformal 구간이 32개 셀 전부에서 두 기준선보다 낮은 Winkler 점수를
    갖는다는 사실만 주장한다(\S\ref{sec:deploy})
```

### main_ko.tex #66
Source of any new number: wording only

Before:
```latex
    ``전형적인 행에서 보간보다 낫다''고 적지 않는다. 척도 고정 Huber 손실이 본류를
```
After:
```latex
    ``전형적인 행에서 보간보다 오차가 작다''고 적지 않는다. 척도 고정 Huber 손실이 본류를
```

### main_ko.tex #67
Source of any new number: sec:bulk (-0.040), tab:dual

Before:
```latex
  & 같은 규칙으로 포함 중앙값 이득이 $+0.285$로 오르지만 한 분할에 유의한
    MSE 열세가 남고, epoch 상한을 늘려도 없어지지 않는다. \Ti{} 중앙값은 물리 필터가 무조건부로 갖고(\S\ref{sec:physfilter}),
    집계 모델 안이 아니라 옆에 보고한다. \Vrot{}에서 그 필터는 10\,ms 선 맞춤의 광자 바닥에서 중앙값으로 persistence와 동률이다 \\
```
After:
```latex
  & 같은 규칙으로 포함 중앙값 이득이 $+0.285$로 오르지만 한 분할에 CI가 0 아래인
    MSE 차($-0.040$)가 남고, epoch 상한을 늘려도 없어지지 않는다. \Ti{} 중앙값은 물리 필터가 무조건부로 갖고(\S\ref{sec:physfilter}),
    집계 모델 안이 아니라 옆에 보고한다. \Vrot{}에서 그 필터의 persistence 대비 중앙값 오차 비는 0.94--1.00으로 3/4 분할에서 구간이 1을 포함한다(10\,ms 선 맞춤의 광자 바닥) \\
```

### main_ko.tex #68
Source of any new number: tab:headline, tab:gap, tab:dual, sec:headline

Before:
```latex
에 대해 평가했다. 관측 모집단에서 모델은 이온온도에 대해 미래를 쓰는 PCHIP을
유의하고 재현 가능한 skill로 이기고(4개 독립 분할과 두 모집단에 걸쳐
$+0.17\sim+0.32$; 모든 방전을 한 번씩 시험하는 사전등록 반복 5겹 재현에서
$[+0.19, +0.25]$·$[+0.19, +0.31]$), 가장 강한 오프라인 평활기와 동률이며, 두 타겟
모두에서 모든 인과 기준선을 --- 8개 셀 전부의 인과 GP와 비인접 영역($\Delta t >
15$\,ms)을 포함해 --- 이긴다. 회전에서는 재현이 네 분할의 동률을
PCHIP·persistence·인과 GP 대비 평균 제곱 오차의 등록된 승리로 바꾸되(PCHIP 대비
컷 $[+0.07, +0.70]$) 그 승리는 꼬리가 실어 나르며, 중앙값에서 \Vrot{}은 여전히
광자 바닥에서 persistence와 동률이다.
```
After:
```latex
에 대해 평가했다. 관측 모집단에서 모델의 이온온도 skill은 미래를 쓰는 PCHIP 대비
4개 독립 분할과 두 모집단에 걸쳐 $+0.17\sim+0.32$로 모든 CI가 0을 제외하고(모든
방전을 한 번씩 시험하는 사전등록 반복 5겹 재현에서 $[+0.19, +0.25]$·$[+0.19,
+0.31]$), 가장 강한 오프라인 평활기 대비 짝지은 skill은 $-0.05\sim+0.11$로 8셀 중
7셀에서 CI가 0을 포함하며, 두 타겟 모두에서 모든 인과 기준선 대비 skill이 0을
넘는다(이온온도는 8개 셀 전부의 인과 GP 대비 $+0.08\sim+0.17$과 비인접
영역($\Delta t > 15$\,ms)의 $+0.27$·$+0.21$에서 CI가 0을 넘는다). 회전에서는
재현이 네 분할의 0을 포함하는 구간을 PCHIP·persistence·인과 GP 대비 평균 제곱 오차
skill의 등록된 0 제외 구간으로 바꾸되(PCHIP 대비 컷 $[+0.07, +0.70]$) 그 효과는
꼬리가 실어 나르며, 중앙값에서 \Vrot{}의 persistence 대비 오차 비는 여전히
0.94--1.00으로 3/4 분할에서 구간이 1을 포함한다(광자 바닥).
```

### main_ko.tex #69
Source of any new number: tab:mnar, tab:campaign

Before:
```latex
결측이면서 도메인 안인 시점에서 모든 인과 방법 대비 우위는 8/8 모집단--분할
셀에서 재가중을 견딘다. 캠페인 경계를 가로질러서는 시퀀스 나우캐스터가 두 모집단
모두에서 4/4 초기화에 대해 PCHIP과 인과 GP를 앞서는데, 그것이 대체한 윈도
```
After:
```latex
결측이면서 도메인 안인 시점에서 인과 기준 persistence 대비 skill은 $+0.28\sim+0.44$로
8/8 모집단--분할 셀에서 CI가 0을 넘은 채 재가중을 견딘다. 캠페인 경계를
가로질러서는 시퀀스 나우캐스터가 두 모집단 모두에서 PCHIP 대비 $+0.17\sim+0.20$,
인과 GP 대비 $+0.11\sim+0.16$으로 4/4 초기화에서 CI가 0을 넘는데, 그것이 대체한 윈도
```

### main_ko.tex #70
Source of any new number: sec:deploy

Before:
```latex
모든 셀에서 적합한 채점 규칙으로 두 기준선의 구간을 이기는 보정된 구간을
내놓는다.
```
After:
```latex
모든 셀에서 적합한 채점 규칙(Winkler)의 점수가 두 기준선보다 낮은 보정된 구간을
내놓는다.
```

### main_ko.tex #71
Source of any new number: tab:replicate, tab:dual

Before:
```latex
작동하지 않는 곳과 그 이유도 함께 보고한다. 회전은 평균에서만 보간을 이기고
중앙값에서는 동률이며 그 skill은 과도 구간에 산다. 이온온도의 우위도 꼬리의 것이다:
```
After:
```latex
작동하지 않는 곳과 그 이유도 함께 보고한다. 회전의 보간 대비 구간은 평균에서만
0을 제외하고($[+0.07, +0.70]$) 중앙값의 persistence 대비 오차 비는 0.94--1.00으로
구간이 1을 포함하며 그 skill은 과도 구간에 산다. 이온온도의 skill도 꼬리의 것이다:
```

### main_ko.tex #72
Source of any new number: tab:gap

Before:
```latex
45\,ms를 넘어서면 포함 모집단은 동률이다. 이 중 어느 것도 천장 논변이 아니며,
```
After:
```latex
45\,ms를 넘어서면 포함 모집단의 구간은 0을 포함한다($-0.004$, $[-0.30, +0.25]$). 이
중 어느 것도 천장 논변이 아니며,
```

### main_ko.tex #73
Source of any new number: tab:family, sec:family

Before:
```latex
폭 스윕은 평평하다. 모델군 축도 닫혔다: shot 인코딩 신경 과정, 타 장치에서
사전학습된 transformer, 4-초기화 앙상블은 전부 동률이며, 같은 shot의 라벨이 없으면
어떤 팔도 학습 평균을 넘지 못한다.
```
After:
```latex
폭 스윕은 평평하다. 모델군 축도 닫혔다: shot 인코딩 신경 과정, 타 장치에서
사전학습된 transformer, 4-초기화 앙상블은 \Ti{} skill을 $-0.09\sim+0.03$만큼 바꿀 뿐
어느 것도 두 모집단 모두에서 0.02 띠를 넘지 못하며, 같은 shot의 라벨이 없으면
모든 팔의 학습 평균 대비 skill이 음수다($-0.003\sim-0.040$).
```

## Second batch

## main.tex, second batch (12 edits)

### main.tex #B1
Source of any new number: wording only

Before:
```latex
        that moves the rest} (\S\ref{sec:bulk}, \S\ref{sec:family}): the \Ti{} win
        is a tail effect---in the median row
```
After:
```latex
        that moves the rest} (\S\ref{sec:bulk}, \S\ref{sec:family}): the \Ti{} skill
        is a tail effect---in the median row
```

### main.tex #B2
Source of any new number: tab:campaign (control OFF rows)

Before:
```latex
        Re-splitting strictly in time across campaigns removes the windowed
        control's advantage over interpolation (2/4 and 0/4)---the fast
```
After:
```latex
        Re-splitting strictly in time across campaigns takes the windowed
        control's skill over interpolation to $-0.001$ to $+0.091$ with the CI above
        zero on only 2/4 and 0/4 initialisations---the fast
```

### main.tex #B3
Source of any new number: tab:ladder

Before:
```latex
  \caption{RMSE ladder for the seed-42 test split (cut population, genuine
  measurements): the causal nowcaster attains the lowest RMSE among deployable
  methods for both targets and edges the future-using interpolants. Same numbers
```
After:
```latex
  \caption{RMSE ladder for the seed-42 test split (cut population, genuine
  measurements): the causal nowcaster attains the lowest RMSE among deployable
  methods for both targets and a lower RMSE than the future-using interpolants. Same numbers
```

### main.tex #B4
Source of any new number: numbers already in sentence

Before:
```latex
thing.} Against \textbf{persistence---the baseline an online system actually
runs---the \Ti{} advantage survives reweighting on all four splits in both
populations} ($+0.28$ to $+0.44$, every CI excluding zero);
```
After:
```latex
thing.} Against \textbf{persistence---the baseline an online system actually
runs---the \Ti{} skill survives reweighting on all four splits in both
populations} ($+0.28$ to $+0.44$, every CI excluding zero);
```

### main.tex #B5
Source of any new number: wording only

Before:
```latex
  \caption{Campaign (temporal) split: the windowed control's advantage over
  interpolation collapses, per-shot standardization repairs it, and the sequence
  nowcaster keeps it in both populations.
```
After:
```latex
  \caption{Campaign (temporal) split: the windowed control's skill over
  interpolation collapses, per-shot standardization repairs it, and the sequence
  nowcaster keeps it in both populations.
```

### main.tex #B6
Source of any new number: tab:campaign (control OFF rows)

Before:
```latex
\emph{windowed} model's advantage over offline interpolation does not survive the
shift: 2/4 (cut) and 0/4 (inclusive) initialisations, and 0/4 against the causal
GP.
```
After:
```latex
\emph{windowed} model's skill over offline interpolation does not survive the
shift: $-0.001$ to $+0.091$, with the CI above zero on 2/4 (cut) and 0/4
(inclusive) initialisations, and 0/4 against the causal GP.
```

### main.tex #B7
Source of any new number: wording only

Before:
```latex
  \caption{Peak-stratified TEST skill of \seqv{}, both populations (4-split means,
  per-split points, PR4 counts). The advantage over interpolation concentrates
  where interpolation is weakest;
```
After:
```latex
  \caption{Peak-stratified TEST skill of \seqv{}, both populations (4-split means,
  per-split points, PR4 counts). The skill over interpolation concentrates
  where interpolation is weakest;
```

### main.tex #B8
Source of any new number: numbers already in sentence

Before:
```latex
turns to $+0.145$ (4/4), with the \Vrot{} median at 1.02$\times$ persistence---but one
split keeps a significant \Ti{} MSE deficit ($-0.040$; pooled $-0.019$), so by the
```
After:
```latex
turns to $+0.145$ (4/4), with the \Vrot{} median at 1.02$\times$ persistence---but one
split keeps a \Ti{} MSE difference of $-0.040$ with the CI below zero (pooled $-0.019$), so by the
```

### main.tex #B9
Source of any new number: wording only

Before:
```latex
bit-identical (best epochs 24--59), so the one deficit is the fit-failure rows'
square, not under-training.
```
After:
```latex
bit-identical (best epochs 24--59), so that one negative MSE difference is the fit-failure rows'
square, not under-training.
```

### main.tex #B10
Source of any new number: numbers already in sentence

Before:
```latex
($-0.005$). The transformer family is also behind the BiLSTM on \Ti{} at 7$\times$
the parameters (cut $-0.09$, inclusive $-0.03$ to $-0.04$).
```
After:
```latex
($-0.005$). The transformer family's $\delta$ to the BiLSTM on \Ti{} is also negative at
7$\times$ the parameters (cut $-0.09$, inclusive $-0.03$ to $-0.04$).
```

### main.tex #B11
Source of any new number: wording only

Before:
```latex
some 62 discharges takes the pooled advantage to zero or below on all four splits.
```
After:
```latex
some 62 discharges takes the pooled skill to zero or below on all four splits.
```

### main.tex #B12
Source of any new number: tab:campaign

Before:
```latex
above zero on 4/4 initialisations in both populations, where the windowed control it replaced had
lost its offline advantage entirely---and the difference is measured, not
```
After:
```latex
above zero on 4/4 initialisations in both populations, where the windowed control it replaced had
lost its offline advantage entirely (CI above zero on 2/4 and 0/4 vs.\ PCHIP, 0/4 vs.\ the
causal GP)---and the difference is measured, not
```

## main_ko.tex, second batch (14 edits)

### main_ko.tex #B1
Source of any new number: wording only (en: 'The edge concentrates')

Before:
```latex
\item[우위의 소재] 우위는 고변동 국소 구간에 집중되며(peak 계층 \Ti{} skill
```
After:
```latex
\item[skill의 소재] skill은 고변동 국소 구간에 집중되며(peak 계층 \Ti{} skill
```

### main_ko.tex #B2
Source of any new number: tab:campaign (control OFF rows)

Before:
```latex
        가로질러 시간으로 엄격히 재분할하면 윈도 대조군의 보간 대비 우위는
        사라지지만(2/4, 0/4)---고속 진단은 기간 사이에 $1.22\,\sigma$(BES)
```
After:
```latex
        가로질러 시간으로 엄격히 재분할하면 윈도 대조군의 보간 대비 skill은
        $-0.001\sim+0.091$로 2/4·0/4 초기화에서만 CI가 0을 넘지만---고속 진단은 기간
        사이에 $1.22\,\sigma$(BES)
```

### main_ko.tex #B3
Source of any new number: wording only

Before:
```latex
  \item \textbf{우위가 사는 곳}: 고변동 국소 구간(peak 계층 \Ti{} skill
```
After:
```latex
  \item \textbf{skill이 사는 곳}: 고변동 국소 구간(peak 계층 \Ti{} skill
```

### main_ko.tex #B4
Source of any new number: tab:ladder

Before:
```latex
  나우캐스터가 두 타겟 모두에서 배치 가능한 방법 중 최저 RMSE를 달성하고 미래를
  쓰는 보간도 앞선다. 표~\ref{tab:ladder}와 같은 수치이며, 둘 다 같은 동결
```
After:
```latex
  나우캐스터가 두 타겟 모두에서 배치 가능한 방법 중 최저 RMSE를 달성하고 미래를
  쓰는 보간보다도 RMSE가 낮다. 표~\ref{tab:ladder}와 같은 수치이며, 둘 다 같은 동결
```

### main_ko.tex #B5
Source of any new number: numbers already in sentence

Before:
```latex
\textbf{persistence --- 온라인 시스템이 실제로 돌리는 기준선 --- 를 상대로는 \Ti{}
우위가 두 모집단 모두에서 4개 분할 전부 재가중을 견딘다}($+0.28\sim+0.44$, 모든
```
After:
```latex
\textbf{persistence --- 온라인 시스템이 실제로 돌리는 기준선 --- 를 상대로는 \Ti{}
skill이 두 모집단 모두에서 4개 분할 전부 재가중을 견딘다}($+0.28\sim+0.44$, 모든
```

### main_ko.tex #B6
Source of any new number: wording only

Before:
```latex
  \caption{캠페인(시간) 분할: 윈도 대조군의 보간 대비 우위는 붕괴하고, shot별
  표준화가 이를 복구하며, 시퀀스 나우캐스터는 두 모집단 모두에서 우위를 유지한다.
```
After:
```latex
  \caption{캠페인(시간) 분할: 윈도 대조군의 보간 대비 skill은 붕괴하고, shot별
  표준화가 이를 복구하며, 시퀀스 나우캐스터는 두 모집단 모두에서 skill을 유지한다.
```

### main_ko.tex #B7
Source of any new number: tab:campaign (control OFF rows)

Before:
```latex
이 표에서 명시할 만한 것이 셋 있다. 첫째, \emph{윈도} 모델의 오프라인 보간 대비
우위는 이동을 견디지 못한다: 2/4(컷)와 0/4(포함) 초기화이며, 인과 GP 대비로는
0/4다.
```
After:
```latex
이 표에서 명시할 만한 것이 셋 있다. 첫째, \emph{윈도} 모델의 오프라인 보간 대비
skill은 이동을 견디지 못한다: $-0.001\sim+0.091$로 CI가 0을 넘는 초기화는 2/4(컷)와
0/4(포함)이며, 인과 GP 대비로는 0/4다.
```

### main_ko.tex #B8
Source of any new number: wording only

Before:
```latex
\subsection{우위는 고변동 국소 구간에 집중된다}
```
After:
```latex
\subsection{skill은 고변동 국소 구간에 집중된다}
```

### main_ko.tex #B9
Source of any new number: wording only

Before:
```latex
  PR4 카운트). 보간 대비 우위는 보간이 가장 약한 곳에 집중되며, \Vrot{} skill은
```
After:
```latex
  PR4 카운트). 보간 대비 skill은 보간이 가장 약한 곳에 집중되며, \Vrot{} skill은
```

### main_ko.tex #B10
Source of any new number: numbers already in sentence

Before:
```latex
중앙값이 persistence의 1.02$\times$에 닿는다---다만 한 분할에 유의한 \Ti{} MSE
열세($-0.040$, 풀링 $-0.019$)가 남아 사전등록 문언상 칸은 조건부로 남는다.
```
After:
```latex
중앙값이 persistence의 1.02$\times$에 닿는다---다만 한 분할에 CI가 0 아래인 \Ti{}
MSE 차($-0.040$, 풀링 $-0.019$)가 남아 사전등록 문언상 칸은 조건부로 남는다.
```

### main_ko.tex #B11
Source of any new number: wording only

Before:
```latex
네 개 모두 비트 단위로 같았다(최선 epoch 24--59). 따라서 남은 열세는 학습 부족이
아니라 맞춤 실패 행의 제곱이다.
```
After:
```latex
네 개 모두 비트 단위로 같았다(최선 epoch 24--59). 따라서 남은 음의 MSE 차는 학습
부족이 아니라 맞춤 실패 행의 제곱이다.
```

### main_ko.tex #B12
Source of any new number: numbers already in sentence

Before:
```latex
transformer 계열은 7배의 파라미터로 \Ti{}에서 BiLSTM보다 열세이기도 하다(컷
$-0.09$, 포함 $-0.03\sim-0.04$).
```
After:
```latex
transformer 계열은 7배의 파라미터로 \Ti{}에서 BiLSTM 대비 $\delta$가 음수이기도
하다(컷 $-0.09$, 포함 $-0.03\sim-0.04$).
```

### main_ko.tex #B13
Source of any new number: wording only

Before:
```latex
모두에서 통합 우위가 0 이하로 내려간다. \textbf{좁은 구간은 일반적 효과와 같지
```
After:
```latex
모두에서 통합 skill이 0 이하로 내려간다. \textbf{좁은 구간은 일반적 효과와 같지
```

### main_ko.tex #B14
Source of any new number: tab:campaign

Before:
```latex
인과 GP 대비 $+0.11\sim+0.16$으로 4/4 초기화에서 CI가 0을 넘는데, 그것이 대체한 윈도
대조군은 오프라인 우위를 완전히 잃었던 자리다. 그리고 그 차이는 단언이 아니라
```
After:
```latex
인과 GP 대비 $+0.11\sim+0.16$으로 4/4 초기화에서 CI가 0을 넘는데, 그것이 대체한 윈도
대조군은 오프라인 우위를 완전히 잃었던(PCHIP 대비 CI가 0을 넘는 초기화 2/4·0/4, 인과
GP 대비 0/4) 자리다. 그리고 그 차이는 단언이 아니라
```

## Left unchanged (with reason)

Rules, definitions, hypotheses and registered wordings (rule 2):
- main.tex "If a causal model beats a future-using interpolant ... a fortiori beats every causal baseline" (sec:intro) and main_ko.tex "인과 모델이 미래를 쓰는 보간을 이긴다면 ... 모든 인과 기준선을 이긴다" / one-pager "이기는지를 묻는다. 이겼다면" -- hypothesis statements, not results.
- "the incumbent method to beat" (sec:related) -- method description.
- Metric definition "(positive = model better; 0 = tie)" / "(양수 = 모델 우위; 0 = 동률)" (sec:eval).
- "``beats every deployable causal method'' is judged against it" / "``배치 가능한 모든 인과 방법을 이긴다''는 ... 판정한다" -- defines the comparator.
- "a cell \emph{beats} iff the 2.5th percentile exceeds zero" / "\emph{이김}으로 본다" -- registered readout rule.
- "Two pre-registered conditions: beat a retrained ... anchor" -- registered rule.
- Captions defining "win rate" (tab:bulk, tab:pooled) and every use of the metric name "win rate"/"승률".
- Pre-registration quotation "``ties in aggregate but wins in the median''" / "``집계에서는 동률이지만 중앙값에서는 이긴다''".
- 4-init ensemble paragraph: "tie threshold / tie band / recorded as a tie" (en, ko 동률) -- the registered verdict label; the paragraph already carries the paired gains, 4/4 significance and the 0.02 rule.
- tab:family verdict labels (improve / edge / tie / worse; 개선 / 경계 / 동률 / 열세) -- rule-defined labels.
- tab:falsify (ko) claim and rule columns ("...보간을 이긴다", "동률로 보고하고", "우위를 ... 한정한다", "동률이거나 진다") -- registered claim/rule wording; the observed-result column was rewritten.
- "유의 열세 N/4" / "significant deficits N/4" -- a defined count, kept as the count.
- "an output which must also win the aggregate" (en) / "집계도 이겨야 하는 출력" (ko) -- describes a requirement, not a result.
- "regime where a causal method is least expected to win" -- expectation, not a result.
- Limitations row 7 (ko) "우위 방향의 해석은 안전하다" -- about the direction of the design bias.
- Abstract clause "removes the windowed control's advantage over interpolation" / "윈도 대조군의 보간 대비 우위를 제거한다" -- the numbers (-0.001..+0.091, CI above zero on 2/4 and 0/4) are given in the contributions and sec:campaign rewrites.

Result sentences whose only available record is a point estimate (no interval exists; stated as point estimate and marked):
- Winkler interval scores (sec:deploy, abstract, contributions, conclusion): scores given and 32/32 count; no bootstrap interval in paper_numbers.json conformal block.
- Skill over persistence at 20 ms of context (+0.356): no interval in THESIS sec. 8al.
- Gap-length profile on V_rot (-0.25 / +0.4..+0.6): no interval in the record.
- Two-discharge case study (+0.42 / +0.03 / -1.06 / -0.73): single discharges, marked "no interval" in the text.
- History-only inclusive model +0.15..+0.23 over PCHIP: per-split CI not in the record.
- Operator-family range "differ by at most 0.023": paired tcn CIs include zero, attention -0.023 significant on 3/4 (stated in sec:context).
