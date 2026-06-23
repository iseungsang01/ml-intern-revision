# Interpolation Baseline References for KSTAR CES Nowcasting

This document justifies the conventional interpolation baselines used to benchmark the
`MultimodalCESPredictor` model against gap-filled ion temperature (CES_TI) and toroidal
rotation velocity (CES_VT) predictions on KSTAR Charge Exchange Spectroscopy (CES) data.
Citations were verified via web search and primary-source fetches in June 2026.

---

## 1. Linear Interpolation

**Description.** Connects adjacent observed CES values with a straight line, yielding a
piecewise-linear reconstruction for gaps.

**Why it is a standard baseline.** Linear interpolation is the ubiquitous zero-effort
gap-filler in plasma diagnostics: it requires no free parameters, has a closed-form
solution, and is the default alignment strategy when multiple diagnostic systems with
different sampling rates (e.g., BES at ~10 kHz vs. CES at ~100 Hz) must be co-registered
to a common time grid. It is explicitly used for this purpose in KSTAR multimodal data
pipelines (see context in Lee et al. 2017 KSTAR CES system papers). Its poor performance
on rapidly evolving profiles (ELM bursts, sawtooth crashes) makes it an informative
lower bound for any learned model.

**Canonical reference (numerical analysis foundation).**
> Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
> *Numerical Recipes: The Art of Scientific Computing* (3rd ed.), Cambridge University Press.
> Chapter 3 (Interpolation and Extrapolation).

Linear interpolation is covered in virtually every numerical-methods text and is too
foundational to require a single definitional paper. The above compendium is the standard
reference that practitioners cite.

**Note.** No single "linear interpolation for CES" domain paper was found during this
research because the method is too elementary to merit a dedicated citation in the fusion
literature — it is assumed rather than described.

---

## 2. Monotone Cubic Hermite Interpolation (PCHIP)

**Description.** Fits a piecewise cubic polynomial through each pair of adjacent data
points such that the cubic segment is guaranteed to be monotone whenever the data are
locally monotone. Derivatives at knots are chosen by the Fritsch-Carlson condition rather
than by global minimisation of curvature (as in natural cubic splines).

**Why it avoids overshoot at plasma transients.** Natural cubic splines minimise global
curvature and enforce C² continuity, which means that a sharp local feature (e.g., a
fast-ion temperature spike during an ELM crash or a sudden CES_VT reversal after a
sawtooth) can produce large oscillatory ripples across multiple grid cells — a Spline
analogue of the Gibbs phenomenon. PCHIP's local, one-pass derivative selection explicitly
prevents the interpolant from exceeding the data range in any interval where the data are
monotone. This makes it the preferred baseline for ion-temperature and rotation profiles
that can exhibit sawtooth and ELM step-like features at 10 ms temporal resolution.

**Primary citation.**
> Fritsch, F. N., & Carlson, R. E. (1980).
> "Monotone Piecewise Cubic Interpolation."
> *SIAM Journal on Numerical Analysis*, **17**(2), 238–246.
> DOI: [10.1137/0717021](https://doi.org/10.1137/0717021)
> [ADS abstract](https://ui.adsabs.harvard.edu/abs/1980SJNA...17..238F/abstract)

**Supporting citation (convexity-preserving extension).**
> Dougherty, R. L., Edelman, A. S., & Hyman, J. M. (1989).
> "Nonnegativity-, Monotonicity-, or Convexity-Preserving Cubic and Quintic Hermite
> Interpolation."
> *Mathematics of Computation*, **52**(186), 471–494.
> DOI: [10.1090/S0025-5718-1989-0962209-1](https://doi.org/10.1090/S0025-5718-1989-0962209-1)
> [OSTI record](https://www.osti.gov/biblio/6134518-nonnegativity-monotonicity-convexity-preserving-cubic-quintic-hermite-interpolation)

PCHIP is implemented in SciPy (`scipy.interpolate.PchipInterpolator`), MATLAB (`pchip`),
and R (`signal::pchip`), making it a reproducible, widely-available baseline.

---

## 3. Natural Cubic Spline

**Description.** Fits a globally C²-smooth piecewise cubic polynomial by solving a
tridiagonal system that minimises the integral of the squared second derivative (total
curvature). Boundary conditions: second derivative = 0 at both endpoints.

**Why it is a standard baseline.** Cubic splines are the canonical "smooth interpolant"
in scientific computing and have been used to reconstruct Thomson scattering electron
temperature profiles, equilibrium flux surfaces, and impurity density gradients in fusion
experiments. They are the natural upgrade from linear interpolation when smoothness is
required, and they serve as the comparison point against which PCHIP's monotonicity
advantage is judged.

**Primary citation (smoothing-spline framework).**
> Reinsch, C. H. (1967).
> "Smoothing by Spline Functions."
> *Numerische Mathematik*, **10**(3), 177–183.
> DOI: [10.1007/BF02162161](https://doi.org/10.1007/BF02162161)
> [Springer link](https://link.springer.com/article/10.1007/BF02162161)
> [UVM mirror PDF](https://tlakoba.w3.uvm.edu/AppliedUGMath/auxpaper_Reinsch_1967.pdf)

**Known limitation.** Natural cubic splines can produce unphysical oscillations near
sharp transients in CES data (Gibbs-like ringing). This is well-documented; see:
> Kvernadze, G. (1998).
> "Convergence and Gibbs' Phenomenon in Cubic Spline Interpolation of Discontinuous
> Functions."
> *Journal of Computational and Applied Mathematics*, **92**(2), 173–188.
> DOI: [10.1016/S0377-0427(97)00199-4](https://doi.org/10.1016/S0377-0427%2897%2900199-4)

---

## 4. Autoregressive (AR) Baseline for Irregular Time Series (IAR)

**Description.** A discrete-time AR(1) model adapted to irregular observation gaps. The
IAR model defines:

    y_{t_j} = φ^(t_j − t_{j−1}) · y_{t_{j−1}} + σ√(1 − φ^{2(t_j − t_{j−1})}) · ε_{t_j}

where φ is the autocorrelation parameter, σ² is the innovation variance, and t_j are the
actual (unequally spaced) observation times. This is the discrete-time analogue of a
continuous-time Ornstein–Uhlenbeck (CAR(1)) process, and it degenerates to standard AR(1)
when observations are equally spaced (t_j − t_{j−1} = 1 for all j).

**Why it is a standard baseline.** CES ion temperature is measured at irregular intervals
(10–100 ms depending on neutral-beam injection scheduling and discharge conditions) with
approximately 8% missing CES_TI and 24% missing CES_VT. Standard AR(1) models assume
equal spacing and are therefore misspecified when applied to unevenly sampled plasma
diagnostics. The IAR model is the natural minimal-parameter baseline for the unevenly
sampled case: it introduces no user-chosen kernel and has only two free parameters (φ, σ),
making it a well-defined null model whose performance is easy to interpret.

**Primary citation.**
> Eyheramendy, S., Elorrieta, F., & Palma, W. (2018).
> "An Irregular Discrete Time Series Model to Identify Residuals with Autocorrelation in
> Astronomical Light Curves."
> *Monthly Notices of the Royal Astronomical Society*, **481**(4), 4311–4322.
> DOI: [10.1093/mnras/sty2487](https://doi.org/10.1093/mnras/sty2487)
> [Oxford Academic](https://academic.oup.com/mnras/article/481/4/4311/5094606)

**Extension (CIAR — bivariate, handles negative autocorrelation).**
> Elorrieta, F., Eyheramendy, S., & Palma, W. (2019).
> "Discrete-Time Autoregressive Model for Unequally Spaced Time-Series Observations."
> *Astronomy & Astrophysics*, **627**, A120.
> DOI: [10.1051/0004-6361/201935560](https://doi.org/10.1051/0004-6361/201935560)
> [A&A full text](https://www.aanda.org/articles/aa/full_html/2019/07/aa35560-19/aa35560-19.html)

**Software.** R package `iAR` on CRAN implements IAR/CIAR fitting via maximum likelihood.
A Python `iar` package is also available on PyPI.

---

## 5. Gaussian-Process Regression (GP)

**Description.** A non-parametric Bayesian interpolant that places a prior over functions
via a kernel (covariance function). Observations update the prior to a posterior GP, whose
mean is used for interpolation/nowcasting and whose variance gives calibrated uncertainty
estimates. For a stationary squared-exponential kernel, GP regression reduces to
kriging/optimal linear interpolation in continuous time, but the kernel can be chosen to
encode plasma-physics constraints (e.g., smoothness near the pedestal, the Chilenski
gibbs-free kernel).

**Why it is the highest-bar conventional baseline.** GP regression has become the
community-standard method for fitting and gap-filling plasma profile data in tokamaks
because it (a) handles arbitrary irregular time grids without re-sampling artifacts,
(b) propagates measurement uncertainties into the posterior, and (c) provides a principled
way to impose boundary conditions and derivative constraints. Chilenski et al. (2015)
demonstrated that GP significantly outperforms polynomial fits and splines for impurity
transport coefficient estimation on Alcator C-Mod, establishing it as the reference method
for fusion profile reconstruction. A 2024 Nuclear Fusion review paper by Michoski et al.
explicitly recommends GP as the default tool for signal regression across multiple DIII-D
diagnostic modalities.

**Primary domain citation (fusion profile fitting).**
> Chilenski, M. A., Greenwald, M., Marzouk, Y., Howard, N. T., White, A. E., Rice, J. E.,
> & Walk, J. R. (2015).
> "Improved Profile Fitting and Quantification of Uncertainty in Experimental Measurements
> of Impurity Transport Coefficients using Gaussian Process Regression."
> *Nuclear Fusion*, **55**(2), 023012.
> DOI: [10.1088/0029-5515/55/2/023012](https://doi.org/10.1088/0029-5515/55/2/023012)
> [IOPscience](https://iopscience.iop.org/article/10.1088/0029-5515/55/2/023012)
> [MIT DSpace](https://dspace.mit.edu/handle/1721.1/96967)

**2024 review citation (GP as standard across fusion diagnostics).**
> Michoski, C., Oliver, T. A., Hatch, D. R., Diallo, A., Kotschenreuther, M., Eldon, D.,
> Waller, M., Groebner, R., & Nelson, A. O. (2024).
> "A Gaussian Process Guide for Signal Regression in Magnetic Fusion."
> *Nuclear Fusion*, **64**(3), 035001.
> DOI: [10.1088/1741-4326/ad1af5](https://doi.org/10.1088/1741-4326/ad1af5)
> [IOPscience](https://iopscience.iop.org/article/10.1088/1741-4326/ad1af5)

**Foundational statistical learning citation.**
> Rasmussen, C. E., & Williams, C. K. I. (2006).
> *Gaussian Processes for Machine Learning*. MIT Press.
> ISBN 0-262-18253-X. [Free PDF](http://www.gaussianprocess.org/gpml/)

---

## 6. Skill Score Convention: 1 − MSE_model / MSE_baseline

**Description.** A normalised accuracy metric inherited from numerical weather prediction.
Given a model and a reference baseline (persistence or mean), the skill score is:

    SS = 1 − MSE_model / MSE_baseline

SS = 1 is perfect; SS = 0 means the model ties the baseline; SS < 0 means the model is
worse than the baseline. The persistence baseline uses the last observed CES value as the
prediction for every future gap timestep; the mean baseline uses the training-set mean.

**Why this convention.** Murphy (1988) established the MSE-based skill score as the
standard decomposition for evaluating forecasts, showing that the skill score factors into
correlation, conditional bias, and unconditional bias terms. In the CES nowcasting context,
persistence is the cheapest physically meaningful baseline (CES_TI and CES_VT change
slowly compared to BES/ECEI turbulence) and a model that fails to beat persistence
provides no actionable signal for the plasma control system. Using `skill_vs_persistence`
as the primary optimisation target in `automl_agent_loop.py` directly follows Murphy's
recommendation that baselines should reflect the simplest defensible prediction strategy.

**Primary citation.**
> Murphy, A. H. (1988).
> "Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation
> Coefficient."
> *Monthly Weather Review*, **116**(12), 2417–2424.
> DOI: [10.1175/1520-0493(1988)116⟨2417:SSBOTM⟩2.0.CO;2](https://doi.org/10.1175/1520-0493(1988)116%3C2417:SSBOTM%3E2.0.CO;2)
> [AMS journal page](https://journals.ametsoc.org/mwr/article/116/12/2417/63823/Skill-Scores-Based-on-the-Mean-Square-Error-and)

---

## Why These Are the Right Bar for CES Nowcasting

The five baselines above span the full complexity spectrum of gap-filling for irregularly
sampled scientific time series. Linear interpolation is the implicit default used
everywhere in KSTAR multi-diagnostic data pipelines and sets the floor: any model that
cannot beat linear interpolation on smooth intervals is unacceptable. Natural cubic spline
adds global smoothness but fails near ELM and sawtooth transients — the very events where
accurate CES reconstruction is most scientifically valuable — making it an important
adversarial case. PCHIP closes the spline gap by guaranteeing monotone local behaviour
and is therefore the strongest purely interpolative competitor on transient-rich data: it
requires no model training and no physics knowledge, yet avoids unphysical overshoot.
IAR(1) introduces time-series structure (the autocorrelation of CES signals over 10–100 ms
gaps) in a way that correctly accounts for irregular spacing; it is the minimal causal
extrapolation baseline against which recurrent/transformer models should be compared.
Gaussian-process regression is the current community standard for fusion profile
reconstruction, handles uncertainty propagation naturally, and represents the highest
achievable performance without incorporating the multimodal BES/ECEI/MC covariates that
`MultimodalCESPredictor` uses. A `MultimodalCESPredictor` that outperforms GP on
`skill_vs_persistence` demonstrates that the BES/ECEI/MC channels carry information
beyond what the CES history alone encodes — which is the core scientific claim of this
work. All five baselines are therefore necessary and sufficient to establish that claim
without over-claiming novelty.

---

## Summary Table

| Baseline                  | Parameters | Handles irregular gaps | Avoids transient overshoot | Uncertainty estimate |
|---------------------------|:----------:|:---------------------:|:--------------------------:|:-------------------:|
| Linear interpolation      | 0          | Yes (piecewise)        | No                         | No                  |
| Natural cubic spline      | 0          | Yes (global solve)     | No (can ring)              | No                  |
| PCHIP (monotone cubic)    | 0          | Yes (local)            | Yes (guaranteed)           | No                  |
| IAR(1) autoregressive     | 2 (φ, σ)  | Yes (exact formulation)| Partial (AR structure)     | Via σ estimate      |
| Gaussian-process regression | kernel hyperparams | Yes       | Depends on kernel          | Yes (posterior var) |

---

*Research conducted June 2026. All URLs verified at time of writing. This file is
read-only reference material; do not modify it during AutoML loop runs.*
