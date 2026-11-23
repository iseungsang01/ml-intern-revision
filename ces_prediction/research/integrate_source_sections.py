"""Integrate audited prose without fitting models or changing result tables."""
import json
import re
from pathlib import Path

import introduction_bibliography_check as B

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/paper/introduction_review'
PAPER = ROOT / 'docs/paper/ppcf_revision'


def main():
    intro = (OUT/'final_introduction_en.tex').read_text(encoding='utf8').rstrip()+'\n\n'
    replacements = {
        'CES reports that became available strictly before the frame': 'past CES reports made visible by the specified availability rule',
        'In the first, history-only and lightweight models are compared with reference estimators on matched rows, information sets and computation budgets.': 'The first-stage plan compares history-only and lightweight models, including CNN and causal temporal-convolution candidates, with reference estimators. Fair comparisons require matched rows and information sets, with computation budgets stated explicitly. Completed comparisons retain their registered support and budget restrictions; they do not establish a budget-matched ranking of every architecture.',
        'the median of at least three of the five subsequent reports': 'the median of measured values in the next five grid rows, requiring at least three measured values',
        ' Such intra-ELM dynamics lie below the 10\\,ms grid used here.': '',
        'Requirement studies for burning-plasma devices ask for comparable time resolution.': 'Diagnostic requirement studies also motivate timely ion-temperature estimates.',
        'The KSTAR instrument observes C\\,VI near 529\\,nm and has a separate background view': 'The KSTAR calibration study describes observation of a C\\,VI charge-exchange line',
        'On DIII-D, main-ion and carbon temperatures differed in the edge, by tens of eV inside the H-mode pedestal top and by more near the separatrix': 'On DIII-D, main-ion and carbon temperatures have been measured to differ near the edge',
        'On TCV, a Kalman observer merges fast interferometry with slower Thomson scattering': 'In an offline replay of TCV data, a Kalman observer combines fast interferometry with slower Thomson scattering',
    }
    for old, new in replacements.items():
        assert old in intro, old
        intro = intro.replace(old, new)
    theory = (OUT/'final_theory_en.tex').read_text(encoding='utf8').rstrip()+'\n\n'
    start = theory.index('A neutral-beam atom')
    end = theory.index('The result is therefore', start)
    theory = theory[:start] + r'''Charge exchange between neutral-beam atoms and fully stripped carbon produces excited ions emitting the C\,VI line. The active emission is weighted near the beam--line-of-sight intersection; passive emission along the line of sight can overlap it~\citep{Fonck1984DeterminationPlasma,Isler1994OverviewCharge,Viezzer2011PassiveEdge}.

For a simplified Gaussian Doppler line, let $W_D$ be its standard deviation in wavelength after accounting for instrumental broadening, and $\lambda_0$ the rest wavelength. The carbon-ion temperature in energy units satisfies
\begin{equation}
 \Theta_C \equiv k_B T_C = m_C c^2\left(\frac{W_D}{\lambda_0}\right)^2 .
\end{equation}
Here $m_C$ is the carbon-ion mass and $c$ is the speed of light. The relation omits effects such as unresolved fine structure; actual spectral analysis must account for the instrumental response and line shape~\citep{Fonck1984DeterminationPlasma,Lee2026NeuralNetworkbased}. In the delivered files the energy-unit carbon-impurity temperature is denoted $T_i$. We use these reports and do not fit raw spectra.

''' + theory[end:]
    theory = theory.replace('A causal estimator is any function of $\\mathcal{I}_t$, which corresponds to filtering in state-estimation terms. Two-sided interpolation also uses later reports, which corresponds to smoothing', 'Our causal estimator is restricted to $\\mathcal{I}_t$, sharing the past-and-present information restriction of filtering. Its supervised target is a reported quantity, rather than an identified physical state. Two-sided interpolation also uses later reports and has a smoothing-like information set')
    theory = theory.replace('the CES reports available before frame $t$', 'the past CES reports made visible by the specified availability rule at frame $t$')
    theory = theory.replace('an offline median of at least three of the five subsequent reports', 'an offline median of measured values in the next five grid rows, requiring at least three measured values')
    theory = theory.replace('a finer output grid alone does not demonstrate improved measurement resolution.', 'a finer output grid alone does not demonstrate improved measurement resolution. In particular, intra-ELM dynamics below the delivered grid are not independently resolved by the present targets.')
    (OUT/'integrated_introduction_en.tex').write_text(intro.rstrip()+'\n', encoding='utf8')
    (OUT/'integrated_theory_en.tex').write_text(theory.rstrip()+'\n', encoding='utf8')
    path = PAPER/'main_ppcf.tex'
    text = path.read_text(encoding='utf8')
    start = text.index('\\section{Introduction}')
    data_heading = '\\section{Data and the reported value}' if '\\section{Data and the reported value}' in text else '\\section{Materials and methods}'
    end = text.index(data_heading, start)
    text = text[:start] + intro + '\n' + theory + '\n' + text[end:]
    text = text.replace('\\section{Data and the reported value}', '\\section{Materials and methods}')
    text = text.replace('\\subsection{Files, chord and rows}', '\\subsection{Data set: files, chord and collection criteria}')
    marker = '\\subsection{Absent frames and the age of a value}'
    if '\\subsubsection{Campaign and device-state classification}' not in text:
        text = text.replace(marker, r'''\subsubsection{Campaign and device-state classification}
All files used here belong to the KSTAR campaign identified above. A shot-level classification by confinement regime, RMP state or device modification has not been established from verified metadata. The validation partitions below are statistical divisions, not physical discharge classes. A future campaign/device-state comparison requires verified operational metadata and a separate registered analysis.

''' + marker)
    text = text.replace('\\section{Method}', '\\subsection{Predictor design and evaluation}')
    start = text.index('\\subsection{Predictor design and evaluation}')
    end = text.index('\\section{Results}', start)
    part = text[start:end]
    part = part.replace('\\subsection{', '\\subsubsection{').replace('\\subsubsection{Predictor design and evaluation}', '\\subsection{Predictor design and evaluation}')
    marker = '\\subsubsection{Estimators and references}'
    if '\\subsubsection{Neural-network architecture and modification criteria}' not in part:
        part = part.replace(marker, r'''\subsubsection{Neural-network architecture and modification criteria}
Architecture selection is an empirical question. The intended first stage compares CNN or lightweight temporal candidates with history-only models and the reference estimators. Regression errors, uncertainty, modality contributions and computational costs determine whether and how a later CES-specific network should be modified. Completed comparisons below retain their stated training budgets and scoring support; they are not evidence that every CNN candidate has already been assessed. Subsequent redesign remains a research plan requiring registration before new fits.

''' + marker)
    text = text[:start] + part + text[end:]
    path.write_text(text, encoding='utf8')
    # Add only cited, individually audited records absent from the existing file.
    bib_path = PAPER/'refs.bib'
    bib = bib_path.read_text(encoding='utf8')
    existing = B.entries(bib)
    keys = sorted({k for g in re.findall(r'\\citep\{([^}]+)', intro+theory) for k in g.split(',')})
    added = []
    for key in keys:
        if key in existing:
            continue
        candidates = list(OUT.glob('paper_'+key+'.json')) + list(OUT.glob('background_'+key+'.json'))
        audit = json.loads(candidates[0].read_text(encoding='utf8'))
        entry = audit.get('bibtex') or audit.get('bibtex_if_worth_citing')
        assert isinstance(entry, str) and key in B.entries(entry), key
        bib += '\n\n' + entry + '\n'
        added.append(key)
    bib_path.write_text(bib, encoding='utf8')
    previous_path = OUT/'host_integration.json'
    previous = json.loads(previous_path.read_text(encoding='utf8')) if previous_path.exists() else {}
    added = sorted(set(added + previous.get('added_bibtex_keys', [])))
    previous_path.write_text(json.dumps({'added_bibtex_keys':added,
        'cited_source_keys':keys, 'host_corrections':replacements,
        'new_scientific_fits':0, 'whole_manuscript_regraded':False}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('INTEGRATED', len(keys), 'cited sources;', len(added), 'new bibliography entries')


if __name__ == '__main__':
    main()
