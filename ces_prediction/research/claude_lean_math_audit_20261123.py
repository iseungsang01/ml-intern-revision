"""Adversarial algebra/identification check of concrete review discrepancies."""
import claude_lean_scout_20261123 as S

PROMPT=S.COMMON+'''
Final targeted mathematical audit. No search is needed: derive the statements below independently rather than treating another review as authority. No numerical/synthetic data or new empirical results are requested. Answer in Korean under1000words. Mark all mathematics as informal/unchecked by Lean.

1. Stationary mean-zero OU latent X has variance S>0 and Cov(X_t,X_s)=S*rho, rho=exp(-theta*a), theta>0. An old noisy report C=X_s+e_s and an independent new report Y=X_t+e_t have independent zero-mean noises with variances r_old,r_new. Derive report-target MSE for zero-order hold C and contrast the optimal noiseless-state conditional mean rho*X_s. A prior review called S*(1-rho^2)+noise the HOLD error. Is that correct? State what changes when report errors are correlated.

2. In a distinct jointly Gaussian model, C=X_s+epsilon (variance r>=0), Z_j=h*X_j+b+eta_j, independent eta variance q>0 at distinct sample times, h nonzero, same fixed unknown shot offset b. Set k=S/(S+r), u=1-rho. Derive the Bayes MSE reduction for estimating X_t from (C,D) rather than C alone, where D=Z_t-Z_s. Check the proposed formula G=B*u^2*(1+k-k*u)^2/[A*u*(2-k*u)+2*q], A=h^2*S, B=h^2*S^2. Is this variance formula valid for arbitrary non-Gaussian noises, or then only linear-MMSE?

Independently check derivative certificate:
dG/du = 2*B*u*(1+k-k*u)*[A*u*(1+k-3*k*u+k^2*u^2)+2*q*(1+k-2*k*u)] / [A*u*(2-k*u)+2*q]^2.
The polynomial1+k-3*k*u+k^2*u^2 equals (1-k)^2+k*(1-u)*(3-k*(u+1)); the second bracket1+k-2*k*u equals1-k+2*k*(1-u). Does this establish nondecreasing gain for0<=u,k<=1 under the assumptions? What happens with independent offset drift Var(b_t-b_s)=tau^2*a? Is the O(a^2) small-age statement contingent on q>0?

3. The ACTUAL implemented network concatenates nonlinear current/anchor embeddings; the selected TCN uses current fast proxies directly. It does not restrict itself to D. For current unanchored Z=h*X_t+eta, prior P(a)=S*(1-k*rho^2), gain h^2*P(a)^2/[h^2*P(a)+q]. At r>0 the limit as a decreases to0 is positive. Does this refute a universal zero-age-gain claim for our actual model? Distinguish an independent report/state target from literally revealing the same current report that is the label. Do not call BES/ECEI a calibrated Ti thermometer.

4. Distinguish assigned masking policy D, realized source age A=g(W,D), and label validity V=v(W), where W is a recorded trajectory. With independent assigned D and unchanged labels, contrast E[L_D(W)|D=d,V=1] and E[L_d(W)|V=1]. What positivity/finite-risk/frozen-model assumptions are needed? Does randomizing D make realized A independent of regime W? Does deterministic every-kth sampling remove this confounding? If every policy can instead be replayed on the SAME complete observed trace and valid-label rows, can policy risk differences be evaluated without random assignment? Explain why none of this identifies unobserved-label risk or physical causal effects; distinguish a frozen model selected on reused validation data from independent prospective risk.

5. A prior review said zeroing fast-input stem columns establishes model-class nesting only almost surely. Is that right, or is the structural function equivalence pointwise for every valid input in the ideal model? Separate this from a.e. equality of conditional expectations and from finite-budget fitted-risk ordering.

Finish with corrected conclusions that can safely enter a methods appendix, what remains a standard result rather than originality, and whether Lean would add value beyond a symbolic algebra certificate for these exact claims.
'''

if __name__=='__main__':S.call('mathematical_audit',PROMPT,'claude-opus-5-5',S.API.credential())
