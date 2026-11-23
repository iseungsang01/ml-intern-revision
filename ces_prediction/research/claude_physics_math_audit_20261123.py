"""Targeted correction of physical-model formulas; no measurements transmitted."""
import json
import claude_physics_scout_20261123 as P
S=P.S

PROMPT=P.COMMON+'''
Final mathematical audit, not another broad search. Search is optional; independently derive the following formulas and reject overclaims. Answer in Korean under1000words. No Lean proof, no empirical results.

We want a useful simple physical formulation involving FAST inputs, rather than recommending only a familiar history-only OU process. Proposed minimal grey-box: temperature variable U is in eV, tau>0 seconds, and tau*dU/dt+U=m(t), m=c+beta_E*E(t)+beta_B*B(t)+beta_P*P(t). E/B are fixed TRAIN-standardized ECEI/BES group-mean voltages, NOT calibrated Te/ne; P is injected MW, NOT absorbed local heating. Parameter count5 (tau,c,three gains). Free-sign gains. Source-free Eulerian internal energy equation is (3/2)*n*(partial_t+v dot grad)*T+n*T*div(v)=-div(q)-Pi:grad(v)+Q, T in joules. Explain assumptions needed for stable local linear lumping and why fitted tau is not an identified transport/collision time. Stability of a fit or a good predictor need not prove these assumptions.

For exact ZOH driving on each interval with LEFT-END input, a_j=exp(-dt_j/tau); U_j=a_j U_(j-1)+(1-a_j)*m_(j-1). Starting at permitted anchor s with U_s=y_s is a deterministic report-domain approximation, not a Bayes-optimal noisy-state estimate. Derive convolution and state-reset conventions. Show same-tau auxiliary filter z permits Uhat(t)=z(t)+exp(-(t-s)/tau)*(y_s-z(s)), independent of arbitrary initial z when both z(t),z(s) use the same filter dynamics. This allows one-pass causal implementation with stored z(s). Note current fast sample at t does NOT causally drive the integral over (t-dt,t) under left-ZOH; using current sample there is a retrospective/right-end quadrature or different hold assumption. Can a direct-feedthrough nested comparator address current sample information without pretending strict left-ZOH physics? Specify parameter count and no new blind hyperparameter search. Hold is a limiting comparator (large tau); zero fast gains still leaves an intercept relaxation, which is different from hold. No guarantee of positivity if gains free.

For OU U stationary mean mu, variance S, corr exp(-abs(dt)/tau), define non-overlapping UNIFORMLY averaged observations A_s=(1/Delta)*integral_[s,s+Delta]U du and A_t analog at t, a=t-s>=Delta. Check Cov(A_s,A_t)=S*exp(-a/tau)*(tau/Delta)^2*(exp(Delta/tau)-1)*(1-exp(-Delta/tau)), and Var(A)=2*S*(tau/Delta)^2*(Delta/tau-1+exp(-Delta/tau)). Add independent report noise r; give conditional mean of new report given ONE old averaged report and contrast with full-history Kalman posterior. Exposure creates correlated process/measurement innovations, not plain scalar point-observation Kalman. Exact Gaussian moment observations are already a conditional model; actual CES Gaussian-fit reports need not be uniform averages.

The deterministic ODE can also average its future exposure analytically ONLY under an explicit frozen/future-input model and process assumptions. F's actual future exposure includes unobserved future forcing; it is not computable exactly from current information. L5 is median of next valid5 reports with validity>=3: averaging rho does NOT compute its conditional mean, generally. Recommend a clean F-only physical-model hypothesis first, and treat L5 extension separately if needed, preserving the paper's two-target reporting rather than recasting L5 as an instantaneous state.

Physics alternatives to compare briefly: CES second moment mixture identity, and deltaT_E=deltaT_L-xi*gradT. deltaT_L can include reversible compression, NOT only irreversible heating. Rank1 PCA has learned loadings as parameters and does not measure displacement; pure 0/pi cross-phase is not universal with separated channel geometry, mode toroidal/poloidal phase and diagnostic transfer. BES normalization P/Vb^(3/2) needs source mapping/fractions/beam cross-section; actual beam-off rows may or may not coexist with past anchors, so do not assert their absence without checking. Independent main-ion temperature, not equilibrium alone, is needed to verify TC=TD. Do not call a performance gap vs trees a pure measurement of nonlinearity; inputs, regularization and fit error also differ.
'''

if __name__=='__main__':
    r=S.call('targeted_physics_math_audit',PROMPT,'claude-opus-5-5',S.API.credential())
    P.audit()
    print(json.dumps({'theme':r['theme'],'stop':r['stop_reason']}))
