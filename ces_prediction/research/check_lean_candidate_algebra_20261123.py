"""Exact symbolic identities, NOT Lean proofs or empirical/synthetic data.

Assumptions and interpretation live in docs/paper/lean_feasibility_20261123.md.
No measurements, models, training artifacts or TEST observations are read.
"""
import json
from pathlib import Path
import sympy as s

ROOT=Path(__file__).resolve().parents[2]

def main():
    checks=[]
    def eq(name,left,right):
        if s.simplify(left-right)!=0:
            raise AssertionError(name)
        checks.append(name)
    S,r,h,q,rho=s.symbols('S r h q rho',positive=True)
    k,u,A,B=s.symbols('k u A B',real=True)
    # Independent reporting errors; no cross-covariances with the latent state.
    eq('hold_minus_noiseless_optimum',2*S*(1-rho)-S*(1-rho**2),S*(1-rho)**2)
    # Gaussian conditioning (or linear residual covariances without Gaussianity).
    cov=h*S*(1-rho)-(rho*S)*(h*S*(rho-1))/(S+r)
    var=2*h**2*S*(1-rho)+2*q-(h*S*(rho-1))**2/(S+r)
    cov_u=h*S*u*(1+k-k*u)
    var_u=h**2*S*u*(2-k*u)+2*q
    eq('conditional_cross_covariance',cov.subs(rho,1-u),cov_u.subs(k,S/(S+r)))
    eq('conditional_proxy_variance',var.subs(rho,1-u),var_u.subs(k,S/(S+r)))
    den=A*u*(2-k*u)+2*q
    gain=B*u**2*(1+k-k*u)**2/den
    poly=1+k-3*k*u+k**2*u**2
    cert=2*B*u*(1+k-k*u)*(A*u*poly+2*q*(1+k-2*k*u))/den**2
    eq('gain_derivative_certificate',s.diff(gain,u),cert)
    eq('nonnegative_polynomial_decomposition',poly,(1-k)**2+k*(1-u)*(3-k*(u+1)))
    eq('nonnegative_linear_decomposition',1+k-2*k*u,(1-k)+2*k*(1-u))
    eq('small_age_q_positive_u_squared_coefficient',s.limit(gain/u**2,u,0),B*(1+k)**2/(2*q))
    eq('small_age_q_zero_u_coefficient',s.limit(gain.subs(q,0)/u,u,0),B*(1+k)**2/(2*A))
    P=S*(1-(S/(S+r))*rho**2)
    direct=h**2*P**2/(h**2*P+q)
    eq('positive_zero_age_direct_proxy_example',direct.subs({S:1,r:1,h:1,q:1,rho:1}),s.Rational(1,6))
    result={'status':'exact_symbolic_identity_checks_passed','checks':checks,'sympy_version':s.__version__,
            'scope':'No Lean checking; no empirical or synthetic observations; polynomial sign argument and model assumptions require the accompanying report.'}
    dest=ROOT/'data/.claude_lean_20261123/algebra_checks.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':main()
