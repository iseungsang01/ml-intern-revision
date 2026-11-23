"""Pure symbolic identities; not a data experiment, simulation or Lean proof."""
import json
from pathlib import Path
import sympy as s
ROOT=Path(__file__).resolve().parents[2]

def main():
    checks=[]
    def eq(name,lhs,rhs):
        assert s.simplify(lhs-rhs)==0,name
        checks.append(name)
    tau,D,a,dt,S,r=s.symbols('tau D a dt S r',positive=True)
    u,v=s.symbols('u v',real=True)
    # Assumes a>=D so the averaging windows do not overlap.
    cross=S/D**2*s.integrate(s.integrate(s.exp(-(v-u)/tau),(v,a,a+D)),(u,0,D))
    target=S*s.exp(-a/tau)*(tau/D)**2*(s.exp(D/tau)-1)*(1-s.exp(-D/tau))
    eq('nonoverlapping_OU_exposure_covariance',cross,target)
    var=2*S/D**2*s.integrate(s.integrate(s.exp(-(u-v)/tau),(v,0,u)),(u,0,D))
    targetvar=2*S*(tau/D)**2*(D/tau-1+s.exp(-D/tau))
    eq('OU_exposure_variance',var,targetvar)
    eq('exposure_variance_point_limit',s.limit(var,D,0),S)
    m,ys,zs,I=s.symbols('m ys zs I',real=True)
    rho=s.exp(-dt/tau)
    integral=s.integrate(s.exp(-(dt-v)/tau)*m/tau,(v,0,dt))
    eq('left_ZOH_forcing_integral',integral,(1-rho)*m)
    eq('auxiliary_filter_anchor_cancellation',(rho*zs+I)+rho*(ys-zs),rho*ys+I)
    x,y=s.symbols('x y',real=True)
    eq('same_forcing_incremental_dissipation',(x-y)*((m-x)/tau-(m-y)/tau),-(x-y)**2/tau)
    p,T1,T2,mass,u1,u2=s.symbols('p T1 T2 mass u1 u2',real=True)
    mu=p*u1+(1-p)*u2
    second=p*(T1/mass+u1**2)+(1-p)*(T2/mass+u2**2)-mu**2
    eq('CES_two_component_total_variance',second,(p*T1+(1-p)*T2)/mass+p*(1-p)*(u1-u2)**2)
    gain=S*s.exp(-a/tau)/(S+r)
    eq('noisy_point_OU_mean_coefficient',gain,s.exp(-a/tau)*(S/(S+r)))
    eq('noisy_point_OU_predictive_variance',S+r-(S*s.exp(-a/tau))**2/(S+r),S+r-gain**2*(S+r))
    k,beta,deltaV,deltaP,deltaB=s.symbols('k beta deltaV deltaP deltaB',real=True)
    eq('beam_log_normalization_is_coefficient_constraint',beta*(deltaV-deltaP+s.Rational(3,2)*deltaB),beta*deltaV-beta*deltaP+s.Rational(3,2)*beta*deltaB)
    record={'status':'exact_symbolic_identities_passed','checks':checks,'sympy_version':s.__version__,
            'assumptions':'OU windows nonoverlapping a>=D; stated ODE and mixture models; no actual observations generated or read',
            'not_established':['physical validity','parameter identifiability','Lean proof','predictive benefit','L5 median expectation']}
    path=ROOT/'data/.claude_physics_20261123/algebra_checks.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))

if __name__=='__main__':main()
