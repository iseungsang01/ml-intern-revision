import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd, sys
D = pd.read_pickle(sys.argv[1])
S=D.S.values; Sp=D.Sprev.values; R=D.R.values; Rp=D.Rprev.values; y=D.y.values
print("rows", len(D), "S defined share %.3f"%np.isfinite(S).mean())
classes = {
 "E_up  (S>+0.3)": S>0.3,
 "E_dn  (S<-0.3)": S<-0.3,
 "mid   (0.1<|S|<=0.3)": (np.abs(S)>0.1)&(np.abs(S)<=0.3),
 "quiet (|S|<=0.1)": np.abs(S)<=0.1,
 "S undefined": ~np.isfinite(S),
}
pairs=[("se_c2","se_gp","two-stage vs causal GP"),("se_c2","se_t2","two-stage vs two-stage twin"),("se_c","se_t","network vs twin"),("se_c","se_gp","network vs causal GP"),("se_c2","se_hold","two-stage vs hold")]
tot={k:D[k].sum() for k in ["se_c2","se_t2","se_c","se_t","se_gp","se_hold","se_off"]}
print("\nclass                   rows   share  | share of SSE: hold   GP   twin2  two-stage | mean err (est-y) two-stage / twin2 / GP? ")
for name,m in classes.items():
    print("%-22s %7d  %.3f |  %.3f  %.3f  %.3f  %.3f | %+.1f / %+.1f eV ; median y %.0f eV"%(name,m.sum()/3,m.mean(),D.loc[m,"se_hold"].sum()/tot["se_hold"],D.loc[m,"se_gp"].sum()/tot["se_gp"],D.loc[m,"se_t2"].sum()/tot["se_t2"],D.loc[m,"se_c2"].sum()/tot["se_c2"],D.loc[m,"e_c2"].mean(),D.loc[m,"e_t2"].mean(),np.median(y[m])))
print()
for a,b,lab in pairs:
    red_tot = tot[b]-tot[a]
    print(lab, " overall skill %.3f"%(1-tot[a]/tot[b]))
    for name,m in classes.items():
        ra=D.loc[m,a].sum(); rb=D.loc[m,b].sum()
        print("   %-22s skill inside %+.3f   share of the total SSE reduction %.3f"%(name,1-ra/rb,(rb-ra)/red_tot))
# Combined 'non-thermal' set: the row is a single-frame excursion beyond 0.3, or the previous row was (history poisoned)
nt = (np.abs(S)>0.3)|(np.abs(Sp)>0.3)
print("\nrows that are a >0.3 single-frame excursion or directly follow one: share %.3f"%nt.mean())
for a,b,lab in pairs:
    red_tot = tot[b]-tot[a]
    ra=D.loc[nt,a].sum(); rb=D.loc[nt,b].sum(); ra2=D.loc[~nt,a].sum(); rb2=D.loc[~nt,b].sum()
    print("   %-28s inside: skill %+.3f, share of reduction %.3f | outside: skill %+.3f"%(lab,1-ra/rb,(rb-ra)/red_tot,1-ra2/rb2))
print("   share of SSE inside: hold %.3f GP %.3f two-stage %.3f"%(D.loc[nt,"se_hold"].sum()/tot["se_hold"],D.loc[nt,"se_gp"].sum()/tot["se_gp"],D.loc[nt,"se_c2"].sum()/tot["se_c2"]))
# ratio-to-local-median classes
print("\nratio to the median of the 10 neighbouring rows:")
rc = {"R>2":R>2,"1.5<R<=2":(R>1.5)&(R<=2),"0.67<=R<=1.5":(R>=0.67)&(R<=1.5),"0.5<=R<0.67":(R>=0.5)&(R<0.67),"R<0.5":R<0.5,"R undefined":~np.isfinite(R)}
for name,m in rc.items():
    line="   %-14s share %.4f | SSE share hold %.3f GP %.3f two-stage %.3f |"%(name,m.mean(),D.loc[m,"se_hold"].sum()/tot["se_hold"],D.loc[m,"se_gp"].sum()/tot["se_gp"],D.loc[m,"se_c2"].sum()/tot["se_c2"])
    for a,b,lab in pairs[:3]:
        ra=D.loc[m,a].sum(); rb=D.loc[m,b].sum()
        line+=" %s: %+.3f (share of red. %.3f);"%(lab.split(" vs ")[0][:9]+"/"+lab.split(" vs ")[1][:9],1-ra/rb,(rb-ra)/(tot[b]-tot[a]))
    print(line)
# two-level bootstrap for the key shares
rng=np.random.default_rng(20261120)
def stats(d):
    S=d.S.values; Sp=d.Sprev.values
    nt=(np.abs(S)>0.3)|(np.abs(Sp)>0.3)
    o={}
    for a,b,k in [("se_c2","se_gp","gp"),("se_c2","se_t2","tw")]:
        A=d[a].values;B=d[b].values
        o["share_"+k]=(B[nt].sum()-A[nt].sum())/(B.sum()-A.sum())
        o["out_"+k]=1-A[~nt].sum()/B[~nt].sum()
        o["in_"+k]=1-A[nt].sum()/B[nt].sum()
    up=S>0.3
    o["up_tw"]=1-d.se_c2.values[up].sum()/d.se_t2.values[up].sum()
    o["up_net_tw"]=1-d.se_c.values[up].sum()/d.se_t.values[up].sum()
    return o
# per-shot sufficient statistics to bootstrap quickly
res=[]
per={}
for r in (2,3,4):
    d=D[D.r==r]
    S=d.S.values; Sp=d.Sprev.values
    nt=(np.abs(S)>0.3)|(np.abs(Sp)>0.3); up=S>0.3
    t=pd.DataFrame({"shot":d.shot.values,
      "c2_in":d.se_c2.values*nt,"gp_in":d.se_gp.values*nt,"t2_in":d.se_t2.values*nt,
      "c2_out":d.se_c2.values*~nt,"gp_out":d.se_gp.values*~nt,"t2_out":d.se_t2.values*~nt,
      "c2_up":d.se_c2.values*up,"t2_up":d.se_t2.values*up,"c_up":d.se_c.values*up,"t_up":d.se_t.values*up,
      "c_in":d.se_c.values*nt,"t_in":d.se_t.values*nt,"c_out":d.se_c.values*~nt,"t_out":d.se_t.values*~nt}).groupby("shot").sum()
    per[r]=t.values; cols=list(t.columns)
ci={k:[] for k in ["share_gp","share_tw","out_gp","out_tw","in_gp","in_tw","up_tw","up_net_tw","out_net_tw","in_net_tw"]}
for b in range(2000):
    r=rng.choice([2,3,4]); M=per[r]; idx=rng.integers(0,len(M),len(M)); s=M[idx].sum(0); v=dict(zip(cols,s))
    ci["share_gp"].append((v["gp_in"]-v["c2_in"])/((v["gp_in"]+v["gp_out"])-(v["c2_in"]+v["c2_out"])))
    ci["share_tw"].append((v["t2_in"]-v["c2_in"])/((v["t2_in"]+v["t2_out"])-(v["c2_in"]+v["c2_out"])))
    ci["out_gp"].append(1-v["c2_out"]/v["gp_out"]); ci["out_tw"].append(1-v["c2_out"]/v["t2_out"])
    ci["in_gp"].append(1-v["c2_in"]/v["gp_in"]); ci["in_tw"].append(1-v["c2_in"]/v["t2_in"])
    ci["up_tw"].append(1-v["c2_up"]/v["t2_up"]); ci["up_net_tw"].append(1-v["c_up"]/v["t_up"])
    ci["out_net_tw"].append(1-v["c_out"]/v["t_out"]); ci["in_net_tw"].append(1-v["c_in"]/v["t_in"])
print("\n95% two-level intervals (re-partition drawn, discharges resampled, 2000 draws):")
for k,v in ci.items():
    print("   %-12s %+.3f [%+.3f, %+.3f]"%(k,np.mean([np.nan]) if False else np.median(v),np.quantile(v,0.025),np.quantile(v,0.975)))
