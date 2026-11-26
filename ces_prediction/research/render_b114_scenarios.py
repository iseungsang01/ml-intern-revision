"""Render already-audited B114 scenario comparisons; no fits or new readout.

Consumers: docs/paper/b114_scenario_summary.md and b114_scenario_summary.png;
root README/HANDOVER link the summary. Every estimate/interval is copied from
b114_comparison.json using the pre-TEST per-unit overall VAL family choice.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'docs/paper'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import TwoSlopeNorm
 p=PAPER/'b114_comparison.json';z=json.loads(p.read_text(encoding='utf8'))
 audit=json.loads((ROOT/'data/.b114/audit.json').read_text());assert audit==z['independent_audit'] and audit['all_pass']
 selection=json.loads((ROOT/'data/.b114/selection.json').read_text());assert selection==z['selection']
 assert sha(ROOT/'data/.b114/selection.json')==audit['selection_sha256']
 for rel,h in audit['sources'].items():assert sha(ROOT/rel)==h
 tags=['U0','U1','U2','U3'];units=['V1_r2_k0','V2_s1_k0','V3_q3','V4a']
 rules=['D10','D50','D100','D300','D650','D1000','S','K10']
 lines=['# GP+fast와 VAL 선택 최상위 후보의 시나리오 비교','',
 '새 학습·새 통계 판독 없이, 완료·독립 감사된 B114 결과에서 기존 수치만 가져온 요약이다. 후보는 각 외부 검증 설계의 내부 VAL에서 선택한 계열·설정으로 고정한다. 시나리오별 TEST 결과에 따라 후보를 바꾸지 않는다.',
 '', '최근 B114 탐색 안에서 VAL 기준으로 선택한 후보이며, 모든 과거 모델의 동일 VAL 비교를 완료했다는 뜻은 아니다. 기존 TEST를 재사용한 탐색 결과이고 구간은 선택·적합에 조건부인 점별 구간이다.','',
 '| 검증 단위 | VAL 선택 계열 | 구조/설정 | 학습 형태 |','|---|---|---|---|']
 chosen={}
 for tag,unit in zip(tags,units):
  f=selection['overall_family_by_unit'][tag];c=selection[f]['by_unit'][tag]
  assert f==min(('cat','neural'),key=lambda k:(selection[k]['by_unit'][tag]['score'],k))
  chosen[unit]='Cat_selected_ensemble' if f=='cat' else 'NN_selected_ensemble'
  lines.append(f"| {unit} | {f} | {c.get('config',c.get('architecture'))} | {c['form']} |")
 lines+=['','시나리오: Dは 관측값 지연(ms), S는 가장 최근 notch 직전 CES까지만 가용(분석시간0 가정), K10은 10번째 프레임마다 CES가 주어지는 조건이다. 모든 표적은 현재 프레임 보고값 F다.'.replace('Dは','D는'),'',
 '**기준은 모두 해당 시나리오의 GP+fast다. MSE 감소율(%) [95% shot-cluster 구간]. 양수는 후보의 오차 감소, 음수는 오차 증가. 별표(*)는 구간이0을 제외한다.**','',
 '| 시나리오 | '+' | '.join(units)+' |','|---|'+'---:|'*4]
 grid=[];annotations=[];detail=[]
 for rule in rules:
  values=[];notes=[];row=[]
  for unit in units:
   key=unit+'/'+rule+'/F';cell=z['cells'][key];name=chosen[unit];pair=cell['pairs'][name+'_over_GP_fast_boost'];lo,hi=pair['cluster']['ci95'];point=pair['point']
   assert pair['cluster']['excludes_zero']==bool(lo>0 or hi<0)
   star='*' if pair['cluster']['excludes_zero'] else ''
   row.append(f'{point*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]{star}')
   values.append(point*100);notes.append(f'{point*100:+.2f}%{star}\n[{lo*100:+.2f}, {hi*100:+.2f}]')
   detail.append((key,name,cell['rows'],cell['models'][name],cell['models']['GP_fast_boost']))
  grid.append(values);annotations.append(notes);lines.append('| '+rule+' | '+' | '.join(row)+' |')
 lines+=['','## 같은 행에서의 절대 오차','', '| 단위/시나리오/F | 후보 | 행 수 | GP+fast MSE [95%] | 후보 MSE [95%] | GP+fast RMSE | 후보 RMSE |','|---|---|---:|---|---|---:|---:|']
 for key,name,n,m,g in detail:
  def mse(v):return f"{v['MSE']:.3f} [{v['cluster_MSE95'][0]:.3f}, {v['cluster_MSE95'][1]:.3f}]"
  lines.append(f"| {key} | {name} | {n} | {mse(g)} | {mse(m)} | {g['RMSE']:.3f} | {m['RMSE']:.3f} |")
 lines+=['','MSE 단위 eV², RMSE 단위 eV. 방전 재표집 구간·각 초기화·두 계열 전체 결과는 원본 b114_comparison.json/md에 보존돼 있다. 학습 예산·TEST 재사용·선택 불확실성 한계를 유지하며, 어느 설계도 주 판독으로 부르지 않는다.','',
 '음의 결과를 뒤집으려면 사용하지 않은 관측에서 동일한 입력·표적·충분한 예산으로 양의 구간을 확인해야 한다. 이번 표는 새로운 물리나 실시간 계산 지연을 입증하지 않는다.','',
 '원본: docs/paper/b114_comparison.json; SHA256: '+sha(p), '재생성: py ces_prediction/research/render_b114_scenarios.py','']
 # Historical candidates are displayed without selecting a TEST minimum.
 historical=[('b111_comparison.json',('LSTM_frozen_D10','LSTM_mixed_post')),
             ('b112_comparison.json',('MLP_direct_ensemble','MLP_residual_ensemble')),
             ('b113_comparison.json',('MLP_direct_ensemble','MLP_residual_ensemble'))]
 # B113 contains its own CatBoost names, not the earlier MLP candidates.
 historical[-1]=('b113_comparison.json',('CatBoost_direct_ensemble','CatBoost_residual_ensemble'))
 lines+=['','## 과거 주요 후보의 같은 시나리오 대비','',
         '아래는 각 완료 배치가 보존한 GP+fast 대비 점추정과 구간을 그대로 가져온 참고표다. TEST 최소값으로 새 모델을 선택하지 않는다. 배치별 학습 예산·초기화 수·bootstrap seed가 달라 구조만의 비교가 아니며, 모든 과거 실험을 망라한 순위도 아니다. LSTM frozen_D10은 D10에서 학습한 후단 보정을 고정한 것이고 mixed_post는 가용 조건을 섞은 후단 보정이다.',
         '', '| 원천 | 후보 | 시나리오 | '+' | '.join(units)+' |','|---|---|---|'+'---:|'*4]
 for filename,names in historical:
  oldpath=PAPER/filename;old=json.loads(oldpath.read_text(encoding='utf8'));assert old['independent_audit']['all_pass']
  for name in names:
   for rule in rules:
    vals=[]
    for unit in units:
     key=unit+'/'+rule+'/F';v=old['cells'][key];current=z['cells'][key]
     assert v['rows']==current['rows']
     assert abs(v['models']['GP_fast_boost']['MSE']-current['models']['GP_fast_boost']['MSE'])<1e-7
     pair=v['pairs'][name+'_over_GP_fast_boost'];lo,hi=pair['cluster']['ci95']
     vals.append(f"{100*pair['point']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"+('*' if pair['cluster']['excludes_zero'] else ''))
    lines.append('| '+filename+' | '+name+' | '+rule+' | '+' | '.join(vals)+' |')
  lines+=['','원천 '+filename+' SHA256: '+sha(oldpath),'']
 (PAPER/'b114_scenario_summary.md').write_text('\n'.join(lines),encoding='utf8')
 plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
 a=np.array(grid);limit=max(1,float(np.abs(a).max()))
 fig,ax=plt.subplots(figsize=(12,9));im=ax.imshow(a,cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
 ax.set_xticks(range(4),units);ax.set_yticks(range(8),rules);ax.xaxis.tick_top()
 for i in range(8):
  for j in range(4):ax.text(j,i,annotations[i][j],ha='center',va='center',fontsize=10,color='white' if abs(a[i,j])>.6*limit else 'black')
 ax.set_title('GP+fast 대비 VAL 선택 후보의 MSE 감소율\n양수: 오차 감소 / 음수: 오차 증가 · [95% 군집 구간] · *: 영 제외',pad=25,fontsize=14)
 fig.colorbar(im,ax=ax,label='MSE 감소율 (%)',shrink=.8)
 fig.text(.5,.025,'B114 각 설계의 기존 VAL 선택 고정 · 현재 보고값 F · TEST 재사용 탐색 · 새 학습/통계 계산 없음',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.04,1,.97]);fig.savefig(PAPER/'b114_scenario_summary.png',dpi=160);plt.close(fig)
 print('Rendered32 existing audited comparisons; no new fits, predictions or bootstrap')
 print('Chosen',chosen)
 for rule,values in zip(rules,grid):print(rule,[round(v,2) for v in values])
if __name__=='__main__':main()
