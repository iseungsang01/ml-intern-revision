"""Verified CPU-only B115 figures/records closeout; no fitting or new score choice."""
import subprocess,sys,time,json
from pathlib import Path
import common115 as C
PY=Path(sys.executable).with_name('python.exe')
def main():
 z=C.read(C.ROOT/'docs/paper/b115_comparison.json');assert z['independent_audit']==C.read(C.OUT/'audit.json') and z['independent_audit']['all_pass']
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
 keys=[k for rule in ('D10','S') for k in z['cells'] if '/'+rule+'/' in k]
 fig,ax=plt.subplots(figsize=(11,7))
 for i,k in enumerate(keys):
  cell=z['cells'][k]
  for offset,prefix,color,label in [(-.12,'NN_base','gray','기존 학습'),(.12,'NN_long','#0072B2','연장·학습률 조정')]:
   p=cell['pairs'][prefix+'_ensemble_over_GP_fast_boost'];lo,hi=p['cluster']['ci95'];v=p['point']
   ax.hlines(i+offset,100*lo,100*hi,color=color)
   ax.plot(100*v,i+offset,'o',color=color,label=label if i==0 else None)
 ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(keys)),keys);ax.invert_yaxis();ax.legend();ax.set_xlabel('GP+fast 대비 MSE 감소율 (%) [95% 군집 구간]');ax.set_title('동일 모델의 학습 연장 전·후: 고정된 평가 행과 재표집 단위')
 fig.tight_layout();fig.savefig(C.ROOT/'docs/paper/b115_training_effect.png',dpi=160);plt.close(fig)
 text='B115 COMPLETE / independently audited. Twelve extended-policyGPUfits and32scenario evaluations normal. Registered statements: '+json.dumps(z['statements'])+'. Original groups/confidencelevel preserved; actual CI widths reported without forced narrowing. No further fits queued.'
 paragraph='''기존에 선택된 신경망 구조·학습 형태·초기화를 고정하고, 조기 종료를 제거한 학습 연장과 단계별 학습률 감소를 시험했다. 원래 학습 구간의 재현, 학습 전용 정규화·가중치·인코딩과 새로운 최적 체크포인트, 독립 순전파·paired 구간 검산을 통과했다. 상세 점추정·구간·영제외 여부와 실제 구간 너비비는 원천 표를 따른다. 평가 군집을 늘리거나 바꾸지 않았으며, 재학습이 독립 평가 자료를 추가한 것으로 해석하지 않는다. 기존 TEST 재사용과 체크포인트 선택에 조건부인 탐색 결과다. 사용하지 않은 관측에서 예산·입력·표적을 맞춘 비교가 결론을 뒤집을 수 있다.\n'''
 p=C.ROOT/'PROJECT_KNOWLEDGE.md';s=p.read_text(encoding='utf8');assert '### 4.83 ' not in s;s=s.replace('## 5. 코드와 산출물','### 4.83 신경망 학습 연장과 군집 오차 점검 (B.115; `docs/paper/b115_comparison.md`)\n\n'+paragraph+'\n## 5. 코드와 산출물',1);p.write_text(s,encoding='utf8',newline='\n')
 p=C.ROOT/'THESIS_RESULTS.md';p.write_text(p.read_text(encoding='utf8')+'\n\n## B115 registered diagnostics and extended training\n\n'+text+'\n\n'+paragraph,encoding='utf8',newline='\n')
 p=C.ROOT/'HANDOVER.md';s=p.read_text(encoding='utf8');i=s.find('\n')+1;p.write_text(s[:i]+'\n**'+text+'**\n'+s[i:],encoding='utf8',newline='\n')
 for rel in ('README.md','ces_prediction/experiments/README.md'):
  p=C.ROOT/rel;p.write_text(p.read_text(encoding='utf8').replace('B115 PREPARED','B115 COMPLETE / independently audited'),encoding='utf8',newline='\n')
 (C.ROOT/'docs/paper/b115_status.md').write_text('# B115 complete\n\n'+text+'\n\n'+paragraph,encoding='utf8')
 p=C.ROOT/'ces_prediction/check_records.py';s=p.read_text(encoding='utf8');assert 'docs/paper/b115_comparison.md' not in s;p.write_text(s.replace('SOURCES = (','SOURCES = ('+chr(10)+'    '+chr(34)+'docs/paper/b115_comparison.md'+chr(34)+',',1),encoding='utf8',newline='\n')
 subprocess.run([str(PY),'ces_prediction/check_records.py'],cwd=C.ROOT,check=True);subprocess.run([str(PY),'-m','pytest','-q'],cwd=C.ROOT,check=True)
 assert subprocess.check_output(['git','branch','--show-current'],cwd=C.ROOT,text=True).strip()=='main'
 files=['README.md','HANDOVER.md','THESIS_RESULTS.md','PROJECT_KNOWLEDGE.md','ces_prediction/check_records.py','ces_prediction/experiments/README.md','docs/paper/b115_status.md','docs/paper/b115_comparison.json','docs/paper/b115_comparison.md','docs/paper/b115_training_effect.png']
 subprocess.run(['git','diff','--check'],cwd=C.ROOT,check=True);subprocess.run(['git','add','--',*files],cwd=C.ROOT,check=True)
 subprocess.run(['git','commit','--only','-m','Publish independently audited B115 training-schedule comparison','--',*files],cwd=C.ROOT,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();subprocess.run(['git','push','origin','main'],cwd=C.ROOT,check=True)
 C.save(C.OUT/'workflow/closeout.json',{'all_pass':True,'commit':commit,'completed':time.time()})
if __name__=='__main__':main()
