"""Publish source-bound B114 interpretation after successful independent audit."""
import json,subprocess,time
import common114 as C

def main():
 report=C.read(C.ROOT/'docs/paper/b114_comparison.json');assert report['independent_audit']['all_pass']
 assert C.read(C.OUT/'audit.json')==report['independent_audit']
 summary='B114 COMPLETE / independently audited. All144full fits and selected-family32scenario cells completed; per-outer-unit VAL selection and independent numerical replay verified. Registered statements: '+json.dumps(report['statements'])+'. Results: docs/paper/b114_comparison.json/md. No extra fits queued; existing estimator unchanged.'
 para='''동일한 기존 스칼라 입력에서 CatBoost 설정 탐색과 잔차 신경망·구간별 선형 인코딩 신경망을 GPU로 적합했다. 각 외부 검증 설계의 내부 VAL만으로 계열별 설정과 직접·잔차형을 선택했으며 설계 간 VAL을 합치지 않았다. 전체 학습의 정상 종료 뒤 선택 모델의 TEST를 평가했고, 입력·행·학습 가중치·신경망 정규화·인코딩 분위수·학습 이력과 선택 재계산, 독립 트리·신경망 순전파 및 paired 구간 검산이 통과했다.\n\n등록 판정은 원천 JSON의 statements를 따른다: '''+json.dumps(report['statements'])+'''. 상세 점추정·구간·영제외와 선택 설정은 원천 표를 함께 읽는다. 기준 모델별 결과를 섞어 해석하지 않는다. 기존 TEST 재사용과 설정 선택에 조건부인 탐색 비교이며 탐색·재적합 불확실성을 구간에 포함하지 않았다. 현행 estimator를 자동 교체하지 않는다. 사용하지 않은 관측에서 예산·입력·표적을 맞춘 비교가 결론을 뒤집을 수 있다. 추가 적합은 없다.\n'''
 p=C.ROOT/'PROJECT_KNOWLEDGE.md';s=p.read_text(encoding='utf8');marker='## 5. 코드와 산출물';assert marker in s and '### 4.82 ' not in s
 p.write_text(s.replace(marker,'### 4.82 VAL 선택을 거친 밤샘 스칼라 모델 비교 (B.114; `docs/paper/b114_comparison.md`)\n\n'+para+'\n'+marker,1),encoding='utf8',newline='\n')
 p=C.ROOT/'THESIS_RESULTS.md';s=p.read_text(encoding='utf8');p.write_text(s+'\n\n## B.114 overnight local GPU search\n\n'+summary+'\n\n'+para,encoding='utf8',newline='\n')
 p=C.ROOT/'HANDOVER.md';s=p.read_text(encoding='utf8');i=s.find('\n')+1;p.write_text(s[:i]+'\n**'+summary+'**\n'+s[i:],encoding='utf8',newline='\n')
 for rel in ('README.md','ces_prediction/experiments/README.md'):
  p=C.ROOT/rel;s=p.read_text(encoding='utf8');p.write_text(s.replace('B114 PREPARED','B114 COMPLETE / independently audited'),encoding='utf8',newline='\n')
 (C.ROOT/'docs/paper/b114_status.md').write_text('# B114 complete\n\n'+summary+'\n\n'+para,encoding='utf8')
 p=C.ROOT/'ces_prediction/check_records.py';s=p.read_text(encoding='utf8');assert 'docs/paper/b114_comparison.md' not in s
 p.write_text(s.replace('SOURCES = (','SOURCES = ('+chr(10)+'    '+chr(34)+'docs/paper/b114_comparison.md'+chr(34)+',',1),encoding='utf8',newline='\n')
 subprocess.run([sys_python(),str(C.ROOT/'ces_prediction/check_records.py')],cwd=C.ROOT,check=True)
 subprocess.run([sys_python(),'-m','pytest','-q'],cwd=C.ROOT,check=True)
 assert subprocess.check_output(['git','branch','--show-current'],cwd=C.ROOT,text=True).strip()=='main','Branch changed; preserve results without committing'
 files=['ces_prediction/check_records.py','PROJECT_KNOWLEDGE.md','THESIS_RESULTS.md','HANDOVER.md','README.md','ces_prediction/experiments/README.md','docs/paper/b114_status.md','docs/paper/b114_comparison.json','docs/paper/b114_comparison.md']
 subprocess.run(['git','diff','--check'],cwd=C.ROOT,check=True)
 subprocess.run(['git','add','--',*files],cwd=C.ROOT,check=True)
 subprocess.run(['git','commit','--only','-m','Publish independently audited B114 overnight search','--',*files],cwd=C.ROOT,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip()
 subprocess.run(['git','push','origin','main'],cwd=C.ROOT,check=True)
 C.save(C.OUT/'workflow/closeout.json',{'all_pass':True,'commit':commit,'completed':time.time()})
def sys_python():
 import sys
 return sys.executable
if __name__=='__main__':main()
