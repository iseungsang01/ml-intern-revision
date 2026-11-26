"""Render a different real S discharge with support closest to150; no scoring."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ces_prediction/experiments/b112_residual'))
from common112 import npz

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
 paper=ROOT/'docs/paper';old=json.loads((paper/'b115_uncertainty.json').read_text());selection=json.loads((ROOT/'data/.b114/selection.json').read_text())
 family=selection['overall_family_by_unit']['U0'];folder='neural' if family=='neural' else 'cat';name='NN_selected_ensemble' if family=='neural' else 'Cat_selected_ensemble'
 arr=npz(ROOT/'data/.b114'/folder/'estimates/U0_S.npz')
 ids,counts=np.unique(arr['discharge'],return_counts=True);previous=old['trace']['anonymous_discharge']
 eligible=[(abs(int(n)-150),int(fid),int(n)) for fid,n in zip(ids,counts) if fid!=previous and n>=30]
 _,shot,n=min(eligible);mask=arr['discharge']==shot
 mapping=json.loads((ROOT/'data/colab_b111/private_map.json').read_text());inverse=np.asarray(mapping['row_inverse'],dtype=np.int64)
 source=inverse[arr['idx'][mask].astype(np.int64)];order=np.argsort(source,kind='stable');x=source[order]-source.min()
 y=arr['truth'][mask][order];gp=arr['GP_fast_boost'][mask][order];pred=arr[name][mask][order]
 assert np.isfinite(y).all() and len(y)==n and shot!=previous
 fig,ax=plt.subplots(figsize=(11,4.8))
 ax.plot(x,y,'o',color='#242424',ms=4,label='CES 보고값')
 ax.plot(x,gp,color='#1765a0',lw=1.8,label='GP+fast')
 ax.plot(x,pred,color='#d87816',lw=1.8,label='기존 VAL 선택 모델')
 ax.set_title(f'다른 shot: 익명 ID {shot} · 관측 {n}개 · S 조건')
 ax.set_xlabel('시간순 원본 행 위치 (ms 축 아님)');ax.set_ylabel('보고값 / 추정값 (eV)');ax.legend(ncol=3);ax.grid(alpha=.18)
 fig.tight_layout();out=paper/'b115_other_shot.png';fig.savefig(out,dpi=160);plt.close(fig)
 note=f'# 다른 shot 예시\n\n기존 익명 shot {previous}를 제외하고, S 조건의 관측 수가150에 가장 가까운 방전을 선택했다. 동률이면 익명 ID가 작은 방전을 선택했다. TEST 오차를 보고 고르지 않았고 새 학습·채점·구간 계산은 없다.\n\n선택: 익명 ID {shot}, 관측 {n}개. 원래 보고값·GP+fast·기존 VAL 선택 모델을 원본 시간순 행 위치로 표시했다. 단일 방전의 육안 비교를 전체 성능 판정으로 쓰지 않는다.\n\n원천: data/.b114/{folder}/estimates/U0_S.npz, data/colab_b111/private_map.json. 재생성: py ces_prediction/research/render_other_shot.py\n'
 (paper/'b115_other_shot.md').write_text(note,encoding='utf8');print('Rendered other discharge:',shot,'rows:',n)
if __name__=='__main__':main()
