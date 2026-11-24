"""Independent CPU artifact audit, alternate row/group/bootstrap arithmetic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import zipfile
from common110 import *

def check_interval(z,values):
    assert np.isfinite(values).all() and z['available']
    bounds=np.percentile(values,[2.5,97.5]);assert np.allclose(bounds,z['ci95'],rtol=1e-9,atol=1e-7)
    assert z['excludes_zero']==bool(bounds[0]>0 or bounds[1]<0)

def main():
    guard();m=read(OUT/'bundle/manifest.json');state=read(ROOT/'data/.colab_connection/b110_remote.json')
    archive=OUT/'b110_results.zip';assert state['downloaded'] and sha(archive)==state['results_sha256']
    dest=OUT/'retrieved';assert not dest.exists(),'Preserve prior audit attempt';dest.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert Path(n).name==n and not n.startswith('.')
        z.extractall(dest)
    for n,h in read(dest/'result_hashes.json').items():assert sha(dest/n)==h
    report=read(dest/'results.json');assert report['sources']==m['sources'] and not report['TEST_scored'] and report['new_fits']==0
    checks=0
    for tag,task in m['tasks'].items():
        src=npz(OUT/'bundle'/(tag+'.npz'));row=npz(dest/(tag+'_rows.npz'));u=task['unit'];unit=report['units'][u]
        assert unit['prefix_pass'] and unit['input_sha256']==sha(OUT/'bundle'/(tag+'.npz')) and unit['weights_sha256']==task['weights_sha256']
        for k in row:
            if k!='LSTM':assert np.array_equal(row[k],src[k])
        assert row['LSTM'].shape==row['truth'].shape and np.isfinite(row['LSTM']).all()
        error=np.vstack([row[n]-row['truth'] for n in MODELS]).T
        age=row['age'];change=row['history_change'];q=float(row['change_q75'])
        masks={'all':np.ones(len(age),bool),'age_10ms':age<=10.000001,'age_over_10ms':age>10.000001,'history_change_low':change<=q,'history_change_high':change>q}
        assert np.all(masks['age_10ms']^masks['age_over_10ms']) and np.all(masks['history_change_low']^masks['history_change_high'])
        for group,keep in masks.items():
            cell=unit['groups'][group];e=error[keep];assert cell['rows']==len(e)
            if not len(e):assert not cell['available'];continue
            squares=np.sum(np.square(e),axis=0)/len(e);means=np.sum(e,axis=0)/len(e)
            for j,n in enumerate(MODELS):
                v=cell['models'][n];assert np.isclose(v['MSE'],squares[j],rtol=1e-12) and np.isclose(v['bias'],means[j],atol=1e-10)
                assert np.isclose(v['centered_MSE'],squares[j]-means[j]**2,rtol=1e-12,atol=1e-7);checks+=1
            for a,b in PAIRS:
                aa,bb=MODELS.index(a),MODELS.index(b);assert squares[bb]>0
                assert np.isclose(cell['contrasts'][a+'/'+b]['point'],1-squares[aa]/squares[bb],atol=1e-12)
            for form in ('cluster','discharge'):
                labels=np.unique(row[form][keep]);ng=len(labels);assert cell[form+'_count']==ng
                if len(e)<2 or ng<2:
                    for v in [*cell['models'].values(),*cell['contrasts'].values()]:assert not v[form]['available']
                    continue
                inv=np.searchsorted(labels,row[form][keep]);counts=np.zeros(ng,np.int64);sums=np.zeros((ng,len(MODELS)));sq=np.zeros_like(sums)
                np.add.at(counts,inv,1);np.add.at(sums,inv,e);np.add.at(sq,inv,e*e)
                rng=rng_for(u,group,form);av=[];ms=[]
                for offset in range(0,DRAW,500):
                    weights=rng.multinomial(ng,np.full(ng,1/ng),min(500,DRAW-offset));den=np.sum(weights*counts,axis=1)
                    av.append(np.einsum('ij,jk->ik',weights,sums)/den[:,None]);ms.append(np.einsum('ij,jk->ik',weights,sq)/den[:,None])
                av=np.concatenate(av);ms=np.concatenate(ms)
                for j,n in enumerate(MODELS):
                    v=cell['models'][n][form];check_interval(v['MSE'],ms[:,j]);check_interval(v['bias'],av[:,j]);check_interval(v['centered_MSE'],ms[:,j]-av[:,j]**2)
                for a,b in PAIRS:
                    aa,bb=MODELS.index(a),MODELS.index(b);check_interval(cell['contrasts'][a+'/'+b][form],1-ms[:,aa]/ms[:,bb])
    report['independent_audit']={'all_pass':True,'model_group_cells':checks,'alignment_points_decomposition_and_intervals_recomputed':True,'results_sha256':sha(archive),'local_fits':0,'local_GPU_work':0}
    public=ROOT/'docs/paper';write(public/'b110_residuals.json',report)
    lines=['# B110: D10/F VAL residual diagnostic','','탐색적 내부 VAL 진단이다. 체크포인트 선택에 사용한 VAL을 재사용하므로 최종 merit/TEST 검증이 아니다. 네 설계를 모두 표시한다. LSTM은 후단 보정 전이다. fair_specialized/fair_mixture는 B105의 TRAIN-matched GP+fast 부스팅이며 fast 단독 기여의 대조는 아니다.','','MSE 단위 eV², 편향 단위 eV. 표의 괄호는 군집 재표집 점별 95% 구간이며 방전 재표집·중심화 MSE 구간은 JSON에 모두 있다. 모델/조건 간 다중성 보정이나 설계 간 합산은 하지 않았다. 새 적합·TEST 평가·Claude 호출 없음.','']
    def fmt(value,ci):
        return f'{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'
    for u,unit in report['units'].items():
        lines.extend(['## '+u,'','| Group | Model | Rows | MSE [95%] | Bias [95%] | Skill vs GP [95%] | Skill excludes 0 |','|---|---|---:|---|---|---|---|'])
        for group,c in unit['groups'].items():
            if not c['rows']:lines.append(f'| {group} | unavailable | 0 | empty group | — | — | — |');continue
            for n,v in c['models'].items():
                form=v.get('cluster',{})
                if 'MSE' not in form:
                    lines.append(f"| {group} | {n} | {c['rows']} | {v['MSE']:.3f} (CI unavailable) | {v['bias']:.3f} (CI unavailable) | — | — |");continue
                pair=c['contrasts'].get(n+'/GP');skill='reference' if pair is None else fmt(pair['point'],pair['cluster']['ci95']);ex='—' if pair is None else str(pair['cluster']['excludes_zero'])
                lines.append(f"| {group} | {n} | {c['rows']} | {fmt(v['MSE'],form['MSE']['ci95'])} | {fmt(v['bias'],form['bias']['ci95'])} | {skill} | {ex} |")
        lines.extend(['','| Group | Contrast | Skill [95%] | Excludes 0 |','|---|---|---|---|'])
        for group,c in unit['groups'].items():
            for key in ('TCN35/fair_specialized','LSTM/fair_specialized','LSTM/TCN35'):
                v=c['contrasts'].get(key)
                if v and v.get('cluster',{}).get('available'):lines.append(f"| {group} | {key} | {fmt(v['point'],v['cluster']['ci95'])} | {v['cluster']['excludes_zero']} |")
        lines.append('')
    lines.extend(['음의 비교를 바꿀 측정은 같은 입력·적합 지원에서 별도 고정한 평가의 MSE 감소와 0을 제외하는 차이 구간이다. 이 VAL 진단으로 구간별 hard rule을 적합하거나 물리 수송·진단 정보 부재·새 방법의 독창성을 주장하지 않는다.',''])
    (public/'b110_residuals.md').write_text('\n'.join(lines),encoding='utf8')
    print('B110 independent audit passed; model/group cells='+str(checks),flush=True)

if __name__=='__main__':main()
