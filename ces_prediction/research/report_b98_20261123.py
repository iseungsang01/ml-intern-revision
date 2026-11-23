"""Render B98's independently checked JSON; no fitting or result selection."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    x=json.loads((ROOT/'docs/paper/b98_physics.json').read_text(encoding='utf8'))
    assert x['independent_readout_pass'] and not x['outer_test_evaluated']
    assert all(p['solver'] is None or p['solver']['success'] for u in x['units'].values() for p in u['parameters'].values())
    lines=['# B.98 — 수식 기반 완화 모형의 내부 검증 pilot','',
           '계수 다섯 개의 강제 1차 완화 모형 M, 이력 완화 H, 정적 anchor 차분 L을 구현했다. 사전등록 후 TRAIN에서만 적합하고 네 대표 unit의 내부 VAL에서 판독했다. 원천은 [b98_physics.json](b98_physics.json), 검사 설명은 [b98_lead_checks.md](b98_lead_checks.md)다. 외부 TEST·L5·전체 네 설계의 확증 결과가 아니다.','',
           '**hold 대비 변화의 상당 부분이 이력 완화에서도 나타나며, 진단·빔 강제항의 추가 효과는 작거나 조건에 따라 불안정하다.** M−H는 ECEI/BES와 NBI를 함께 추가한 비교이며 fast가 NBI 너머로 주는 몫을 분리하지 않는다. 수식이나 계수가 물리적으로 식별됐다는 결론은 없다.','',
           f"본 실행은 {x['fit_seconds']:.2f}초였다(원자료 읽기·적합·초기 직접 재귀 검사·저장 포함; 후속 독립 raw 판독 시간은 별도). 네 unit마다 M/H/L을 적합했으며 모든 bounded τ solver가 정상 종료했다. 모든 표는 같은 unit 안에서 일곱 규칙 모두에 anchor가 있는 동일한 F 행이다. B.79의 기존 invalid-NBI 파일은 회귀 적합/채점에서 제외했고, 정규화는 기존 TRAIN 통계를 그대로 사용했다.",'',
           '## 읽는 법','',
           'MSE skill = 100×[1−MSE(M)/MSE(대조)] (%). 표의 각 cell은 점추정 [pointwise95% shot-cluster interval]; zero 제외 여부다. 음수는 M의 MSE가 대조보다 큰 방향이다. 다중 비교 보정이 없고 모델 채택 기준으로 사용하지 않는다. 구간은 fixed-fit 내부 VAL의 조건부 기술이며 재적합/선택 불확실성을 포함하지 않는다. 특히 적은 군집과 TRAIN/VAL 군집 중복 때문에 외부 일반화 증거로 해석하면 안 된다.','',
           'M은 과거 interval의 왼쪽 입력을 유지하는 강제항 적분, L은 현재/anchor 입력 차분을 쓴다. 두 모형의 차이는 순수 비선형성이나 물리 효과의 크기가 아니다. 타임스탬프 기반 입력 인과성과 실제 노출/도착 시각 보장은 별개다. 본 실행 시간은 추론 지연 측정이 아니며 이번 pilot은 실제10ms 배포 지연을 검증하지 않았다.','']
    for name,u in x['units'].items():
        lines += ['## '+name,'',
                  f"VAL shot 군집 {u['clusters']}, 방전 {u['shots']}, TRAIN과 공유된 군집 {u['train_val_cluster_overlap']}. M의 τ={u['tau']['M']:.6f}s, H의 τ={u['tau']['H']:.6f}s. 유효 기억시간의 적합점이며 수송/충돌시간의 측정값이 아니다.",'',
                  '| 규칙 | 공통 행 | M / hold skill (%) | M / H skill (%) | M / L skill (%) |',
                  '|---|---:|---:|---:|---:|']
        for rule,m in u['metrics'].items():
            cells=[]
            for key in ['skill_M_over_hold','skill_M_over_H','skill_M_over_L']:
                v=m[key];cells.append(f"{100*v['point']:+.3f} [{100*v['lo']:+.3f}, {100*v['hi']:+.3f}]; "+('제외' if v['excludes_zero'] else '포함'))
            lines.append('| '+rule+' | '+str(m['rows'])+' | '+' | '.join(cells)+' |')
        raw=u['independent_raw_checks']
        lines += ['',f"독립 raw 재구성 질의 {raw['queries']}, 직접 재귀 최대 절대 차이 {raw['direct_raw_max_abs_eV']:.3e}eV. 규칙별 MSE(eV²)·MAE(eV)·계수·조건수·solver 상태는 JSON에 모두 보존했다.",'']
    lines += ['## 판정과 다음 경계','',
              '구현·원자료 대응·직접 재귀·현재/미래 정보 배제 검사는 통과했다. 그러나 이 pilot만으로 물리적 강제항의 견고한 추가 가치를 확립하지 않았다. 시간순 조건의 S에서는 H/L 대비 음의 방향도 있으며 일부 구간은 zero를 제외한다. 임의로 모델을 늘리거나 재적합하지 않았다.','',
              '부정적 판단을 바꿀 측정은 독립 campaign 또는 보정·식별 가능한 입력/계측 응답을 갖춘 후속 평가다. NBI-only와 fast-added-to-NBI를 이 같은 모형에서 분리하려면 별도 사전등록이 필요하다. 기존 GP/tree 및 신경망과의 정확한 같은 행 비교, L5 모형, outer TEST 판독도 이 pilot에서 수행하지 않았다. 원고의 estimator나 두 표적/네 설계 보고 체계는 바꾸지 않는다.','',
              '기존 Lean B.97은 가시 source 선택의 이산 명세다. 새 ODE Python 코드 전체를 증명한 것이 아니며, 이번 구현은 별도 실제 trace·기호식·직접 재귀 검사로 확인했다. GPU 확인 학습의 복구·계속 실행은 별도 [B.99](b99_recovery.md)를 따른다.']
    (ROOT/'docs/paper/b98_physics.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':main()
