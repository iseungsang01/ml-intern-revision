# Colab T4 실행 방법

공식 MCP 서버 설치/초기화 및 브라우저 연결은 성공했다. 첫 진단은 CPU 런타임
(Python3.13.18,PyTorch2.14.1+cpu)이므로 T4 선택을 저자에게 요청했다.
`research/colab_t4_operator.py check`로 실제 T4를 확인한 뒤 `start`가 같은
9.26MiB archive를 분할 전달하고 해시 검증 후 원격 측정을 시작한다.
`poll`은 상태와 집계 결과만 회수한다. source3805f7d의 CPU export 검사와
pytest89 통과, archive에는 A-D NPZ/weights와 portable/probe/run/manifest만 있다.
API 키/.env/TEST 전송 없음. 실제 T4/측정 결과는 별도 확인 전까지 미확인이다.

공식 MCP 연결도 준비한다: `ces_prediction/research/colab_mcp_bridge.py`는
분리된 `data/.colab_mcp_env`의 Google 공식 colab-mcp 고정 commit과 공식 MCP SDK로
브라우저의 Colab 노트북에 연결한다. 직접 Google 계정 인증을 자동화하지 않는다.
연결 전에는 런타임 실행이나 데이터 전송이 없다. 연결 상태/도구/요청 결과는
ignored `data/.colab_connection/`에 보존한다. 로컬 학습은 중단 상태를 유지한다.
아래 파일 업로드 노트북은 직접 MCP 연결이 불가능할 때의 대체 경로다.

`b108_colab/export108.py`는 기존 B106 실자료 TRAIN/VAL과 고정 checkpoint로
Colab에 올릴 비공개 묶음을 만든다. `.env`, API 키, TEST, shot/file ID는 넣지 않는다.
등록/소스 push 후 CPU에서만 export하며 현재 B107 GPU 학습은 유지한다.

생성 위치: `data/colab_b108/COLAB_T4.ipynb`, `ces_t4_bundle.zip`.

1. Colab에서 **파일 → 노트북 업로드**로 `COLAB_T4.ipynb`를 연다.
2. **런타임 → 런타임 유형 변경 → T4 GPU**, 다음 **모두 실행**.
3. 파일 선택 창에 `ces_t4_bundle.zip`을 올린다.
4. 실행 종료 후 내려받는 `b108_results.zip`을 로컬에서 분석한다.

측정은 실제 dropout0.1·float32·동일 모델/목적함수로 cuDNN 켬/끔과 1/4병렬을
각3회 비교한다. 짧은 실제 배치 학습·검증·최장 배치 메모리·정상 종료를 확인하며
어떤 probe checkpoint도 본 연구 모델로 채택하지 않는다. 정확한 로컬/T4 속도 배수는
동일 측정을 로컬 GPU가 비었을 때 대조하기 전에는 주장하지 않는다.

Colab T4 실행 및 속도 결과는 아직 미확인이다. 사용자가 인증한 Colab T4 런타임에서
실행하며 추가 API 키는 필요 없다. private runtime에만 올리고 공개 공유하지 않는다.
런타임 재실행 시 기존 `/content/ces_b108` 폴더가 있으면 세션을 재시작해 별도 시도로
기록하며, 기존 결과를 덮어쓰지 않는다. 정상 종료하지 못한 경로도 결과 로그에 남는다.
