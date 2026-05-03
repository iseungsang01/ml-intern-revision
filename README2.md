# ML Intern Revision - Current Code Structure

이 문서는 현재 저장소가 실제로 어떻게 동작하는지 기준으로 정리한 구조 문서입니다. 기존 `README.md`의 KSTAR Multi-Agent R&D 설명은 일부 별도 실험 스크립트와 기본 `ml-intern` 런타임을 섞어서 설명하므로, 여기서는 실제 엔트리포인트를 분리해서 설명합니다.

## 한 줄 요약

현재 기본 제품 경로는 **Hugging Face 기반 ML 코딩 에이전트**입니다.

- CLI: `ml-intern` 명령으로 `agent.main:cli`가 실행됩니다.
- Web/API: FastAPI 백엔드가 세션을 만들고, SSE로 에이전트 이벤트를 프론트엔드에 전달합니다.
- KSTAR CES AutoML: `ces_prediction/automl_agent_loop.py`에 별도 실험 루프가 있지만, 현재 `ml-intern` CLI나 웹 서버 기본 경로에 연결되어 있지는 않습니다.

## 최상위 디렉터리

```text
.
├── agent/                 # 실제 ml-intern CLI와 핵심 agent loop
├── backend/               # FastAPI API 서버, 세션 관리, SSE 이벤트 스트리밍
├── frontend/              # React + Vite 웹 UI
├── ces_prediction/        # KSTAR CES 예측용 별도 PyTorch 실험 코드
├── configs/               # CLI/Web 에이전트 설정 JSON
├── scripts/               # KPI/SFT 데이터 생성 보조 스크립트
├── tests/                 # pytest 테스트
├── Dockerfile             # HF Spaces/production용 frontend build + backend serve
├── pyproject.toml         # Python package, dependencies, ml-intern entrypoint
└── README.md              # 기존 문서
```

## 1. CLI 실행 경로

`pyproject.toml`에서 CLI 엔트리포인트가 정의되어 있습니다.

```toml
[project.scripts]
ml-intern = "agent.main:cli"
```

즉, 설치 후 `ml-intern`을 실행하면 `agent/main.py`의 `cli()` 함수가 실행됩니다.

### Interactive mode

```bash
ml-intern
```

실행 흐름:

```text
ml-intern
  -> agent.main:cli()
  -> main()
  -> HF token 확인/프롬프트
  -> configs/cli_agent_config.json 로드
  -> ToolRouter(local_mode=True) 생성
  -> submission_loop(...) 시작
  -> 사용자 입력을 USER_INPUT submission으로 전달
  -> agent.core.agent_loop.process_submission()
```

CLI local mode에서는 `bash`, `read_file`, `write_file`, `edit_file` 같은 로컬 파일 도구가 실제 사용자 로컬 파일시스템에 대해 동작합니다.

### Headless mode

```bash
ml-intern "fine-tune llama on my dataset"
```

실행 흐름:

```text
ml-intern "prompt"
  -> agent.main:cli()
  -> headless_main(prompt)
  -> configs/cli_agent_config.json 로드
  -> config.yolo_mode = True
  -> submission_loop(...) 시작
  -> prompt 1회 제출
  -> turn_complete 또는 error 후 종료
```

Headless mode는 자동 승인 모드(`yolo_mode=True`)로 실행됩니다.

### CLI 옵션

현재 구현된 주요 옵션:

```bash
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

`--max-iterations -1`은 내부적으로 매우 큰 값으로 바뀌어 사실상 unlimited처럼 처리됩니다.

## 2. 핵심 Agent Runtime

핵심 루프는 `agent/core/agent_loop.py`에 있습니다.

주요 구성:

```text
agent/core/agent_loop.py
  - submission_loop()
  - process_submission()
  - LLM call via litellm.acompletion()
  - tool_calls 처리
  - approval_required 이벤트 처리
  - context compaction
  - turn_complete/error/interrupted 이벤트 발행

agent/core/session.py
  - Session 상태 관리
  - 이벤트 큐
  - context manager
  - session log 저장 및 업로드

agent/context_manager/manager.py
  - system_prompt_v3.yaml 로드
  - message history 관리
  - context compaction/summarization

agent/core/tools.py
  - ToolRouter
  - built-in tools 등록
  - MCP tools 등록
```

현재 기본 system prompt는 `agent/prompts/system_prompt_v3.yaml`입니다. 이 프롬프트는 ML 작업에서 research-first workflow를 강하게 지시합니다.

## 3. Tool 구조

`ToolRouter`는 built-in tool과 MCP tool을 한곳에서 관리합니다.

### CLI local mode

CLI에서는 `ToolRouter(..., local_mode=True)`로 생성됩니다.

이 경우:

```text
agent/tools/local_tools.py
```

의 로컬 도구들이 우선 등록됩니다. 로컬 도구는 실제 현재 작업 디렉터리의 파일을 읽고 쓰며, sandbox가 아닙니다.

### Web/API mode

웹 세션에서는 `ToolRouter(self.config.mcpServers, hf_token=hf_token)`로 생성되며 `local_mode=False`입니다.

이 경우:

```text
agent/tools/sandbox_tool.py
agent/tools/sandbox_client.py
```

기반 sandbox 도구가 사용됩니다.

### 주요 built-in tools

현재 코드상 주요 도구 범주:

- `research`: 별도 read-only research sub-agent 실행
- Hugging Face docs/search tools
- HF papers tool
- HF dataset inspection
- plan tool
- HF jobs
- HF repo files/git tools
- GitHub example/search/read tools
- local 또는 sandbox file/bash/edit tools

중요한 점: README의 `Evaluation Agent / Briefing Agent / Researcher Agent`는 이 일반 `research` sub-agent와 다릅니다. 일반 `research` 도구는 문서/코드/논문 리서치용 서브 에이전트이고, KSTAR AutoML의 세 agent 클래스는 `ces_prediction/automl_agent_loop.py` 안의 별도 코드입니다.

## 4. Web/API 실행 경로

백엔드는 FastAPI 앱입니다.

```text
backend/main.py
  -> FastAPI app 생성
  -> /api router 등록
  -> /auth router 등록
  -> static/ 존재 시 frontend build 서빙
  -> lifespan에서 KPI scheduler 시작/종료
```

세션 관리는 `backend/session_manager.py`가 담당합니다.

```text
POST /api/session
  -> SessionManager.create_session()
  -> configs/frontend_agent_config.json 로드값 기반 config clone
  -> ToolRouter 생성
  -> Session 생성
  -> _run_session() task 시작
  -> ready event 발행

POST /api/chat/{session_id}
  -> user input 또는 approval 제출
  -> SSE StreamingResponse 반환
  -> agent event를 data: JSON 형태로 stream
```

주요 API route는 `backend/routes/agent.py`에 있습니다.

대표 엔드포인트:

- `GET /api/health`
- `GET /api/config/model`
- `POST /api/session`
- `POST /api/chat/{session_id}`
- `GET /api/events/{session_id}`
- `POST /api/interrupt/{session_id}`
- `POST /api/approve`
- `POST /api/compact/{session_id}`
- `POST /api/shutdown/{session_id}`

인증은 `backend/routes/auth.py`, `backend/dependencies.py` 쪽에서 처리합니다. 개발 모드에서는 OAuth 설정이 없으면 인증이 우회되는 구조입니다.

## 5. Frontend 구조

프론트엔드는 React + Vite입니다.

```text
frontend/
├── package.json
├── vite.config.ts
└── src/
    ├── App.tsx
    ├── components/
    │   ├── Chat/
    │   ├── CodePanel/
    │   ├── Layout/
    │   ├── SessionSidebar/
    │   └── WelcomeScreen/
    ├── hooks/
    ├── lib/
    ├── store/
    ├── types/
    └── utils/
```

프론트엔드는 백엔드 API/SSE를 통해 세션을 만들고 메시지 이벤트를 렌더링합니다.

개발 실행:

```bash
cd frontend
npm install
npm run dev
```

백엔드 개발 실행 예시:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860
```

## 6. Docker / HF Spaces 실행 경로

`Dockerfile`은 2-stage build입니다.

```text
Stage 1: node:20-alpine
  -> frontend npm install
  -> npm run build

Stage 2: python:3.12-slim
  -> Python dependencies 설치
  -> agent/, backend/, configs/ 복사
  -> frontend dist를 /app/static 으로 복사
  -> WORKDIR /app/backend
  -> bash start.sh
```

`backend/start.sh`는 다음을 실행합니다.

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

`backend/main.py`는 `/app/static`이 있으면 프론트엔드 정적 파일을 `/`에 mount합니다.

## 7. Config 파일

### CLI config

```text
configs/cli_agent_config.json
```

현재 기본:

```json
{
  "model_name": "anthropic/claude-opus-4-6",
  "save_sessions": true,
  "session_dataset_repo": "smolagents/ml-intern-sessions",
  "yolo_mode": false,
  "confirm_cpu_jobs": true,
  "auto_file_upload": true,
  "mcpServers": {
    "hf-mcp-server": {
      "transport": "http",
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

### Frontend/backend config

```text
configs/frontend_agent_config.json
```

현재 기본:

```json
{
  "model_name": "bedrock/us.anthropic.claude-opus-4-6-v1",
  "save_sessions": true,
  "session_dataset_repo": "smolagents/ml-intern-sessions",
  "yolo_mode": false,
  "confirm_cpu_jobs": true,
  "auto_file_upload": true,
  "mcpServers": {
    "hf-mcp-server": {
      "transport": "http",
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

`agent/config.py`의 기본 `max_iterations`는 300입니다. README의 "default: 50" 설명과 다를 수 있으므로 현재 기준으로는 300이 맞습니다.

## 8. KSTAR CES Prediction 코드

`ces_prediction/`은 현재 기본 `ml-intern` 런타임과 분리된 별도 실험 코드입니다.

```text
ces_prediction/
├── dataset.py             # CSV 기반 KSTAR_CES_Dataset
├── model.py               # MultimodalCESPredictor PyTorch 모델
├── train.py               # 학습 루프, metrics.json/weights 저장
├── automl_agent_loop.py   # Evaluation/Briefing/Researcher Agent 실험 루프
└── slack_notifier.py      # Slack 요약 알림
```

### 데이터셋

`dataset.py`는 `../data/*.csv`를 읽습니다.

기대 컬럼:

- target: `CES_TI`, `CES_VT`
- BES input: `BES_` prefix 컬럼
- ECEI input: `ECEI_` prefix 컬럼
- MC input: `MC1` prefix 컬럼
- time input: `time`

각 sample은 sliding window 형태입니다.

```python
{
    "bes": Tensor(window_size, n_bes),
    "ecei": Tensor(window_size, n_ecei),
    "mc": Tensor(window_size, n_mc),
    "dt": Tensor(window_size, 1),
    "target": Tensor(2)
}
```

### 모델

`model.py`의 `MultimodalCESPredictor`는 다음 branch를 late fusion합니다.

- BES Conv1d branch
- ECEI Conv1d branch
- MC Conv1d branch
- time/dt Conv1d branch
- fused FC head -> `[CES_TI, CES_VT]`

주의: `model.py`의 `__main__` 테스트 코드는 현재 `forward()`에 필요한 `dt` 인자를 넘기지 않아 그대로 실행하면 실패합니다. 학습 경로(`train.py`)에서는 `dt`를 넘깁니다.

### 학습

`train.py`는 다음을 수행합니다.

```text
1. KSTAR_CES_Dataset(data_dir="../data", window_size=10)
2. index % 5 == 0 을 validation으로 사용
3. MultimodalCESPredictor 학습
4. MSE + negative Ti penalty 사용
5. weights/multimodal_ces.pth 저장
6. metrics.json 저장
```

실행 예시:

```bash
cd ces_prediction
python train.py
```

### AutoML loop

`automl_agent_loop.py`에는 세 클래스가 있습니다.

- `EvaluationAgent`: architecture dry-run 후 `python train.py` 실행, `metrics.json` 읽기
- `BriefingAgent`: validation loss history 관리, plateau 감지, `HANDOFF.md` 작성
- `ResearcherAgent`: LiteLLM으로 `model.py`를 새 코드로 재작성

실행 예시:

```bash
cd ces_prediction
python automl_agent_loop.py
```

하지만 이 루프는 현재 다음과 같이 기본 제품 경로와 분리되어 있습니다.

- `pyproject.toml` entrypoint에 연결되어 있지 않음
- `agent/main.py`에서 import 또는 호출하지 않음
- `backend/session_manager.py` 또는 API route에서 호출하지 않음
- Dockerfile에도 `ces_prediction/`이 복사되지 않음

따라서 현재 `ml-intern` 명령을 실행한다고 해서 이 KSTAR AutoML loop가 자동으로 돌지는 않습니다.

## 9. 테스트

개발 의존성 설치:

```bash
python -m pip install -e ".[dev]"
```

테스트 실행:

```bash
pytest
```

테스트는 다음 범주를 포함합니다.

- agent loop / tool-call 안정성
- CLI rendering
- HF access
- KPI scheduler
- SFT build/tagging
- user quotas
- KSTAR architecture 관련 테스트

## 10. 현재 코드상 주의할 점

현재 구조를 이해할 때 중요한 불일치/주의점입니다.

1. 기존 README의 KSTAR Multi-Agent R&D 설명은 기본 `ml-intern` 실행 경로와 직접 연결되어 있지 않습니다.
2. `ces_prediction/automl_agent_loop.py`는 별도 스크립트입니다.
3. Docker production image는 `agent/`, `backend/`, `configs/`, built frontend만 복사합니다. `ces_prediction/`은 포함하지 않습니다.
4. `pyproject.toml`의 package include는 `agent*`만 포함합니다. 따라서 `ces_prediction`은 Python package로 설치되지 않습니다.
5. `model.py`의 standalone forward-pass 테스트는 `dt` 누락으로 수정이 필요합니다.
6. 여러 소스 파일에 한글 주석 인코딩이 깨진 흔적이 있습니다. 실행에는 직접 영향이 없을 수 있지만 유지보수 문서로는 정리 필요합니다.

## 11. 실제 현재 동작 정리

현재 사용자가 `ml-intern`을 실행하면:

```text
KSTAR AutoML loop가 실행되는 것이 아니라,
일반 ML engineering agent가 실행된다.

이 agent는:
  - system_prompt_v3 기반으로 동작하고
  - LiteLLM으로 모델을 호출하고
  - tool calls를 반복 실행하고
  - local 또는 sandbox 도구로 파일/명령을 처리하고
  - research sub-agent를 필요 시 도구로 호출한다.
```

현재 웹 앱을 실행하면:

```text
FastAPI가 session을 만들고,
각 session마다 agent runtime을 task로 띄우고,
프론트엔드는 SSE 이벤트를 받아 chat UI로 보여준다.
```

현재 KSTAR CES 실험을 하려면:

```text
cd ces_prediction
python train.py

또는

cd ces_prediction
python automl_agent_loop.py
```

처럼 별도로 실행해야 합니다.

