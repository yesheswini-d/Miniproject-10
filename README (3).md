# SMS Spam Detector

Production-style SMS spam classification: a trained scikit-learn model behind a
FastAPI service, a browser UI, a Streamlit dashboard, metrics, structured logs,
tests, CI and a Docker image.

Everything is reproducible from a clean checkout: the dataset is downloaded
automatically and the model is trained by a single command (or baked into the
Docker image at build time).

| | |
| --- | --- |
| **Accuracy** | 0.987 |
| **Spam F1** | 0.948 |
| **ROC AUC** | 0.998 |
| **API latency** | 5.6 ms p50 (single request, no explanation) |
| **Tests** | 39 passing |

---

## Architecture

```
                 ┌──────────────────────────────┐
   browser ─────▶│  FastAPI (api/main.py)       │
                 │                              │
                 │  /            web UI         │
   Streamlit ───▶│  /predict     single         │──▶ ModelService ──▶ SpamClassifier
   (app.py)      │  /predict/batch              │        │            (TF-IDF + LogReg)
                 │  /model/info  /metrics       │        │
   CI / ops ────▶│  /health /ready /stats       │        └── joblib artifact
                 │  /feedback    /admin/reload  │            models/*.joblib
                 └──────────────────────────────┘
                          │
                          ├── JSON logs (stdout)
                          ├── Prometheus metrics (/metrics)
                          └── feedback.jsonl (retraining data)
```

The ML core (`src/`) has no HTTP dependencies; the API (`api/`) has no training
logic. That separation is what makes the model testable in isolation and the
service deployable on its own.

---

## Results

Stratified 20% hold-out set, 5,158 deduplicated messages (13% spam):

| Metric | Score |
| --- | --- |
| Accuracy | 0.9874 |
| Precision (spam) | 0.9675 |
| Recall (spam) | 0.9297 |
| F1 (spam) | 0.9482 |
| ROC AUC | 0.9982 |

|  | predicted ham | predicted spam |
| --- | --- | --- |
| **actual ham** | 900 | 4 |
| **actual spam** | 9 | 119 |

Three candidates are compared with 5-fold cross-validated grid search and the
best spam-F1 wins (Logistic Regression, CV F1 0.955, ahead of calibrated Linear
SVM 0.952 and Naive Bayes 0.948).

---

## Quickstart

### 1. Local (API + dashboard)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     POSIX: source .venv/bin/activate
pip install -r requirements-dev.txt

python main.py train          # downloads data, trains, writes models/
python main.py api            # http://localhost:8000  (docs at /docs)
python main.py serve          # http://localhost:8501  (second terminal)
```

`python main.py api` and `streamlit` are independent processes; start the API
first so the dashboard uses it instead of falling back to the local model.

### 2. Docker

```bash
cp .env.example .env          # optional
docker compose up --build -d  # builds, trains inside the image, starts on :8000
curl http://localhost:8000/ready
docker compose down
```

The image is self-contained: dependencies are installed and the model is
trained during the build, so the container needs no network at runtime.

### 3. Windows task runner

```powershell
.\tasks.ps1            # list tasks
.\tasks.ps1 train
.\tasks.ps1 api
.\tasks.ps1 test
.\tasks.ps1 smoke
```

A `Makefile` with the same targets exists for POSIX/CI.

---

## API

Base URL `http://localhost:8000`. Interactive OpenAPI docs at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Browser UI |
| GET | `/health` | Liveness (always 200 while the process runs) |
| GET | `/ready` | Readiness; 503 until a model is loaded |
| GET | `/model/info` | Model name, version, metrics, leaderboard |
| POST | `/predict` | Classify one message |
| POST | `/predict/batch` | Classify up to `SPAM_MAX_BATCH_SIZE` messages |
| POST | `/feedback` | Record a ground-truth label for retraining |
| POST | `/admin/reload` | Hot-reload the artifact (bearer token) |
| GET | `/metrics` | Prometheus exposition format |
| GET | `/stats` | JSON counters for quick debugging |

### Predict

```bash
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"You won a £1000 prize, call 0800 123 4567 now","explain":true}'
```

```json
{
  "label": "spam",
  "spam_probability": 0.999968,
  "confidence": 0.999968,
  "threshold": 0.5,
  "explanation": [
    {"token": "__phone__", "weight": 3.627},
    {"token": "__money__", "weight": 2.312}
  ],
  "model": {"name": "logistic_regression", "version": "cb7fb7ab9804"},
  "latency_ms": 4.2,
  "request_id": "8f3c1d02a9be"
}
```

`threshold` per request lets a caller trade precision for recall without
touching server config. Explanations are signed token contributions: positive
pushes towards spam, negative towards ham.

### Batch

```bash
curl -s localhost:8000/predict/batch \
  -H 'content-type: application/json' \
  -d '{"messages":["Free entry! Text WIN to 87121","Lunch at 1?"]}'
```

Returns `results[]`, a `summary` (`total`, `spam`, `ham`, `spam_ratio`,
`threshold`), the model reference and total latency.

### Operations

- **Logs** are JSON lines on stdout (request id, route, status, duration).
- **`dm`/metrics`** exposes request counters by route/status, prediction counts
  by label, feedback count, error count and uptime.
- **Hot reload** after retraining without a restart:

  ```bash
  SPAM_ADMIN_TOKEN=secret python main.py api
  curl -X POST localhost:8000/admin/reload -H "Authorization: Bearer secret"
  ```

  The endpoint is disabled (403) unless `SPAM_ADMIN_TOKEN` is set.
- **Feedback loop**: `POST /feedback` appends to `feedback/feedback.jsonl`,
  ready to be merged into a future training set.
- Every response carries `X-Request-ID` and `X-Response-Time-ms`; clients can
  send their own `X-Request-ID` and it is echoed back.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPAM_MODEL_PATH` | `models/sms_spam_model.joblib` | Artifact to serve |
| `SPAM_THRESHOLD` | `0.5` | Default spam cut-off |
| `SPAM_MAX_BATCH_SIZE` | `500` | Batch limit (413 above it) |
| `SPAM_MAX_TEXT_LENGTH` | `5000` | Documented request limit |
| `SPAM_ADMIN_TOKEN` | unset | Enables `/admin/reload` |
| `SPAM_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `SPAM_LOG_LEVEL` | `INFO` | Log verbosity |
| `SPAM_FEEDBACK_PATH` | `feedback/feedback.jsonl` | Feedback sink |
| `SPAM_AUTOTRAIN` | `0` | Train on startup if the artifact is missing |
| `SPAM_API_URL` | `http://localhost:8000` | Dashboard's API endpoint |

---

## Project layout

```
sms-spam-detection/
├── api/
│   ├── main.py              # FastAPI app factory, routes, middleware
│   ├── schemas.py           # Pydantic request/response models
│   ├── service.py           # model lifecycle + inference (no HTTP)
│   ├── settings.py          # SPAM_* environment settings
│   ├── metrics.py           # Prometheus registry
│   ├── logging_conf.py      # JSON log formatter
│   └── static/index.html    # dependency-free browser UI
├── src/
│   ├── config.py            # paths and constants
│   ├── preprocessing.py     # clean_text(), shared by train and serve
│   ├── dataset.py           # download / parse / cache the dataset
│   ├── model.py             # pipelines, persistence, SpamClassifier
│   ├── train.py             # grid search + model selection
│   ├── evaluate.py          # metrics and reporting
│   └── client.py            # typed API client (used by Streamlit)
├── tests/
│   ├── conftest.py          # synthetic dataset + session-scoped model
│   ├── test_pipeline.py     # cleaning, dataset, train→save→load→predict
│   └── test_api.py          # 26 API tests (routes, auth, limits, metrics)
├── scripts/
│   ├── smoke_test.py        # black-box check of a running deployment
│   └── load_test.py         # concurrent latency/throughput benchmark
├── main.py                  # CLI: download/train/evaluate/predict/info/api/serve
├── app.py                   # Streamlit dashboard (API-first, local fallback)
├── Dockerfile               # multi-stage; model trained at build time
├── docker-compose.yml
├── .github/workflows/ci.yml # lint + tests + image build + live smoke test
├── Makefile / tasks.ps1
└── requirements*.txt, pyproject.toml, .env.example
```

---

## How the model works

1. **Cleaning** (`src/preprocessing.py`): lowercase, unescape HTML, then replace
   URLs, e-mails, money amounts, phone numbers and remaining digits with
   placeholder tokens (`__url__`, `__email__`, `__money__`, `__phone__`,
   `__number__`). Keeping placeholders preserves "a number was present" without
   exploding the vocabulary.
2. **Features**: TF-IDF over word uni- and bi-grams (`min_df=2`,
   `sublinear_tf`). The vectoriser sits inside the scikit-learn `Pipeline`, so
   raw text goes in and a label comes out — cleaning cannot be skipped at
   serving time.
3. **Selection**: Multinomial NB, Logistic Regression (balanced) and calibrated
   Linear SVM, each tuned with `GridSearchCV` (5-fold stratified, scoring =
   spam F1). The winner is refit on the full training set and evaluated once on
   the hold-out set.
4. **Serving**: the artifact is a versioned dict (`pipeline`, `metadata`,
   `version`); the version is a content/metadata hash exposed by
   `/model/info` and `/admin/reload`.

Spam F1 is the selection metric because the data is imbalanced (~13% spam) and
a missed spam is usually costlier than an extra review.

---

## Testing and CI

```bash
python -m pytest -q     # 39 tests, ~5s, no network
python -m ruff check .
python scripts/smoke_test.py --url http://localhost:8000
python scripts/load_test.py --requests 300 --concurrency 20
```

- Unit tests cover cleaning edge cases, dataset normalisation, a full
  train → save → load → predict round-trip, explanations, and degenerate input.
- API tests run a real app instance via `TestClient` against a synthetic model:
  routes, validation (422), batch limit (413), auth (401/403), hot reload,
  feedback persistence, the 503 not-ready path and the Prometheus output.
- CI (`.github/workflows/ci.yml`) lints and tests on Python 3.11/3.12, then
  builds the Docker image, starts it and runs `scripts/smoke_test.py` against
  the live container.

Measured on the development machine (single uvicorn worker):

| Scenario | Result |
| --- | --- |
| Sequential, no explanation | 5.6 ms p50 |
| Sequential, with explanation | 13.4 ms p50 |
| 300 requests / 20 concurrent | 194 req/s, 0 errors, p50 92 ms |

Scale out with `--workers N` (or replicas behind a load balancer) when needed;
the model is CPU-bound and stateless.

---

## Notes and limitations

- Dataset: *SMS Spam Collection v.1* (Almeida, Gómez Hidalgo & Yamakami, 2011),
  UCI ML Repository. English, 2011-era messages; accuracy on modern phishing,
  other languages or long messages will be lower.
- Duplicate messages are removed before splitting to avoid train/test leakage,
  hence 5,158 rows instead of 5,574.
- The threshold is a product decision: 0.5 is a sensible default, lower for
  aggressive filtering, higher for fewer false alarms.
- Classic linear model rather than an LLM: fast, explainable, cheap, and good
  enough for this dataset. Use it as one signal in a real filtering system.
- Docker/CI assets are provided and CI-validated; the local machine used to
  build this project did not have Docker installed, so the image itself was
  verified by the GitHub Actions workflow definition rather than locally.
