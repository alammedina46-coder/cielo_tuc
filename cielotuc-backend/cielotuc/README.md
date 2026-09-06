# CIELO·TUC Backend

AI-powered hyper-local weather prediction for Tucumán, Argentina.

## Stack
- **FastAPI** — async REST API
- **PostgreSQL + PostGIS** — spatial database
- **Redis + Celery** — task queue and scheduler
- **PyTorch CNN-LSTM** — weather prediction model
- **MLflow** — model versioning and tracking

## Quick start

```bash
# 1. Copy env file and fill in your keys
cp .env.example .env

# 2. Start all services
docker compose up -d

# 3. Initialize database (creates tables + seeds data)
python scripts/init_db.py

# 4. Train the initial model (downloads ~10 years of data)
python scripts/train_initial_model.py --years 10

# 5. API is live at http://localhost:8000
# 6. API docs at http://localhost:8000/docs
# 7. MLflow at http://localhost:5000
```

## Project structure

```
cielotuc/
├── app/
│   ├── api/v1/endpoints/   ← FastAPI routers
│   ├── core/               ← Config, settings
│   ├── db/                 ← SQLAlchemy session
│   ├── ml/
│   │   ├── models/         ← CNN-LSTM architecture
│   │   ├── pipeline/       ← Data ingestion + feature engineering
│   │   └── training/       ← Training loop + MLflow
│   ├── models/             ← ORM models (DB tables)
│   ├── schemas/            ← Pydantic request/response schemas
│   ├── services/           ← Business logic + Celery tasks
│   └── main.py             ← App factory
├── scripts/
│   ├── init_db.py          ← First-run DB setup
│   └── train_initial_model.py ← Initial model training
├── tests/
│   ├── unit/               ← ML model unit tests
│   └── integration/        ← API integration tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /api/v1/zones/ | All Tucumán zones |
| GET | /api/v1/zones/{id} | Zone detail |
| GET | /api/v1/forecast/{zone_id} | Full forecast (current + 12h + 7d) |
| GET | /api/v1/forecast/{zone_id}/zonda | Zonda risk index |
| GET | /api/v1/sensors/status | All sensor status |
| POST | /api/v1/sensors/readings | Ingest sensor reading |
| POST | /api/v1/alerts/flood | Manual FLOOD·TUC alert |
| GET | /api/v1/alerts/flood/history | Alert log |
| GET | /api/v1/model/metrics | Active model performance |
| GET | /api/v1/model/comparison | vs SMN vs Weather.com |
| POST | /api/v1/model/retrain | Trigger retraining |

## Celery scheduled tasks

| Task | Schedule | Description |
|------|----------|-------------|
| fetch_weather_data | Every 15 min | Pull from SMN + NASA APIs |
| run_predictions | Every 30 min | Run model for all zones |
| validate_predictions | Every 30 min | Compare predictions vs reality |
| retrain_model | Monthly (day 1, 02:00) | Full retraining pipeline |

## Running tests

```bash
# Unit tests (no DB required)
pytest tests/unit/ -v

# Integration tests (requires running DB)
pytest tests/integration/ -v

# All tests
pytest -v
```
