# PromoSense — Causal Promotion & Pricing Intelligence

PromoSense is an end-to-end data science decision-support platform that estimates the **incremental effect of promotions**, models **customer and product price sensitivity**, and recommends promotion strategies that maximize **incremental profit rather than raw volume or revenue**.

Traditional analytics compares sales before and during a discount. PromoSense estimates the **counterfactual outcome**:
> *What would demand and profit have been if the promotion had not occurred?*

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────┐
│                   SvelteKit Dashboard                  │
│       Visualization + Interactive Scenario Simulator   │
└───────────────────────────┬────────────────────────────┘
                            │ REST
                            ▼
┌────────────────────────────────────────────────────────┐
│                        FastAPI                         │
│       API + Schema Validation + Simulation Engine      │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
       ┌──────────────┐          ┌───────────────────┐
       │  PostgreSQL  │          │   Model Services  │
       │              │          │                   │
       │ Analytics DB │          │ Baseline Demand   │
       │ Transactions │          │ Causal / Uplift   │
       │ Promotions   │          │ Price Elasticity  │
       └──────────────┘          └─────────┬─────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │     MLflow      │
                                  │ Model Registry  │
                                  └─────────────────┘
```

---

## Directory Structure

```text
promosense/
├── .github/workflows/       # GitHub Actions CI pipeline
├── backend/                 # FastAPI REST API & services
│   ├── app/
│   │   ├── api/             # API route endpoints
│   │   ├── core/            # Database connection & configurations
│   │   ├── models/          # SQLAlchemy analytical models
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── services/        # Prediction & scenario simulation services
│   │   └── main.py          # FastAPI application entrypoint
│   └── tests/               # API endpoint unit tests
├── data/                    # Data directory (git-ignored)
│   ├── raw/                 # Original Dunnhumby source CSVs
│   ├── interim/             # Cleaned parquet partitions
│   └── processed/           # Analytical feature tables
├── docker/                  # Docker initialization scripts
├── docs/                    # Methodological & data audit documentation
│   ├── technical_specification.md
│   ├── dataset_inventory.md
│   ├── dataset_data_dictionary.md
│   ├── dataset_to_architecture_mapping.md
│   ├── causal_feasibility_audit.md
│   ├── model_feasibility_matrix.md
│   └── phase_0_5_decisions.md
├── frontend/                # SvelteKit user dashboard (Phase 14)
├── ml_pipeline/             # Production ML & Causal modules
│   ├── src/
│   │   ├── causal/          # Propensity scoring & EconML meta-learners
│   │   ├── config/          # Pipeline settings & hyperparameters
│   │   ├── data/            # Data validation & ETL loaders
│   │   ├── elasticity/      # Log-log econometric elasticity models
│   │   ├── evaluate/        # Qini curves, AUUC, RMSE metrics
│   │   ├── features/        # Leakage-free temporal feature engineering
│   │   ├── models/          # XGBoost / LightGBM baseline demand models
│   │   └── optimization/    # Profit grid search & simulation engine
│   └── tests/               # Feature, causal, and leakage unit tests
├── notebooks/               # Exploratory notebooks (EDA only)
├── docker-compose.yml       # Local PostgreSQL + MLflow orchestrator
├── Makefile                 # Developer CLI shortcuts
├── pyproject.toml           # Unified Python configuration (PEP 621, Ruff, Mypy)
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker & Docker Compose

### 2. Environment Setup
```bash
# Clone the repository and navigate inside
cd promosense

# Copy the environment template
cp .env.example .env

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3. Start Infrastructure
Spin up the PostgreSQL analytical store and MLflow tracking server:
```bash
make docker-up
```
- **PostgreSQL**: `localhost:5432` (db: `promosense`, user: `promosense_user`)
- **MLflow UI**: `http://localhost:5000`

### 4. Code Quality & Testing
```bash
# Run linting
make lint

# Auto-format code
make format

# Run test suite
make test
```

---

## Methodological Guardrails

1. **True Shelf Price Reconstruction**: In-store retail discounts and coupons are explicitly added back to sales value to derive the actual pre-promotion shelf price. Elasticity is modeled against shelf price, avoiding reverse causality.
2. **Leakage Prevention**: All customer behavioral features are strictly computed prior to campaign start dates (`DAY < START_DAY`).
3. **Transparent Scenario Assumptions**: Because the Dunnhumby dataset does not contain wholesale unit costs, profit optimization operates on user-specified margin assumptions and campaign costs with full disclosure in all UI views and API payloads.
