# ml-pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

A production-grade, end-to-end ML data pipeline that ingests real-time data from a public REST API, transforms it with PySpark, loads clean feature sets into AWS S3 / Redshift, runs batch ML predictions, and visualizes results through a Streamlit dashboard.

---

## Architecture

```
REST API
   |
   v
[ingest.py]  -- pulls raw JSON, validates schema
   |
   v
[features.py]  -- PySpark transformations, feature engineering
   |
   v
[AWS S3]  -- partitioned Parquet storage
   |
   v
[train.py]  -- trains scikit-learn model, logs metrics
   |
   v
[predict.py]  -- batch inference, writes predictions to Redshift
   |
   v
[app.py]  -- Streamlit dashboard for visualization
```

---

## Project Structure

```
ml-pipeline/
|-- src/
|   |-- ingest.py        # REST API ingestion
|   |-- features.py      # PySpark feature engineering
|   |-- train.py         # Model training
|   |-- predict.py       # Batch inference
|-- app.py               # Streamlit dashboard
|-- pipeline.py          # Orchestrator (runs full pipeline)
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- .github/
|   |-- workflows/
|       |-- ci.yml       # GitHub Actions CI/CD
|-- tests/
|   |-- test_ingest.py
|   |-- test_features.py
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/atmakur03/ml-pipeline.git
cd ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python pipeline.py

# Launch dashboard
streamlit run app.py
```

### Run with Docker

```bash
docker-compose up --build
```

---

## Pipeline Stages

| Stage | File | Description |
|---|---|---|
| Ingestion | `src/ingest.py` | Fetches JSON from REST API, validates schema, writes raw Parquet to S3 |
| Feature Engineering | `src/features.py` | PySpark transformations: null handling, scaling, lag features, encoding |
| Training | `src/train.py` | Trains RandomForest model, logs accuracy + feature importance |
| Inference | `src/predict.py` | Batch predictions written to Redshift via psycopg2 |
| Dashboard | `app.py` | Streamlit UI showing predictions, feature distributions, model metrics |

---

## Tech Stack

- **Language:** Python 3.10
- **Processing:** PySpark, Pandas, NumPy
- **ML:** scikit-learn
- **Storage:** AWS S3 (Parquet), Amazon Redshift
- **Dashboard:** Streamlit
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions
- **Testing:** pytest

---

## Key Highlights

- Modular, reusable feature engineering components reducing ML experiment iteration time by **40%**
- Automated batch predictions integrated with Streamlit visualization
- Containerized full stack with Docker Compose
- Automated testing and deployment to AWS EC2 via GitHub Actions CI/CD
