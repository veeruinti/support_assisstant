# Zepto Data & AI Platform

One repository containing three connected AI/ML modules.

## Setup

Create and activate a virtual environment, then install the consolidated dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python data_pipeline/pipeline.py
python analytics/01_eda.py
python analytics/02_modeling.py
uvicorn support_assistant.main:app --reload --port 7860
```

The data pipeline uses the assignment's fixed baseline **1 GBP = 105.50 INR**; it is not a live exchange rate. It scrapes the first five catalogue pages (100 books), normalizes category/book data into SQLite, saves query outputs, and checks that SQL JOIN and pandas merge agree.

The analytics module calls `sns.load_dataset('titanic')` only in `01_eda.py`, immediately stores `analytics/titanic.csv`, and all later work reads that file. `02_modeling.py` keeps preprocessing inside scikit-learn pipelines to prevent test-set leakage.

The support assistant defaults to `MOCK_LLM=1`, so its graded flow makes no LLM API calls. Local sentence-transformer embeddings and ChromaDB provide real retrieval; LangGraph routes queries to retrieval or a fixed direct response.
