# QuickEDA

QuickEDA is a simple, container-ready EDA tool built with Streamlit.

A lightweight Streamlit-based Exploratory Data Analysis (EDA) tool.

## Features

- Upload CSV file
- Basic statistics
- Data preview
- Simple visualization

---

## Setup (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run (Local)

```bash
streamlit run app.py
```

Then open:

http://localhost:8501

---

## Run with Docker

```bash
docker build -t quickeda .
docker run -p 8501:8501 quickeda
```

Then open:

http://localhost:8501
