# 🐛 BugScope — Full Stack Bug Tracking & Analytics Dashboard

A full-stack bug tracking and analytics platform powered by Python, Streamlit, and scikit-learn,
with a standalone `index.html` frontend dashboard.

## 📁 Project Structure

```
bug_tracker/
├── backend/
│   ├── analytics.py        # scikit-learn ML models
│   └── streamlit_app.py    # Streamlit web application
├── data/
│   ├── generate_data.py    # Generates 500 sample bugs
│   ├── bugs.csv            # Generated dataset
│   └── analytics.json      # Pre-computed analytics
├── frontend/
│   └── index.html          # Standalone HTML dashboard
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate sample data
```bash
python data/generate_data.py
```

### 3. Run ML analytics
```bash
python backend/analytics.py
```

### 4. Launch Streamlit app
```bash
cd backend
streamlit run streamlit_app.py
```

### 5. Open HTML dashboard
Open `frontend/index.html` in any browser — no server required!

## 🤖 ML Models (scikit-learn)

| Model | Task | Metric |
|-------|------|--------|
| Random Forest Classifier | Severity Prediction | 34% Accuracy |
| Gradient Boosting Regressor | Resolution Time | MAE: 7.46 days |
| KMeans Clustering | Bug Grouping | k=4 clusters |

## 📊 Dashboard Features

- **Dashboard**: KPI cards, component/status/severity/assignee charts
- **ML Insights**: Feature importances, resolution predictions, cluster analysis
- **Bug Explorer**: Searchable, filterable table of all 500 bugs
- **Trends**: Monthly volume, environment breakdown, stacked severity chart

## 📦 Tech Stack

- **Backend**: Python, pandas, numpy, scikit-learn
- **Frontend (app)**: Streamlit
- **Frontend (static)**: Vanilla HTML/CSS/JS + Chart.js
- **Data**: 500 synthetic bugs across 10 components, 6 assignees
