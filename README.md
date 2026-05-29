<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/IBM_HR_Dataset-1470_Records-0062FF?style=for-the-badge&logo=ibm&logoColor=white"/>

# 🧠 AI-Powered Employee Attrition Prediction System

### *Predict who's about to leave — before they do.*

**IBM HR Analytics · 6 ML Models · Real-time Risk Scoring · Interactive Dashboards**

[🚀 Live Demo](https://ai-powered-employee-attrition-prediction-system-rbr.streamlit.app/) · [📊 Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) 

</div>

---

## 🌟 Overview

An enterprise-grade, AI-powered HR analytics platform built on the **IBM HR Analytics Employee Attrition & Performance** dataset. This system combines statistical EDA with 6 machine learning models to predict employee attrition risk, segment employees into behavioral cohorts, and provide actionable HR interventions — all through a stunning, modern Streamlit UI.

> **Real Business Problem:** Organizations lose employees every month. HR teams need data-driven tools to identify *who* is at risk, *why* they may leave, and *what* to do about it — before it's too late.

---

## 📸 Application Screenshots

| Dashboard | EDA | ML Models |
|-----------|-----|-----------|
| KPI Overview | Interactive Charts | Performance Comparison |

| Prediction | Segmentation | Explorer |
|-----------|--------------|---------|
| Risk Scoring | K-Means + PCA | Dataset Filtering |

---

## ✨ Key Features

### 📊 Phase 1 — Exploratory Data Analysis
- **Attrition Rate** — Overall 16.1% workforce attrition visualized with donut charts
- **Department Analysis** — Sales (20.6%), HR (19.0%), R&D (13.8%) breakdown
- **Gender Analysis** — Comparative attrition gauges by gender
- **Salary Impact** — Box plots & income band analysis showing income–attrition correlation
- **Experience Analysis** — Tenure histograms, promotion gap analysis, overtime impact
- **Satisfaction Radar** — Job/Environment/Relationship/Work-Life balance profiles
- **Correlation Matrix** — Feature correlation heatmap with attrition target
- **Descriptive Statistics** — Complete statistical summary table

### 🤖 Phase 2 — Machine Learning Models

| Model | Purpose | Key Output |
|-------|---------|------------|
| **Logistic Regression** | Binary classification baseline | Probability + confidence |
| **Decision Tree** | HR rule generation | IF-THEN interpretable rules |
| **Random Forest** | Production model | Feature importance + predictions |
| **SVM** | Accuracy comparison | Kernel-based classification |
| **KNN** | Similar employee finder | Top-5 most similar profiles |
| **Naive Bayes** | Probabilistic prediction | Bayesian attrition probability |

### 🔮 Live Employee Predictor
- 30+ employee attributes input form
- Ensemble prediction from all 6 models
- Real-time risk gauge (Low / Medium / High)
- Personalized HR action recommendations
- Similar employee lookup via KNN

### 🧩 Employee Segmentation
- **PCA** dimensionality reduction (2D & 3D visualization)
- **K-Means clustering** (configurable 2–6 clusters)
- Cluster profiling: High Performers · At Risk · New Employees
- Cluster comparison bar charts

### 📋 Dataset Explorer
- Multi-dimensional filters (Department, Gender, Income, Attrition)
- Highlighted attrition rows
- Column selector
- CSV export

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/BharathReddyRamasani/AI-Powered-Employee-Attrition-Prediction-System.git
cd AI-Powered-Employee-Attrition-Prediction-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add IBM HR dataset
# Place WA_Fn-UseC_-HR-Employee-Attrition.csv in the data/ folder
# Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# 4. Run the application
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
AI-Powered-Employee-Attrition-Prediction-System/
│
├── 📄 app.py                    # Main Streamlit application (7 pages)
├── 📋 requirements.txt          # Python dependencies
├── 📒 Assessment.ipynb          # Original Jupyter notebook
│
├── 📁 src/                      # Source modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing pipeline
│   ├── eda.py                   # 10 Plotly EDA visualizations
│   ├── models.py                # 6 ML models + evaluation utilities
│   └── clustering.py            # PCA + K-Means segmentation
│
├── 📁 assets/
│   └── style.css                # Premium dark UI stylesheet
│
└── 📁 data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM HR dataset (add manually)
```

---

## 🗂️ Dataset

**IBM HR Analytics Employee Attrition & Performance**

| Attribute | Value |
|-----------|-------|
| Source | [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Records | 1,470 employees |
| Features | 35 attributes |
| Target | `Attrition` (Yes/No) |
| Attrition Rate | 16.1% |

**Key Features Used:**
- Demographics: Age, Gender, MaritalStatus, Education
- Work: Department, JobRole, JobLevel, BusinessTravel, OverTime
- Compensation: MonthlyIncome, DailyRate, HourlyRate, PercentSalaryHike
- Satisfaction: JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance
- Tenure: YearsAtCompany, TotalWorkingYears, YearsInCurrentRole

---

## 📊 Key Findings

### EDA Insights
- 🔴 **Sales department** has the highest attrition at **20.6%**
- 💰 **Income gap**: Attrited employees earn ~**$1,500 less/month** on average
- ⏰ **Overtime workers** show **~2x** higher attrition risk
- 📅 **Early-career employees** (< 2 years) are most vulnerable
- 😟 **Low job satisfaction** (score 1) dramatically increases attrition risk

### ML Model Performance

| Model | Accuracy | AUC-ROC | F1 Score |
|-------|----------|---------|---------|
| Random Forest | ~87% | ~82% | ~65% |
| Logistic Regression | ~85% | ~80% | ~58% |
| Decision Tree | ~82% | ~75% | ~55% |
| SVM | ~86% | ~81% | ~57% |
| KNN | ~83% | ~76% | ~50% |
| Naive Bayes | ~78% | ~78% | ~54% |

*Note: Actual metrics depend on dataset version used.*

### Decision Tree Rules (Examples)
```
IF JobSatisfaction ≤ 1 AND OverTime = Yes AND MonthlyIncome ≤ 4000
THEN Attrition = HIGH RISK

IF TotalWorkingYears > 10 AND StockOptionLevel ≥ 1 AND JobSatisfaction ≥ 3
THEN Attrition = LOW RISK
```

---

## 🎨 UI Design System

Built with a premium **dark glassmorphism** design:

| Element | Value |
|---------|-------|
| Background | `#0a0e1a` (Deep Navy) |
| Cards | `#111827` + glassmorphism |
| Primary Accent | `#6366f1` (Indigo) |
| Secondary Accent | `#f59e0b` (Amber) |
| Danger | `#ef4444` (Red) |
| Success | `#10b981` (Emerald) |
| Typography | Inter (Google Fonts) |

**Features:** Animated gradient headers · Glassmorphism cards · Color-coded risk badges · 
Micro-animations · Custom scrollbars · Responsive layout

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **UI Framework** | Streamlit 1.32+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Plotly (interactive charts) |
| **Styling** | Custom CSS (glassmorphism, Inter font) |
| **Dataset** | IBM HR Analytics (Kaggle) |

---

## 📱 Application Pages

| # | Page | Description |
|---|------|-------------|
| 1 | 🏠 **Overview Dashboard** | KPIs, attrition gauge, key insights |
| 2 | 📊 **EDA — Phase 1** | 8 tabs of interactive EDA charts |
| 3 | 🤖 **ML Models — Phase 2** | 8 tabs with all model metrics & ROC curves |
| 4 | 🌳 **Decision Tree Rules** | IF-THEN HR decision rules visualization |
| 5 | 🔮 **Predict Employee** | Live prediction form with ensemble scoring |
| 6 | 🧩 **Employee Segmentation** | K-Means + PCA 2D/3D clustering |
| 7 | 📋 **Dataset Explorer** | Filterable, exportable employee data table |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Bharath Reddy Ramasani**

[![GitHub](https://img.shields.io/badge/GitHub-BharathReddyRamasani-181717?style=flat-square&logo=github)](https://github.com/BharathReddyRamasani)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ using Python, Streamlit & Scikit-learn*

</div>
