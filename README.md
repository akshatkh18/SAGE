
# 🧠 SAGE — Smart Adaptive Guided Experimentation

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-boosting-green)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-boosting-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-purple)](https://shap.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-hyperparameter--tuning-blue)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Your end-to-end ML co-pilot.** Upload a dataset, SAGE handles everything else — profiling, preprocessing, model training, explainability, and a professional PDF report. Zero cost, zero hardcoding.

🚀 **Live Demo:** [huggingface.co/spaces/akshatkh18/SAGE](https://huggingface.co/spaces/akshatkh18/SAGE) *(coming soon)*

---

## 🤔 Why SAGE?

Most AutoML tools are black boxes. They automate the pipeline but never tell you *why* they made each decision.

SAGE is different:
- Every preprocessing step comes with a **human-readable justification**
- The insight engine detects data quality issues with **severity levels** (critical / warning / info)
- SHAP explainability translates model decisions into **plain English**
- Every experiment is **logged to SQLite** so you can compare runs over time
- The final output is a **professional PDF report** you can share with anyone

- ---

## ⚙️ Features

| Module | What it does |
|--------|-------------|
| 🔍 **EDA** | Auto-profiles your dataset — missing values, skew, outliers, correlations, data quality insights |
| 🧹 **Preprocessing** | Adaptive pipeline — imputation, encoding, winsorizing, datetime extraction, ID detection |
| 🏟️ **Model Arena** | Trains 7 models, ranks by F1/R², runs Optuna tuning on the winner |
| 💡 **Explainability** | SHAP summary, bar, and waterfall plots + natural language explanations |
| 🗃️ **Experiment Memory** | SQLite logging — compare every run you've ever done |
| 📄 **PDF Report** | Professional 7-page report with charts, tables, and plain-language summary |

---

## 🧠 Models in the Arena
=======
# SAGE

<p align="center">
  <h1 align="center">Smart Adaptive Guided Experimentation</h1>

  <p align="center">
    An Explainable AutoML Platform that automates the complete Machine Learning workflow while keeping every decision transparent.
  </p>

  <p align="center">
    <a href="https://huggingface.co/spaces/akshat18/SAGE"><strong>Live Demo</strong></a>
    ·
    <a href="https://github.com/akshatkh18/SAGE/issues">Report Bug</a>
    ·
    <a href="https://github.com/akshatkh18/SAGE/issues">Request Feature</a>
  </p>
</p>

---

## Badges

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-blue)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-2F80ED)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)
![MIT License](https://img.shields.io/badge/License-MIT-yellow)

---

# Overview

SAGE (Smart Adaptive Guided Experimentation) is an end-to-end Explainable AutoML platform that automates the complete machine learning workflow, from dataset analysis to model interpretation.

Unlike conventional AutoML tools that often behave as black boxes, SAGE explains every preprocessing decision, interprets model predictions using SHAP, tracks experiments in SQLite, and generates professional PDF reports for reproducibility.

The platform is designed for students, researchers, data scientists, and machine learning practitioners who want automation without sacrificing transparency.

---

# Why SAGE?

Traditional AutoML platforms usually focus on obtaining the best model while hiding intermediate decisions.

SAGE focuses on **automation with explainability**.

Key differences include:

- Automated exploratory data analysis
- Adaptive preprocessing pipelines
- Human-readable preprocessing explanations
- Multi-model benchmarking
- Automatic hyperparameter optimization
- SHAP-based explainability
- Experiment history
- Professional PDF reports

---

# Features

## Exploratory Data Analysis

Automatically analyzes uploaded datasets and generates:

- Dataset overview
- Missing value analysis
- Outlier detection
- Correlation analysis
- Feature distributions
- Duplicate detection
- Skewness detection
- Data quality scoring
- Natural language insights

---

## Adaptive Preprocessing

The preprocessing engine dynamically builds pipelines based on dataset characteristics.

Supported operations:

- Missing value imputation
- Label Encoding
- One-Hot Encoding
- Datetime feature extraction
- Feature Scaling
- Winsorization
- High Cardinality Detection
- Identifier Column Detection
- Automatic preprocessing explanations

---

## Model Arena

Benchmarks multiple ML algorithms simultaneously.

Supported algorithms:
>>>>>>> df1a9a1b999ff21f0a7fbf7a301a8f90b57ae1ba

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
<<<<<<< HEAD
- XGBoost
- LightGBM
- KNN

Winner gets Optuna hyperparameter tuning (10–50 trials, auto-scaled to dataset size).

---

## 🛠️ Tech Stack

- **UI:** Streamlit
- **ML:** scikit-learn, XGBoost, LightGBM
- **Tuning:** Optuna
- **Explainability:** SHAP
- **Storage:** SQLite
- **Reports:** ReportLab
- **Logging:** Python logging (daily rotating log files)


---

## 🚀 Run Locally

```bash
git clone https://github.com/akshatkh18/SAGE.git
cd SAGE
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app/main.py
=======
- K Nearest Neighbors
- XGBoost
- LightGBM

The best-performing model is automatically selected for optimization.

---

## Hyperparameter Optimization

Uses Optuna to optimize the winning model.

Features include:

- Automatic search space
- Cross-validation
- Dataset-aware number of trials
- Performance comparison before and after tuning

---

## Explainable AI

Model predictions are interpreted using SHAP.

Available visualizations:

- Feature Importance
- SHAP Summary Plot
- SHAP Bar Plot
- SHAP Waterfall Plot

Every visualization is accompanied by natural language explanations.

---

## Experiment Tracking

Stores every experiment inside SQLite.

Logged information includes:

- Dataset metadata
- Selected preprocessing pipeline
- Model metrics
- Hyperparameters
- Timestamp
- Execution time

---

## Automated PDF Reports

Generate professional reports containing:

- Dataset Summary
- Data Quality Assessment
- EDA Visualizations
- Model Comparison
- Hyperparameter Results
- SHAP Interpretations
- Final Conclusions

---

# Workflow

```text
                    Dataset Upload
                          │
                          ▼
              Exploratory Data Analysis
                          │
                          ▼
             Adaptive Preprocessing Engine
                          │
                          ▼
                  Model Benchmarking
                          │
                          ▼
             Hyperparameter Optimization
                          │
                          ▼
                 Explainability (SHAP)
                          │
                          ▼
               Experiment Tracking
                          │
                          ▼
              Professional PDF Report
>>>>>>> df1a9a1b999ff21f0a7fbf7a301a8f90b57ae1ba
```

---

<<<<<<< HEAD
## 📁 Project Structure

```
sage/
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── eda/
│   │   ├── profiler.py          # Auto EDA engine
│   │   └── insights.py          # Rule-based NL insight engine
│   ├── preprocessing/
│   │   └── pipeline.py          # Adaptive preprocessing pipeline
│   ├── models/
│   │   └── arena.py             # 7-model arena + Optuna tuning
│   ├── explainability/
│   │   └── shap_engine.py       # SHAP explainer + NL layer
│   ├── storage/
│   │   └── db.py                # SQLite experiment logging
│   ├── reports/
│   │   └── generator.py         # PDF report generator
=======
# Technology Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Hyperparameter Optimization | Optuna |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Database | SQLite |
| Report Generation | ReportLab |
| Logging | Python Logging |

---

# Project Structure

```text
sage/
│
├── app/
│   ├── main.py
│   │
│   ├── eda/
│   │   ├── profiler.py
│   │   └── insights.py
│   │
│   ├── preprocessing/
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   └── arena.py
│   │
│   ├── explainability/
│   │   └── shap_engine.py
│   │
│   ├── reports/
│   │   └── generator.py
│   │
│   ├── storage/
│   │   └── db.py
│   │
>>>>>>> df1a9a1b999ff21f0a7fbf7a301a8f90b57ae1ba
│   ├── pages/
│   │   ├── preprocessing_page.py
│   │   ├── model_arena_page.py
│   │   ├── explainability_page.py
│   │   ├── experiments_page.py
│   │   └── report_page.py
<<<<<<< HEAD
│   └── utils/
│       ├── logger.py            # Centralized logging
│       └── exceptions.py        # Custom exception hierarchy
├── artifacts/                   # Models, plots, reports (gitignored)
├── requirements.txt
└── README.md
=======
│   │
│   └── utils/
│       ├── logger.py
│       └── exceptions.py
│
├── artifacts/
├── requirements.txt
├── README.md
└── LICENSE
>>>>>>> df1a9a1b999ff21f0a7fbf7a301a8f90b57ae1ba
```

---

<<<<<<< HEAD
## 📊 Sample Output

### Leaderboard
| Model | F1 | AUC | Accuracy |
|-------|----|-----|----------|
| 🥇 LightGBM | 0.6131 | 0.8688 | 0.8725 |
| XGBoost | 0.5706 | 0.8315 | 0.8525 |
| Random Forest | 0.5652 | 0.8500 | 0.8600 |

### SHAP Feature Importance (Churn Dataset)
| Feature | Mean SHAP |
|---------|-------------|
| NumOfProducts | 0.864 |
| Age | 0.779 |
| IsActiveMember | 0.383 |

---

## 👨‍💻 Built By

**Akshat Gupta** — B.Tech AI/ML, JECRC University (2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-akshatkh18-black?logo=github)](https://github.com/akshatkh18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/akshat-gupta18)

---
=======
# Installation

## Clone the Repository

```bash
git clone https://github.com/akshatkh18/SAGE.git
cd SAGE
```

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/main.py
```

---

# Usage

1. Launch the Streamlit application.

2. Upload any structured CSV dataset.

3. Explore automatically generated EDA insights.

4. Review preprocessing decisions.

5. Train multiple machine learning models.

6. Compare evaluation metrics.

7. Optimize the best-performing model.

8. Interpret predictions using SHAP.

9. Download the generated PDF report.

---

# Supported Models

| Model | Classification | Regression |
|---------|:--------------:|:----------:|
| Logistic Regression | ✓ | ✓ |
| Decision Tree | ✓ | ✓ |
| Random Forest | ✓ | ✓ |
| Extra Trees | ✓ | ✓ |
| K-Nearest Neighbors | ✓ | ✓ |
| XGBoost | ✓ | ✓ |
| LightGBM | ✓ | ✓ |

---

# Evaluation Metrics

## Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Regression

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Explainability

SAGE integrates SHAP to provide interpretable machine learning.

Generated outputs include:

- Global Feature Importance
- SHAP Summary Plot
- SHAP Bar Plot
- SHAP Waterfall Plot
- Natural Language Explanations

This enables users to understand why the model made specific predictions instead of treating it as a black box.

---

# Sample Results

## Model Leaderboard

| Rank | Model | F1 Score | ROC-AUC | Accuracy |
|------|----------------|---------:|---------:|----------:|
| 🥇 | LightGBM | **0.6131** | **0.8688** | **0.8725** |
| 🥈 | XGBoost | 0.5706 | 0.8315 | 0.8525 |
| 🥉 | Random Forest | 0.5652 | 0.8500 | 0.8600 |
| 4 | Extra Trees | 0.5578 | 0.8427 | 0.8575 |
| 5 | Decision Tree | 0.5214 | 0.7813 | 0.8250 |
| 6 | Logistic Regression | 0.5039 | 0.7965 | 0.8362 |
| 7 | KNN | 0.4917 | 0.7542 | 0.8213 |

---

## SHAP Feature Importance

| Feature | Mean SHAP Value |
|----------|----------------:|
| NumOfProducts | 0.864 |
| Age | 0.779 |
| IsActiveMember | 0.383 |
| Balance | 0.271 |
| Geography | 0.194 |

---

# Future Roadmap

The following features are planned for future releases.

- Deep Learning support
- Time Series Forecasting
- Automatic Feature Engineering
- Feature Selection Module
- MLflow Integration
- Model Deployment
- Docker Support
- Multi-user Authentication
- Cloud Storage Integration
- LLM-powered Dataset Insights
- Distributed Training
- REST API
- Batch Prediction
- CI/CD Integration
- Custom Model Plugins

---

# Performance Goals

- Fully automated ML workflow
- Explainable predictions
- Minimal manual preprocessing
- Reproducible experiments
- Production-ready reporting

---

# Contributing

Contributions are welcome.

If you would like to improve SAGE:

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature/your-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push to your branch.

```bash
git push origin feature/your-feature
```

5. Open a Pull Request.

Please ensure that your code follows the existing project structure and coding style.

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# Acknowledgements

SAGE is built using several outstanding open-source projects.

- Streamlit
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Optuna
- Pandas
- NumPy
- Matplotlib
- Seaborn
- ReportLab

Special thanks to the maintainers and contributors of these libraries.

---

# Author

**Akshat Gupta**

B.Tech in Artificial Intelligence and Machine Learning  
JECRC University, Jaipur, India

GitHub: https://github.com/akshatkh18

LinkedIn: https://linkedin.com/in/akshat-gupta18

---

<p align="center">

Built with Python, Streamlit, Scikit-learn, XGBoost, LightGBM, SHAP and Optuna.

If you found this project useful, consider giving it a ⭐ on GitHub.

</p>
>>>>>>> df1a9a1b999ff21f0a7fbf7a301a8f90b57ae1ba
