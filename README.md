# 🚀 AI Analytics Platform

An **end-to-end Machine Learning + AutoML + Explainable AI platform** built using:

* ⚡ FastAPI (Backend ML Engine)
* 🎨 Streamlit (Frontend Dashboard)
* 🤖 Scikit-learn + XGBoost + SVM + RandomForest
* 📊 Plotly (Visualization)
* 📈 Advanced Metrics (R2, MAE, MSE, RMSE)
* 🧠 AutoML Pipeline
* 📦 Real-time Model Comparison

---

# 🌐 Live Demo

### 🚀 Frontend (Streamlit)

👉 [https://ai-analytics-platform-kqeyp7jxpnpmhqpslurdjm.streamlit.app/](https://ai-analytics-platform-kqeyp7jxpnpmhqpslurdjm.streamlit.app/)

### ⚡ Backend (FastAPI)

👉 [https://ai-analytics-platform-1.onrender.com/docs](https://ai-analytics-platform-1.onrender.com/docs)

---

# 📌 Features

## 🤖 AutoML Engine

* Automatic model training
* Regression & Classification detection
* Model comparison system

### Supported Models:

* Linear Regression / Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* XGBoost (Advanced Boosting Model)

---

## 📊 Advanced Metrics System

For regression tasks:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

For classification:

* Accuracy Score
* Model-wise comparison

---

## 📈 Visualization Dashboard

* Model performance bar charts
* R² comparison graph
* Dataset preview
* Statistical overview

---

## 🧠 AI Intelligence Features

* Auto task detection (Regression / Classification)
* Best model selection
* Performance ranking system

---

## ⚡ Backend (FastAPI)

* Fast ML inference API
* File upload support (CSV / Excel)
* JSON response system
* Auto model training pipeline

### Endpoint

```
POST /analyze/?target=column_name
```

---

## 🎨 Frontend (Streamlit)

* Interactive dashboard
* File upload UI
* Real-time API integration
* Metrics visualization
* Best model highlight

---

# 🏗 Architecture

```
                ┌─────────────────────┐
                │  Streamlit Frontend │
                │  (UI Dashboard)     │
                └─────────┬───────────┘
                          │ API Call
                          ▼
        ┌─────────────────────────────────┐
        │       FastAPI Backend           │
        │  AutoML + Model Training       │
        │  Metrics + Prediction Engine   │
        └─────────┬──────────────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │ ML Models (Sklearn, XGB)  │
     │ -Logistic, SVM, Linear, XGB    │
     └───────────────────────────┘
```

---

# 📂 Project Structure

```
AI-Analytics-Platform/
│
├── backend/
│   ├── main.py          # FastAPI backend
|   ├── __pycache__     
│   ├── runtime.txt
|   ├── requirements.txt
│
├── frontend/
│   ├── app.py          # Streamlit dashboard
|   ├── runtime.txt
│   ├── requirements.txt
|
├── requirements.txt
├── README.md
```

---

# ⚙️ Tech Stack

## 🧠 Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

## ⚡ Backend

* FastAPI
* Uvicorn

## 🎨 Frontend

* Streamlit
* Plotly

---

# 🚀 API Response Format

### Example Output:

```json
{
  "task": "regression",
  "metric": "R2 Score",
  "rows": 81,
  "columns": 5,
  "scores": {
    "LinearRegression": 0.675,
    "RandomForest": 0.8098,
    "SVR": 0.5215,
    "XGBoost": 0.8302
  },
  "metrics": {
    "RandomForest": {
      "R2 Score": 0.8098,
      "MAE": 2.4554,
      "MSE": 18.7019,
      "RMSE": 4.3246
    }
  },
  "best_model": "XGBoost"
}
```

---

# 📊 Use Cases

* Automated Machine Learning (AutoML)
* Business Data Analytics
* Model Benchmarking
* Data Science Learning Platform
* AI Dashboard Systems

---

# 🔥 How to Run Locally

## 1️⃣ Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 2️⃣ Frontend

```bash
cd frontend
streamlit run app.py
```

---

# 🌍 Deployment

## Backend (Render)

* FastAPI deployed on Render
* Public API endpoint enabled

## Frontend (Streamlit Cloud)

* Connected to GitHub repo
* Live dashboard available

---

# ⚠️ Important Notes

* Always pass correct `target` column
* Dataset must be clean numeric format for best results
* Large datasets may take time due to ML training

---

# 🚀 Future Enhancements

* 🔐 User authentication system
* 💾 Save trained models
* 📥 Download model (.pkl)
* 📊 SHAP explainability graphs
* 📄 PDF report generation
* ⚡ Real-time prediction API
* 🐳 Docker deployment

---

# 👨‍💻 Author

**Prasanna Kumar Kommiri**

Machine Learning Engineer | AI Systems Developer | Data Scientist

---

# ⭐ If you like this project

Give a ⭐ on GitHub and share it with others.
