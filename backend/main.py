from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

app = FastAPI(title="AI Analytics Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend Running"}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), target: str = ""):

    contents = await file.read()

    df = pd.read_csv(BytesIO(contents)) if file.filename.endswith(".csv") else pd.read_excel(BytesIO(contents))

    if target not in df.columns:
        return {"error": f"Target column '{target}' not found"}

    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- CLASSIFICATION ----------------
    if y.dtype == "object" or len(np.unique(y)) < 20:

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "XGBoost": XGBClassifier(eval_metric="mlogloss")
        }

        scores = {}
        metrics = {}

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")

            scores[name] = round(acc, 4)

            metrics[name] = {
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4)
            }

        best_model = max(scores, key=scores.get)

        return {
            "task": "classification",
            "metric": "Accuracy",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "scores": scores,
            "metrics": metrics,
            "best_model": best_model
        }

    # ---------------- REGRESSION ----------------
    else:

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR(),
            "XGBoost": XGBRegressor()
        }

        scores = {}
        metrics = {}

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)

            scores[name] = round(r2, 4)

            metrics[name] = {
                "R2 Score": round(r2, 4),
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4)
            }

        best_model = max(scores, key=scores.get)

        return {
            "task": "regression",
            "metric": "R2 Score",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "scores": scores,
            "metrics": metrics,
            "best_model": best_model
        }
