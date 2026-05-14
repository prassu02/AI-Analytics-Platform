from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = FastAPI()

# -----------------------------------
# CORS
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# HOME
# -----------------------------------
@app.get("/")
def home():

    return {
        "message": "AI Analytics Platform API Running"
    }

# -----------------------------------
# ANALYZE
# -----------------------------------
@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    target: str = ""
):

    try:

        # -----------------------------
        # READ FILE
        # -----------------------------
        if file.filename.endswith(".csv"):

            df = pd.read_csv(file.file)

        else:

            df = pd.read_excel(file.file)

        # -----------------------------
        # VALIDATE TARGET
        # -----------------------------
        if target not in df.columns:

            return {
                "error": f"Target column '{target}' not found"
            }

        # -----------------------------
        # CLEAN
        # -----------------------------
        df = df.drop_duplicates()

        df = df.fillna(0)

        # -----------------------------
        # FEATURES
        # -----------------------------
        X = df.drop(columns=[target])

        y = df[target]

        X = pd.get_dummies(X)

        # -----------------------------
        # TASK DETECTION
        # -----------------------------
        if y.dtype == "object" or len(np.unique(y)) < 20:

            task = "classification"

            le = LabelEncoder()

            y = le.fit_transform(y)

            model1 = LogisticRegression(max_iter=1000)

            model2 = RandomForestClassifier()

        else:

            task = "regression"

            model1 = LinearRegression()

            model2 = RandomForestRegressor()

        # -----------------------------
        # SPLIT
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # -----------------------------
        # MODEL 1
        # -----------------------------
        model1.fit(X_train, y_train)

        pred1 = model1.predict(X_test)

        # -----------------------------
        # MODEL 2
        # -----------------------------
        model2.fit(X_train, y_train)

        pred2 = model2.predict(X_test)

        # -----------------------------
        # SCORE
        # -----------------------------
        if task == "classification":

            score1 = accuracy_score(y_test, pred1)

            score2 = accuracy_score(y_test, pred2)

        else:

            score1 = r2_score(y_test, pred1)

            score2 = r2_score(y_test, pred2)

        scores = {
            "Logistic/Linear": float(score1),
            "RandomForest": float(score2)
        }

        best_model = max(scores, key=scores.get)

        # -----------------------------
        # RETURN JSON
        # -----------------------------
        return {
            "task": task,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "scores": scores,
            "best_model": best_model
        }

    except Exception as e:

        return {
            "error": str(e)
        }
