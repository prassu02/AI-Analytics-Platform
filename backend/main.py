from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from xgboost import XGBClassifier, XGBRegressor

app = FastAPI()

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
@app.get("/")
@app.head("/")
def home():

    return {
        "message": "AI Analytics Platform Backend Running"
    }

# ---------------------------------------------------
# ANALYZE DATASET
# ---------------------------------------------------
@app.post("/analyze/")
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str = ""
):

    try:

        # -------------------------------------------
        # READ FILE
        # -------------------------------------------
        if file.filename.endswith(".csv"):

            df = pd.read_csv(file.file)

        else:

            df = pd.read_excel(file.file)

        # -------------------------------------------
        # CHECK TARGET
        # -------------------------------------------
        if target not in df.columns:

            return {
                "error": f"Target column '{target}' not found"
            }

        # -------------------------------------------
        # CLEANING
        # -------------------------------------------
        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].mean()
        )

        # -------------------------------------------
        # FEATURES + TARGET
        # -------------------------------------------
        X = df.drop(columns=[target])

        y = df[target]

        # -------------------------------------------
        # ENCODING
        # -------------------------------------------
        X = pd.get_dummies(X)

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0)

        # -------------------------------------------
        # TASK DETECTION
        # -------------------------------------------
        if y.dtype == "object" or len(y.unique()) < 20:

            task = "classification"

        else:

            task = "regression"

        # -------------------------------------------
        # SPLIT
        # -------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # -------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------
        if task == "classification":

            encoder = LabelEncoder()

            y_train = encoder.fit_transform(y_train)

            y_test = encoder.transform(y_test)

            models = {

                "LogisticRegression":
                LogisticRegression(max_iter=1000),

                "RandomForest":
                RandomForestClassifier(),

                "SVM":
                SVC(),

                "XGBoost":
                XGBClassifier(
                    eval_metric="mlogloss"
                )

            }

            metric_name = "Accuracy"

        # -------------------------------------------
        # REGRESSION
        # -------------------------------------------
        else:

            models = {

                "LinearRegression":
                LinearRegression(),

                "RandomForest":
                RandomForestRegressor(),

                "SVR":
                SVR(),

                "XGBoost":
                XGBRegressor()

            }

            metric_name = "R2 Score"

        # -------------------------------------------
        # TRAIN MODELS
        # -------------------------------------------
        scores = {}

        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            if task == "classification":

                score = accuracy_score(
                    y_test,
                    predictions
                )

            else:

                score = r2_score(
                    y_test,
                    predictions
                )

            scores[name] = round(float(score), 4)

        # -------------------------------------------
        # BEST MODEL
        # -------------------------------------------
        best_model = max(
            scores,
            key=scores.get
        )

        # -------------------------------------------
        # RESPONSE
        # -------------------------------------------
        return {

            "task": task,

            "metric": metric_name,

            "rows": int(df.shape[0]),

            "columns": int(df.shape[1]),

            "scores": scores,

            "best_model": best_model

        }

    except Exception as e:

        return {
            "error": str(e)
        }
