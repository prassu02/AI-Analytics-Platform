import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from xgboost import XGBClassifier, XGBRegressor


def run_automl(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if y.dtype == 'object' or len(np.unique(y)) < 20:
        task = 'classification'
    else:
        task = 'regression'

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if task == 'classification':

        le = LabelEncoder()

        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC(),
            'XGBoost': XGBClassifier()
        }

        metric_name = 'Accuracy'

    else:

        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(),
            'SVR': SVR(),
            'XGBoost': XGBRegressor()
        }

        metric_name = 'R2 Score'

    scores = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        if task == 'classification':
            score = accuracy_score(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)

        scores[name] = round(score, 4)

    result = {
        'task': task,
        'metric': metric_name,
        'scores': scores,
        'best_model': max(scores, key=scores.get)
    }

    return result