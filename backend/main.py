from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from preprocessing import clean_data, feature_engineering
from automl import run_automl

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def home():
    return {'message': 'AI Analytics Platform API Running'}


@app.post('/analyze/')
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str = 'target'
):

    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file)

    else:
        df = pd.read_excel(file.file)

    df = clean_data(df)

    df = feature_engineering(df)

    result = run_automl(df, target)

    return result