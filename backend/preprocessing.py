import pandas as pd
import numpy as np


def clean_data(df):

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')

        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    return df


def feature_engineering(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[f'{col}_square'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))

    return df