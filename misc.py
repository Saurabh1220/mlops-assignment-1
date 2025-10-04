from __future__ import annotations
import numpy as np, pandas as pd
from typing import Tuple, Iterable
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def load_boston_df() -> pd.DataFrame:
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_xy(df: pd.DataFrame, target_col: str="MEDV", test_size: float=0.2, seed: int=42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_tr, X_te, y_tr, y_te

def make_preprocessor(feature_names: Iterable[str]) -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), list(feature_names))], remainder="drop")

def make_pipeline(model, feature_names: Iterable[str]) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(feature_names)), ("model", model)])

def train_and_eval(model, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, y_te: np.ndarray) -> float:
    pipe = make_pipeline(model, X_tr.columns)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    return mean_squared_error(y_te, preds)
