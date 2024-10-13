import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import lightgbm
from typing import List, Dict
from config import Config
from utils import load_data, write_data, write_encoding


def build_encoding(df: pd.DataFrame, cat_columns: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Encode each of column from `cat_columns` to its frequency rank (int)

    ToDo: union all cities with 1 transaction into one state to new category
    ToDo: union User and Card in one categorical feature

    :returns: dict with encodings for each column: {col_name: {col_value: 1, ...}} 
    """
    encoding = {}
    for column in cat_columns:
        df_frequency = df[column].value_counts().index.values
        encoding_column = {value: code for code, value in enumerate(df_frequency)}
        encoding[column] = encoding_column

    return encoding


def apply_encoding(df: pd.DataFrame, cat_columns: List[str], encoding: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    Apply encoding (mapper) to each of column from `cat_columns`.
    Add to each column name suffix `_encoded`, drop original column
    """
    for column in cat_columns:
        df[column + "_encoded"] = df[column].copy()
        df[column + "_encoded"] = df[column + "_encoded"].replace(encoding.get(column, {}))

    df = df.drop(columns=cat_columns)
    return df


def evaluate_metrics(config: Config):
    data = load_data(config, "data_prepared.csv")
    
    target = data["target"].values
    data = data.drop(columns=["target"])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    for train_index, test_index in kfold.split(data):
        train = data.iloc[train_index]
        y_train = target[train_index]

        test = data.iloc[test_index]
        y_test = target[test_index]

        encoding = build_encoding(train, config.categorical_columns)
        train_encoded = apply_encoding(train, config.categorical_columns, encoding)
        test_encoded = apply_encoding(test, config.categorical_columns, encoding)

        print(train_encoded.shape[0], test_encoded.shape[0])
        print(train.iloc[:10])
        print(train_encoded.iloc[:10])
        print("\n")
        
        model = lightgbm.LGBMClassifier()
        model.fit(train_encoded, y_train)
        predict = model.predict(test_encoded)

        metric = f1_score(y_test, predict)
        metrics.append(metric)

    metric_mean = np.mean(metrics)
    logging.info("F1-score on cross-validation: ", metric_mean)



config = Config()
evaluate_metrics(config)