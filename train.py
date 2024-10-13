import logging
from typing import Dict, List

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from config import Config
from utils import load_data, write_encoding, write_model


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
        encoding_column = encoding.get(column, {})
        df[column + "_encoded"] = df[column].apply(lambda x: encoding_column.get(x, -1))

    df = df.drop(columns=cat_columns)
    return df


def evaluate_metrics(config: Config):
    """
    Load and encode data, compute metrics on cross-validation, write F1-score to log
    ToDo: save loss on train and validation, save train time
    """
    data = load_data(config, "data_prepared.csv")

    target = data["target"].values
    data = data.drop(columns=["target"]).sample(100000)

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

        model = lightgbm.LGBMClassifier(is_unbalance=True)
        model.fit(
            train_encoded,
            y_train,
            categorical_feature=[f"{c}_encoded" for c in config.categorical_columns],
            feature_name="auto",
        )
        predict = model.predict(test_encoded)

        metric = f1_score(y_test, predict)
        metrics.append(metric)

    metric_mean = np.mean(metrics)
    logging.info("F1-score on cross-validation: ", metric_mean)


def train_production(config: Config):
    """
    Train model for production: load data and encode, train model, save encoding and model
    ToDo: save metrics on train
    """
    data = load_data(config, "data_prepared.csv").sample(100000)
    target = data["target"].values
    data = data.drop(columns=["target"])

    encoding = build_encoding(data, config.categorical_columns)
    data_encoded = apply_encoding(data, config.categorical_columns, encoding)
    write_encoding(config, encoding, config.encoding_filename)

    model = lightgbm.LGBMClassifier()
    model.fit(
        data_encoded,
        target,
        categorical_feature=[f"{c}_encoded" for c in config.categorical_columns],
        feature_name="auto",
    )
    write_model(config, model, config.model_filename)


config = Config()
evaluate_metrics(config)
train_production(config)
