import logging
from typing import Dict, List

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from config import Config
from utils import load_data, write_encoding, write_meta, write_model


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
        encoding_column = {str(value): code for code, value in enumerate(df_frequency)}
        encoding[column] = encoding_column

    return encoding


def apply_encoding(df: pd.DataFrame, cat_columns: List[str], encoding: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    Apply encoding (mapper) to each of column from `cat_columns`.
    Add to each column name suffix `_encoded`, drop original column
    """
    for column in cat_columns:
        encoding_column = encoding.get(column, {})
        df[column + "_encoded"] = df[column].copy()
        df[column + "_encoded"] = df[column + "_encoded"].apply(lambda x: encoding_column.get(x, -1))

    df = df.drop(columns=cat_columns)
    return df


def evaluate_metrics(config: Config, plot_learning_curve: bool = False):
    """
    Load and encode data, compute metrics on cross-validation, write F1-score to log

    ToDo: save loss on train and validation, save train time
    ToDo: save learning curve to file

    :param config: current configuration
    :param plot_learning_curve: if True, plot learning curve on last fold
    """
    data = load_data(config, "data_prepared.csv")

    target = data["target"].values
    data = data.drop(columns=["target"]).sample(100000)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    for i, (train_index, test_index) in enumerate(kfold.split(data)):
        X_train = data.iloc[train_index]
        y_train = target[train_index]

        X_test = data.iloc[test_index]
        y_test = target[test_index]

        encoding = build_encoding(X_train, config.categorical_columns)
        train_encoded = apply_encoding(X_train, config.categorical_columns, encoding)
        test_encoded = apply_encoding(X_test, config.categorical_columns, encoding)

        model = lightgbm.LGBMClassifier(**config.model_params)
        model.fit(
            train_encoded,
            y_train,
            categorical_feature=[f"{c}_encoded" for c in config.categorical_columns],
            feature_name="auto",
            eval_set=[(X_test, y_test), (X_train, y_train)],
        )
        predict = model.predict(test_encoded)

        metric = f1_score(y_test, predict)
        metrics.append(metric)

    metric_mean = np.mean(metrics)
    logging.info("F1-score on cross-validation: ", metric_mean)

    meta = {"full_model_params": model.get_params(), "extra_model_params": config.model_params, "f1-score": metric_mean}

    if plot_learning_curve:
        lightgbm.plot_metric(model)
        plt.show()

    write_meta(config, meta, config.meta_filename)


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
