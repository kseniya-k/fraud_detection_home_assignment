import datetime
import logging

import pandas as pd

from config import Config
from utils import load_data, write_data


def parse_dollar_column(df: pd.DataFrame, column: str, feature_name: str) -> pd.DataFrame:
    """
    Parse df column from format `$123.34` to float `123.45`. Save result to new column `feature_name`
    """
    df[feature_name] = df[column].apply(lambda x: float(x[1:]))
    return df


def parse_month_year_column(df: pd.DataFrame, column: str, feature_name: str) -> pd.DataFrame:
    """
    Parse df column from format `month/year` to datetime. Save result to new column `feature_name`
    """
    df[feature_name] = df[column].apply(lambda x: datetime.date(int(x.split("/")[-1]), int(x.split("/")[0]), 1))
    df[feature_name] = pd.to_datetime(df[feature_name])
    return df


def transform_data(
    config: Config, transactions: pd.DataFrame, cards: pd.DataFrame, users: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocess data:
    - parse date and money columns
    - join data
    - drop unused columns
    """
    transactions["target"] = transactions["Is Fraud?"].apply(lambda x: 0 if x == "No" else 1)
    transactions["datetime"] = transactions.apply(
        lambda x: datetime.datetime(
            x["Year"],
            x["Month"],
            x["Day"],
            int(x["Time"].split(":")[0]),
            int(x["Time"].split(":")[1]),
            0,
        ),
        axis=1,
    )
    transactions["datetime"] = pd.to_datetime(transactions["datetime"])

    # drop dollar sign $
    for column, feature_name in config.transactions_transform_dollar.items():
        transactions = parse_dollar_column(transactions, column, feature_name)

    for column, feature_name in config.cards_transform_dollar.items():
        cards = parse_dollar_column(cards, column, feature_name)

    for column, feature_name in config.users_transform_dollar.items():
        users = parse_dollar_column(users, column, feature_name)

    # parse dates
    for column, feature_name in config.cards_transform_month_year.items():
        cards = parse_month_year_column(cards, column, feature_name)

    # drop columns
    users = users.drop(columns=config.drop_columns_users)
    cards = cards.drop(columns=config.drop_columns_cards)

    # fillna
    for column in ["Merchant State", "Zip"]:
        transactions[column] = transactions[column].fillna("unknown")
    transactions["Errors?"] = transactions["Errors?"].fillna("no_error")

    # join
    data = transactions.merge(users, right_index=True, left_on="User").merge(
        cards.rename(columns={"CARD INDEX": "Card"}), on=["User", "Card"]
    )
    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract some simple features:
    - merchant_country: str - USA if `Merchant State` is a USA state, else country
    - is_city_equal: bool - if user city is equal to merchant city
    - amount_limit_rate: float - amount divided by user credit limit
    - debt_limit_rate: float - user debt divided by user credit limit
    - is_pin_change_year: bool - if transaction were in the year of changing card PIN
    - days_before_expire: int - days between transaction and card expiration
    - days_from_acc_open: int - days between bank account opening and transaction
    - errors_count: int - total count of transaction errors

    ToDo: add daily transactions count by user, by card, by appartment
    ToDo: add amount of time between transactions from one user and card
    """
    df["merchant_country"] = df["Merchant State"].apply(lambda x: "USA" if len(x) == 2 and x.upper() == x else x)
    df["is_city_equal"] = df.apply(lambda x: x["City"] == x["Merchant City"], axis=1)
    df["amount_limit_rate"] = df["amount"] / df["credit_limit"]
    df["debt_limit_rate"] = df["total_debt"] / df["credit_limit"]
    df["is_pin_change_year"] = df["Year PIN last Changed"] == df["Year"]
    df["days_before_expire"] = (df["expires"] - df["datetime"]).dt.days
    df["days_from_acc_open"] = (df["datetime"] - df["account_open_date"]).dt.days
    df["errors_count"] = df["Errors?"].apply(lambda x: len(x.split(",")) if x != "no_errors" else 0)
    return df


def prepare_data(config: Config, output_name: str):
    """
    Preprocess data:
    - parse date and money columns
    - join data
    - drop unused columns
    - save result as .csv to config.base_path + output_name

    Each dataframe expected to have all columns from example
    """
    transactions = load_data(config, config.filename_transactions)
    cards = load_data(config, config.filename_cards)
    users = load_data(config, config.filename_users)

    logging.info("Apply column transformations")
    data = transform_data(config, transactions, cards, users)

    logging.info("Add new features")
    data = add_features(data)
    data = data.drop(columns=config.drop_columns_data)
    write_data(config, data, "data_prepared.csv", overwrite=True)


config = Config()
print("confg opened")
prepare_data(config, "test_preparation")
print("data prepared")
