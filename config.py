from pathlib import Path
from typing import Dict, List


class Config:
    """
    Class with constants and configurations
    """

    # input and output data location
    base_path: Path = Path("./data")
    filename_transactions: str = "credit_card_transactions-ibm_v2.csv"
    filename_cards: str = "sd254_cards.csv"
    filename_users: str = "sd254_users.csv"

    # columns for certain types of preprocessing
    # each input file has its own dict with format: {`column_name`: `feature_name`}
    transactions_transform_dollar: Dict[str, str] = {"Amount": "amount"}
    cards_transform_dollar: Dict[str, str] = {"Credit Limit": "credit_limit"}
    users_transform_dollar: Dict[str, str] = {
        "Per Capita Income - Zipcode": "imcome_per_zipcode",
        "Yearly Income - Person": "yearly_income_person",
        "Total Debt": "total_debt",
    }

    cards_transform_month_year = {
        "Expires": "expires",
        "Acct Open Date": "account_open_date",
    }

    # columns that will be deleted
    drop_columns_cards: List[str] = [
        "Card Number",
        "CVV",
        "Card on Dark Web",
        "Expires",
        "Credit Limit",
        "Acct Open Date",
    ]
    drop_columns_users: List[str] = [
        "Person",
        "Birth Year",
        "Birth Month",
        "Address",
        "Apartment",
        "Zipcode",
        "Latitude",
        "Longitude",
        "Per Capita Income - Zipcode",
        "Yearly Income - Person",
        "Total Debt",
    ]
    drop_columns_data: List[str] = [
        "Card",
        "Is Fraud?",
        "Month",
        "Day",
        "Time",
        "Amount",
        "Zip",
        "Errors?",
        "datetime",
        "expires",
        "account_open_date",
    ]

    # columns that will be encoded to int
    categorical_columns: List[str] = [
        "User",
        "Use Chip",
        "Merchant Name",
        "Merchant City",
        "Merchant State",
        "MCC",
        "Gender",
        "City",
        "State",
        "Card Brand",
        "Card Type",
        "Has Chip",
        "merchant_country",
    ]

    encoding_filename: str = "encoding.json"
    model_filename: str = "model.lightgbm"
    meta_filename: str = "meta.json"

    # extra params for model training
    validation_rate: float = 0.05
    model_params = {"n_estimators": 100, "scale_pos_weight": 99, "max_depth": 10}

    def get_path(self, filename: str) -> Path:
        """
        Concatenates base_path and filename
        """
        return self.base_path.joinpath(filename)
