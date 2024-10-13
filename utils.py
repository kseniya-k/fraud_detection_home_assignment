from typing import Any, Dict, List

import pandas as pd

from config import Config


def load_data(config: Config, name: str) -> pd.DataFrame:
    """
    Load .csv data from config.base_path + name
    """
    path = config.get_path(name)
    if not path.exists():
        raise ValueError(f"Dataframe not found on path {path}")

    df = pd.read_csv(path)

    if df.shape[0] == 0:
        raise ValueError(f"Dataframe on path {path} is empty")

    return df


def write_data(config: Config, df: pd.DataFrame, name: str, overwrite: False):
    """
    Write data to .csv file to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Dataframe on path {path} already exists")

    df.to_csv(path, index=False)


def write_encoding(
    config: Config, encoding: Dict[str, Dict[str, int]], name: str, overwrite: False
):
    """
    Write encoding to .json file to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Encoding on path {path} already exists")

    with open(path) as file:
        json.dump(encoding, path)


def write_model(config: Config, model: Any, name: str, overwrite: False):
    """
    Write model as .lightgbm file to to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Model on path {path} already exists")

    model.booster_.save_model(path)
