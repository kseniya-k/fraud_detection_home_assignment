import json
from typing import Any, Dict

import lightgbm
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


def write_data(config: Config, df: pd.DataFrame, name: str, overwrite: bool = True):
    """
    Write data to .csv file to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Dataframe on path {path} already exists")

    df.to_csv(path, index=False)


def write_encoding(
    config: Config,
    encoding: Dict[str, Dict[str, int]],
    name: str,
    overwrite: bool = True,
):
    """
    Write encoding to .json file to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Encoding on path {path} already exists")

    with open(path, "w") as file:
        json.dump(encoding, file)


def load_encoding(config: Config, name: str) -> Dict[str, Dict[str, int]]:
    """
    Load encoding from .json file config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not path.exists():
        raise ValueError(f"Encoding not found on path {path}")

    with open(path, "r") as file:
        encoding = json.load(file)

    return encoding


def write_model(config: Config, model: Any, name: str, overwrite: bool = True):
    """
    Write model as .lightgbm file to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Model on path {path} already exists")

    model.booster_.save_model(path)


def load_model(config: Config, name: str):
    """
    Load model from .lightgbm file config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not path.exists():
        raise ValueError(f"Model not found on path {path}")

    return lightgbm.Booster(model_file=path)


def write_meta(config: Config, meta: Dict[str, Any], name: str, overwrite: bool = True):
    """
    Write meta-information about config and model parameters to .json file to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Meta on path {path} already exists")

    with open(path, "w") as file:
        json.dump(meta, file)
