import pandas as pd
from typing import List, Dict
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
    Write data as .csv to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Dataframe on path {path} already exists")
    
    df.to_csv(path, index=False)


def write_encoding(config: Config, encoding: Dict[str, Dict[str, int]], name: str, overwrite: False):
    """
    Write encoding as .json to config.base_path + name
    """
    path = config.base_path.joinpath(name)

    if not overwrite and path.exists():
        raise ValueError(f"Dataframe on path {path} already exists")
    
    with open(path) as file:
        json.dump(encoding, path)