import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    root_dir: str

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root_dir, "data", "raw")

    @property
    def train_dir(self) -> str:
        return os.path.join(self.raw_dir, "train")

    @property
    def test_csv(self) -> str:
        return os.path.join(self.raw_dir, "test.csv")


def load_all_training_tables(paths: DatasetPaths) -> Dict[str, pd.DataFrame]:
    """Load all CSVs from the train directory into memory.

    Returns a mapping from logical name to DataFrame.
    """
    train_dir = paths.train_dir
    tables = {
        "city_indexes": "city_indexes.csv",
        "city_search_index": "city_search_index.csv",
        "land_transactions": "land_transactions.csv",
        "land_transactions_nearby_sectors": "land_transactions_nearby_sectors.csv",
        "pre_owned_house_transactions": "pre_owned_house_transactions.csv",
        "pre_owned_house_transactions_nearby_sectors": "pre_owned_house_transactions_nearby_sectors.csv",
        "new_house_transactions": "new_house_transactions.csv",
        "new_house_transactions_nearby_sectors": "new_house_transactions_nearby_sectors.csv",
        "sector_POI": "sector_POI.csv",
    }

    loaded: Dict[str, pd.DataFrame] = {}
    for key, filename in tables.items():
        csv_path = os.path.join(train_dir, filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing expected training file: {csv_path}")
        loaded[key] = pd.read_csv(csv_path)

    return loaded


def load_test(paths: DatasetPaths) -> pd.DataFrame:
    csv_path = paths.test_csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing expected test file: {csv_path}")
    return pd.read_csv(csv_path)


def split_month_sector(df: pd.DataFrame, month_col: str = "month", sector_col: str = "sector") -> pd.DataFrame:
    """Convert Kaggle month string (e.g., '2019 Jan') and 'sector n' to year, month, time, sector_id.

    Adds columns: 'year', 'month_num', 'time', 'sector_id'.
    """
    month_codes = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    result = df.copy()
    if month_col in result.columns:
        # handle both '2019 Jan' and '2019-Jan'
        m = result[month_col].astype(str).str.replace('-', ' ', regex=False)
        result["year"] = m.str.slice(0, 4).astype(int)
        result["month_num"] = m.str.slice(5, None).map(month_codes)
        result["time"] = (result["year"] - 2019) * 12 + result["month_num"] - 1

    if sector_col in result.columns:
        result["sector_id"] = result[sector_col].astype(str).str.slice(7, None).astype(int)

    return result


def prepare_train_target(nht: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    """Return a (time x sector) matrix for the target 'amount_new_house_transactions'.

    Missing months/sectors are filled with zeros; includes sectors 1..96.
    """
    nht_aug = split_month_sector(nht)
    target_wide = (
        nht_aug.set_index(["time", "sector_id"]).amount_new_house_transactions.unstack()
    )
    # Ensure all 96 sectors present
    for sector in range(1, 97):
        if sector not in target_wide.columns:
            target_wide[sector] = 0
    target_wide = target_wide.sort_index(axis=1)
    target_wide = target_wide.fillna(0)
    return target_wide, target_wide.columns


def explode_test_id(test_df: pd.DataFrame) -> pd.DataFrame:
    """Split the 'id' column of the test set into month and sector, with derived fields.
    Adds 'month', 'sector', 'year', 'month_num', 'time', 'sector_id'.
    """
    test = test_df.copy()
    parts = test.id.str.split("_", expand=True)
    test["month"] = parts[0]
    test["sector"] = parts[1]
    test = split_month_sector(test)
    return test


