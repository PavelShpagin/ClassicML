from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data import split_month_sector


def aggregate_monthly_totals(nht: pd.DataFrame) -> pd.DataFrame:
    """Simple aggregation: total amount per time over all sectors (for plotting)."""
    df = split_month_sector(nht)
    return df.groupby("time")["amount_new_house_transactions"].sum().reset_index()


def build_time_lagged_features(
    nht: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 6, 12],
    rolling_windows: List[int] = [3, 6, 12],
) -> pd.DataFrame:
    """Create lag and rolling features on a per-sector basis for the target series.

    Returns a long-format DataFrame keyed by ['time', 'sector_id'] with generated features.
    """
    df = split_month_sector(nht)
    df = df.sort_values(["sector_id", "time"]).copy()

    features: List[pd.DataFrame] = []
    for sector_id, g in df.groupby("sector_id", sort=False):
        g = g.set_index("time").sort_index()
        s = g["amount_new_house_transactions"].fillna(0)

        feat = pd.DataFrame(index=g.index)
        for lag in lags:
            feat[f"lag_{lag}"] = s.shift(lag)
        for w in rolling_windows:
            feat[f"roll_mean_{w}"] = s.rolling(w).mean()
            feat[f"roll_geo_mean_{w}"] = np.exp(np.log(s.replace(0, np.nan)).rolling(w).mean()).replace(
                [np.inf, -np.inf], np.nan
            )

        feat["sector_id"] = sector_id
        features.append(feat.reset_index())

    out = pd.concat(features, ignore_index=True)
    return out


def join_static_sector_features(
    per_time_df: pd.DataFrame,
    sector_poi: pd.DataFrame,
) -> pd.DataFrame:
    """Join static POI attributes by sector_id."""
    sp = sector_poi.copy()
    if "sector_id" not in sp.columns and "sector" in sp.columns:
        sp["sector_id"] = sp["sector"].astype(str).str.slice(7, None).astype(int)
    cols_to_use = [c for c in sp.columns if c not in {"sector"}]
    merged = per_time_df.merge(sp[cols_to_use], on="sector_id", how="left")
    return merged


