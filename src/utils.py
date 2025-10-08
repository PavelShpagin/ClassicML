from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


# Month short-name to month number mapping used across scripts
MONTH_CODES: Dict[str, int] = {
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


def build_amount_wide(train_nht: pd.DataFrame) -> pd.DataFrame:
	"""Return a (time x sector) matrix for amount_new_house_transactions with sectors 1..96.

	The input DataFrame must contain columns: ['month', 'sector', 'amount_new_house_transactions'].
	"""
	# Parse sector_id
	df = train_nht.copy()
	df["sector_id"] = df["sector"].astype(str).str.extract(r"(\d+)").astype(int)

	# Parse month into year, month_num, and time index from 2019 Jan = 0
	df["year"] = df["month"].astype(str).str.split("-").str[0].astype(int)
	df["month_name"] = df["month"].astype(str).str.split("-").str[1]
	df["month_num"] = df["month_name"].map(MONTH_CODES)
	df["time"] = (df["year"] - 2019) * 12 + df["month_num"] - 1

	# Pivot to wide format and ensure all sectors 1..96 exist
	amount = df.set_index(["time", "sector_id"])  # type: ignore[arg-type]
	amount = amount["amount_new_house_transactions"].unstack()
	amount = amount.fillna(0)
	for sector in range(1, 97):
		if sector not in amount.columns:
			amount[sector] = 0
	amount = amount[[i for i in range(1, 97)]]
	return amount


def geometric_mean_with_zero_guard(
	amount_wide: pd.DataFrame,
	lookback_months: int = 6,
	zero_guard_window: int = 6,
) -> pd.Series:
	"""Compute per-sector geometric mean over the last N months with a zero guard.

	If any of the last `zero_guard_window` months is zero for a sector, prediction is set to 0.
	"""
	last_months = amount_wide.tail(lookback_months)
	geo = np.exp(np.log(last_months.replace(0, np.nan)).mean(axis=0, skipna=True))
	geo = geo.fillna(0)
	zero_mask = amount_wide.tail(zero_guard_window).min(axis=0) == 0
	geo[zero_mask] = 0
	return geo


def compute_december_boost(amount_wide: pd.DataFrame, cap: float = 2.0, default: float = 1.3) -> Dict[int, float]:
	"""Compute per-sector December boost factor from historical ratios.

	Boost is mean(December) / mean(non-December), computed per sector, capped by `cap`.
	If insufficient data, falls back to `default`.
	"""
	dec_mask = (amount_wide.index.to_series() % 12) == 11
	boost: Dict[int, float] = {}
	for sector in range(1, 97):
		series = amount_wide[sector].astype(float)
		dec_vals = series[dec_mask]
		non_dec_vals = series[~dec_mask & ((amount_wide.index.to_series() % 12) != 11)]
		dec_vals = dec_vals[dec_vals > 0]
		non_dec_vals = non_dec_vals[non_dec_vals > 0]
		if len(dec_vals) > 0 and len(non_dec_vals) > 0:
			ratio = float(dec_vals.mean() / non_dec_vals.mean())
			boost[sector] = min(ratio, cap)
		else:
			boost[sector] = default
	return boost


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


