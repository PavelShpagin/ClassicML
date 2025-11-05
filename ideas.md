# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv
/kaggle/input/china-real-estate-demand-prediction/test.csv
/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv
/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv
/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv
/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv
/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv
/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv
import polars as pl
import polars.selectors as cs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv
from sklearn.model_selection import TimeSeriesSplit
import os
pth = "/kaggle/input/china-real-estate-demand-prediction"

def add_prefix(df, prefix, exclude=("sector", "month")):
    return df.rename(lambda c: c if c in exclude else f"{prefix}{c}")

ci = (
    pl.read_csv(f"{pth}/train/city_indexes.csv")
      .head(6)
      .fill_null(-1)
      .drop("total_fixed_asset_investment_10k")
      .pipe(add_prefix, prefix="ci_")
)

csi = pl.read_csv(f"{pth}/train/city_search_index.csv")

sp = (
    pl.read_csv(f"{pth}/train/sector_POI.csv")
      .fill_null(-1)
      .pipe(add_prefix, prefix="sp_")
)

train_lt = (
    pl.read_csv(f"{pth}/train/land_transactions.csv", infer_schema_length=10000)
      .pipe(add_prefix, prefix="lt_")
)

train_ltns = (
    pl.read_csv(f"{pth}/train/land_transactions_nearby_sectors.csv")
      .pipe(add_prefix, prefix="ltns_")
)

train_pht = (
    pl.read_csv(f"{pth}/train/pre_owned_house_transactions.csv")
      .pipe(add_prefix, prefix="pht_")
)

train_phtns = (
    pl.read_csv(f"{pth}/train/pre_owned_house_transactions_nearby_sectors.csv")
      .pipe(add_prefix, prefix="phtns_")
)

train_nht = (
    pl.read_csv(f"{pth}/train/new_house_transactions.csv")
      .pipe(add_prefix, prefix="nht_")
)

train_nhtns = (
    pl.read_csv(f"{pth}/train/new_house_transactions_nearby_sectors.csv")
      .pipe(add_prefix, prefix="nhtns_")
)

test = (
    pl.read_csv(f"{pth}/test.csv")
      .with_columns(id_split=pl.col("id").str.split("_"))
      .with_columns(
          month=pl.col("id_split").list.get(0),
          sector=pl.col("id_split").list.get(1),
      )
      .drop("id_split")
)
month_codes = {m: i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1)}
data = (
    pl.DataFrame(train_nht["month"].unique())
    .join(
        pl.DataFrame(train_nht["sector"].unique().to_list() + ["sector 95"])
        .rename({"column_0": "sector"}),
        how="cross",
    )
    .with_columns(
        sector_id=pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8),
        year=pl.col("month").str.split("-").list.get(0).cast(pl.Int16),
        month_num=pl.col("month").str.split("-").list.get(1)
            .replace(month_codes)
            .cast(pl.Int8),
    )
    .with_columns(
        time=((pl.col("year") - 2019) * 12 + pl.col("month_num") - 1).cast(pl.Int8)
    )
    .sort("sector_id", "time")
    .join(train_nht, on=["sector", "month"], how="left")
    .fill_null(0)

    .join(train_nhtns, on=["sector", "month"], how="left")
    .fill_null(-1)

    .join(train_pht, on=["sector", "month"], how="left")
    .fill_null(-1)

    .join(train_phtns, on=["sector", "month"], how="left")
    .fill_null(-1)

    .join(ci.rename({"ci_city_indicator_data_year": "year"}), on=["year"], how="left")
    .fill_null(-1)

    .join(sp, on=["sector"], how="left")
    .fill_null(-1)

    .join(train_lt, on=["sector", "month"], how="left")
    .fill_null(-1)

    .join(train_ltns, on=["sector", "month"], how="left")
    .fill_null(-1)
    .with_columns(cs.float().cast(pl.Float32))
)
for col in data.columns:
    if data[col].dtype == pl.Int64:
        c_min, c_max = data[col].min(), data[col].max()

        if c_min == 0 and c_max == 0:
            data = data.drop(col)
            print(col, "0" * 20)
            continue

        if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max and c_max < np.iinfo(np.int8).max:
            data = data.with_columns(pl.col(col).cast(pl.Int8))
        elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max and c_max < np.iinfo(np.int16).max:
            data = data.with_columns(pl.col(col).cast(pl.Int16))
        elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max and c_max < np.iinfo(np.int32).max:
            data = data.with_columns(pl.col(col).cast(pl.Int32))
        elif np.iinfo(np.int64).min < c_min < np.iinfo(np.int64).max and c_max < np.iinfo(np.int64).max:
            data = data.with_columns(pl.col(col).cast(pl.Int64))

        print(col, data[col].dtype, c_min, c_max)
        
data = data.drop("month","sector","year")
nht_num_new_house_transactions Int16 0 2669
nht_area_new_house_transactions Int32 0 294430
nht_price_new_house_transactions Int32 0 208288
nht_area_per_unit_new_house_transactions Int16 0 2003
nht_num_new_house_available_for_sale Int16 0 12048
nht_area_new_house_available_for_sale Int32 0 1220617
pht_area_pre_owned_house_transactions Int32 -1 126073
pht_num_pre_owned_house_transactions Int16 -1 1277
ci_national_year_end_total_population_10k Int32 -1 141260
ci_gdp_per_capita_yuan Int32 -1 156427
ci_annual_average_wage_urban_non_private_employees_yuan Int32 -1 147947
ci_annual_average_wage_urban_non_private_on_duty_employees_yuan Int32 -1 152324
ci_number_of_universities Int8 -1 84
ci_number_of_middle_schools Int16 -1 555
ci_number_of_primary_schools Int16 -1 992
ci_number_of_kindergartens Int16 -1 2223
ci_hospitals_health_centers Int16 -1 6159
ci_number_of_operating_bus_lines Int8 -1 18
ci_operating_bus_line_length_km Int16 -1 643
ci_number_of_industrial_enterprises_above_designated_size Int16 -1 6878
ci_total_current_assets_10k Int32 -1 159482630
ci_total_fixed_assets_10k Int32 -1 39934058
ci_main_business_taxes_and_surcharges_10k Int32 -1 4862051
ci_real_estate_development_investment_completed_10k Int32 -1 31022573
ci_residential_development_investment_completed_10k Int32 -1 20870742
ci_science_expenditure_10k Int32 -1 2439456
ci_education_expenditure_10k Int32 -1 6269391
sp_population_scale Int32 -1 31077700
sp_residential_area Int16 -1 24964
sp_office_building Int16 -1 13187
sp_commercial_area Int16 -1 3681
sp_resident_population Int32 -1 17099643
sp_office_population Int32 -1 26152000
sp_number_of_shops Int32 -1 1267755
sp_catering Int32 -1 350067
sp_retail Int32 -1 776801
sp_hotel Int32 -1 64186
sp_transportation_station Int32 -1 75064
sp_education Int16 -1 24366
sp_leisure_and_entertainment Int32 -1 52882
sp_bus_station_cnt Int16 -1 17281
sp_subway_station_cnt Int16 -1 418
sp_rentable_shops Int32 -1 43378
sp_leisure_entertainment_entertainment_venue_game_arcade Int16 -1 453
sp_leisure_entertainment_entertainment_venue_party_house Int16 -1 558
sp_leisure_entertainment_cultural_venue_cultural_palace Int16 -1 186
sp_office_building_industrial_building_industrial_building Int8 -1 0
sp_education_training_school_education_middle_school Int16 -1 867
sp_education_training_school_education_primary_school Int16 -1 1289
sp_education_training_school_education_kindergarten Int16 -1 4189
sp_education_training_school_education_research_institution Int16 -1 1302
sp_medical_health Int32 -1 55785
sp_medical_health_specialty_hospital Int16 -1 3430
sp_medical_health_tcm_hospital Int8 -1 63
sp_medical_health_physical_examination_institution Int8 -1 123
sp_medical_health_veterinary_station Int8 -1 124
sp_medical_health_pharmaceutical_healthcare Int32 -1 34583
sp_medical_health_rehabilitation_institution Int16 -1 5249
sp_medical_health_first_aid_center Int8 -1 120
sp_medical_health_blood_donation_station Int16 -1 186
sp_medical_health_disease_prevention_institution Int16 -1 422
sp_medical_health_general_hospital Int16 -1 2613
sp_medical_health_clinic Int16 -1 9248
sp_transportation_facilities_service_bus_station Int16 -1 15291
sp_transportation_facilities_service_subway_station Int16 -1 2377
sp_transportation_facilities_service_airport_related Int8 -1 4
sp_transportation_facilities_service_port_terminal Int16 -1 240
sp_transportation_facilities_service_train_station Int16 -1 589
sp_transportation_facilities_service_light_rail_station Int8 -1 0
sp_transportation_facilities_service_long_distance_bus_station Int16 -1 308
sp_number_of_leisure_and_entertainment_stores Int8 -1 15
sp_number_of_other_stores Int8 -1 27
sp_number_of_other_anchor_stores Int8 -1 9
sp_number_of_home_appliance_stores Int8 -1 113
sp_number_of_skincare_cosmetics_stores Int8 -1 106
sp_number_of_fashion_stores Int16 -1 622
sp_number_of_service_stores Int8 -1 67
sp_number_of_jewelry_stores Int8 -1 109
sp_number_of_lifestyle_leisure_stores Int8 -1 2
sp_number_of_supermarket_convenience_stores Int8 -1 69
sp_number_of_catering_food_stores Int16 -1 238
sp_number_of_residential_commercial Int8 -1 5
sp_number_of_office_building_commercial Int8 -1 5
sp_number_of_commercial_buildings Int8 -1 8
sp_number_of_hypermarkets Int8 -1 3
sp_number_of_department_stores Int8 -1 4
sp_number_of_shopping_centers Int8 -1 13
sp_number_of_hotel_commercial Int8 -1 1
sp_number_of_third_tier_shopping_malls_in_business_district Int8 -1 1
sp_number_of_second_tier_shopping_malls_in_business_district Int8 -1 1
sp_number_of_city_winner_malls Int8 -1 3
sp_number_of_shopping_malls_with_street_facing_shops Int8 -1 0
sp_number_of_unranked_malls Int8 -1 27
sp_number_of_community_malls Int8 -1 3
sp_number_of_community_winner_malls Int8 -1 3
sp_number_of_key_focus_malls Int8 -1 0
sp_shopping_malls_with_street_facing_shops_dense Int8 -1 0
sp_key_focus_malls_dense Int8 -1 0
sp_transportation_facilities_service_light_rail_station_dense Int8 -1 0
sp_office_building_industrial_building_industrial_building_dense Int8 -1 0
lt_num_land_transactions Int8 -1 5
data2 = data.sort("time", "sector_id")

for m in [1, 2, 12]:
    data2 = data2.join(
        data.drop("month_num").with_columns(pl.col("time") + m),
        on=["sector_id", "time"],
        how="left",
        suffix=f"_{m}"
    )

data2 = data2.sort("time", "sector_id")
lag=-1
data3 = data2.with_columns(
    pl.col("nht_amount_new_house_transactions")
      .shift(lag)
      .over("sector_id")
      .alias("label"),

    cs=((pl.col("month_num") - 1) / 6 * np.pi).cos(),
    sn=((pl.col("month_num") - 1) / 6 * np.pi).sin(),
    cs6=((pl.col("month_num") - 1) / 3 * np.pi).cos(),
    sn6=((pl.col("month_num") - 1) / 3 * np.pi).sin(),
    cs3=((pl.col("month_num") - 1) / 1.5 * np.pi).cos(),
    sn3=((pl.col("month_num") - 1) / 1.5 * np.pi).sin(),
)
data3 = data3.drop("sector_id")
cat_features = [
    "month_num"
]

border = 66 + lag - 12 * 0 - 1
border1 = 6 * 3

trainPool = Pool(
    data=data3
        .filter(pl.col("time") <= border)
        .filter(pl.col("time") > border1)
        .drop(["label"])
        .to_pandas()
        .fillna(-2),

    label=data3
        .filter(pl.col("time") <= border)
        .filter(pl.col("time") > border1)["label"]
        .to_pandas(),

    cat_features=cat_features,
)

testPool = Pool(
    data=data3
        .filter(pl.col("time") > border)
        .filter(pl.col("time") <= 66 + lag)
        .drop(["label"])
        .to_pandas()
        .fillna(-2),

    label=data3
        .filter(pl.col("time") > border)
        .filter(pl.col("time") <= 66 + lag)["label"]
        .to_pandas(),

    cat_features=cat_features,
)
def custom_score(y_true, y_pred, eps=1e-12):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return 0.0

    ape = np.abs((y_true - np.maximum(y_pred,0) ) / np.maximum(y_true, eps))

    bad_rate = np.mean(ape > 1.0)
    if bad_rate > 0.30:
        return 0.0

    mask = ape <= 1.0
    good_ape = ape[mask]

    if good_ape.size == 0:
        return 0.0

    mape = np.mean(good_ape)

    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    score = max(0.0, 1.0 - scaled_mape)
    return score

class CustomMetric:
    def is_max_optimal(self):
        return True # greater is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        y_pred = approx
        y_true = target

        output_weight = 1 # weight is not used

        score = custom_score( target, approx )
 
        return score, output_weight

    def get_final_error(self, error, weight):
        return error
#custom loss for catboost
class CustomObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        result = []
        delta = 10**(-6)
        for i in range(len(targets)):
            diff = (targets[i] - approxes[i])
            der1 = np.sign(diff) if (2*targets[i] - approxes[i]) < 0 else np.sign(diff)*5
            der2 = 0
            result.append((der1, der2))
        return result
testPool2 = Pool(
    data=data3
        .filter(pl.col("time") == 66)
        .drop(["label"])
        .to_pandas()
        .fillna(-2),
    cat_features=cat_features,
)

cb = CatBoostRegressor(
    iterations=21000,
    learning_rate=0.0125,
    one_hot_max_size=256,
    custom_metric=[
        "RMSE",
        "MAPE",
        "SMAPE",
        "MAE",
    ],
    loss_function=CustomObjective(),
    eval_metric=CustomMetric(),
    l2_leaf_reg=0.3,
    random_seed=4,
)

cb.fit(
    trainPool,
    eval_set=testPool,
    verbose=0,
)

month = np.maximum(cb.predict(testPool2), 0)
/usr/local/lib/python3.11/dist-packages/catboost/core.py:1790: UserWarning: Failed to optimize method "evaluate" in the passed object:
Failed in nopython mode pipeline (step: nopython frontend)
Untyped global name 'custom_score': Cannot determine Numba type of <class 'function'>

File "../../tmp/ipykernel_13/1764738461.py", line 43:
<source missing, REPL/exec in use?>

  self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
# ---------------------------
#  Imports
# ---------------------------
import numpy as np
import pandas as pd

from scipy.stats import norm


# ---------------------------
#  Month mapping
# ---------------------------
def build_month_codes():
    return {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }


# ---------------------------
#  Parse id into month text and sector string
# ---------------------------
def split_test_id_column(df):
    parts = df.id.str.split('_', expand=True)
    df['month_text'] = parts[0]
    df['sector'] = parts[1]
    return df


# ---------------------------
#  Add parsed time fields to a dataframe
# ---------------------------
def add_time_and_sector_fields(df, month_codes):
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    if 'month' not in df.columns:
        df['month'] = df['month_text'].str.slice(5, None).map(month_codes)
        df['year'] = df['month_text'].str.slice(0, 4).astype(int)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    else:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    return df


# ---------------------------
#  Load competition tables used for submission
# ---------------------------
def load_competition_data():
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
    return train_nht, test


# ---------------------------
#  Build training matrix: amount_new_house_transactions [time x sector_id]
# ---------------------------
def build_amount_matrix(train_nht, month_codes):
    train_nht = add_time_and_sector_fields(train_nht.copy(), month_codes)
    pivot = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
    pivot = pivot.fillna(0)
    all_sectors = np.arange(1, 97)
    for s in all_sectors:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[all_sectors]
    return pivot


# ---------------------------
#  Compute sector-level December multipliers from training
# ---------------------------
def compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.8, clip_high=1.5):
    is_december = (a_tr.index.values % 12) == 11
    dec_means = a_tr[is_december].mean(axis=0)
    nondec_means = a_tr[~is_december].mean(axis=0)
    dec_counts = a_tr[is_december].notna().sum(axis=0)
    raw_mult = dec_means / (nondec_means + eps)
    overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))
    raw_mult = raw_mult.where(dec_counts >= min_dec_obs, overall_mult)
    raw_mult = raw_mult.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    clipped_mult = raw_mult.clip(lower=clip_low, upper=clip_high)
    return clipped_mult.to_dict()


# ---------------------------
#  Apply December bump on the forecast horizon
# ---------------------------
def apply_december_bump(a_pred, sector_to_mult):
    dec_rows = [t for t in a_pred.index.values if (t % 12) == 11]
    if len(dec_rows) == 0:
        return a_pred
    for sector in a_pred.columns:
        m = sector_to_mult.get(sector, 1.0)
        a_pred.loc[dec_rows, sector] = a_pred.loc[dec_rows, sector] * m
    return a_pred


# ---------------------------
#  Exponential weighted geometric mean per sector
# ---------------------------
def ewgm_per_sector(a_tr, sector, n_lags, alpha):
    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)], dtype=float)
    weights = weights / weights.sum()
    recent_vals = a_tr.tail(n_lags)[sector].values
    if (len(recent_vals) != n_lags) or (recent_vals <= 0).all():
        return 0.0
    mask = recent_vals > 0
    pos_vals = recent_vals[mask]
    pos_w = weights[mask]
    if pos_vals.size == 0:
        return 0.0
    pos_w = pos_w / pos_w.sum()
    log_vals = np.log(pos_vals + 1e-12)
    wlm = np.sum(pos_w * log_vals) / pos_w.sum()
    return float(np.exp(wlm))
    
def simple_wm_per_sector(a_tr, sector, n_lags, alpha):
    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)], dtype=float)
    weights = weights / weights.sum()
    recent_vals = a_tr.tail(n_lags)[sector].values
    if (len(recent_vals) != n_lags) or (recent_vals <= 0).all():
        return 0.0
    mask = recent_vals > 0
    pos_vals = recent_vals[mask]
    pos_w = weights[mask]
    if pos_vals.size == 0:
        return 0.0
    pos_w = pos_w / pos_w.sum()
    wm = np.sum(pos_w * pos_vals) / pos_w.sum()
    return wm


# ---------------------------
#  Build horizon predictions [time=67..78 x sectors]
# ---------------------------
def predict_horizon(a_tr, alpha, n_lags, t2, allow_zeros):
    idx = np.arange(67, 79)
    cols = a_tr.columns
    a_pred = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for sector in cols:
        if (a_tr.tail(t2)[sector] == 0).mean() > allow_zeros / t2 + 1e-8 or (a_tr[sector].sum() == 0):
            a_pred[sector] = 0.0
            continue
        base_last_value = a_tr[sector].iloc[-1]
        base_ewgm = ewgm_per_sector(a_tr=a_tr, sector=sector, n_lags=n_lags, alpha=alpha)
        
        a_pred[sector] = 0.34*base_last_value + 0.34*base_ewgm
        # a_pred[sector] = base_ewgm
        
    a_pred.index.rename('time', inplace=True)
    return a_pred


# ---------------------------
#  Convert wide predictions into submission aligned with test ids
# ---------------------------
def build_submission_df(a_pred, test_raw, month_codes):
    test = split_test_id_column(test_raw.copy())
    test = add_time_and_sector_fields(test, month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    out = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    return out


# ---------------------------
#  End-to-end generation with December bump
# ---------------------------
def generate_submission_with_december_bump(alpha=0.5, n_lags=6, t2=6, allow_zeros=0, clip_low=0.85, clip_high=1.4):
    month_codes = build_month_codes()
    train_nht, test = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    a_pred = predict_horizon(a_tr=a_tr, alpha=alpha, n_lags=n_lags, t2=t2, allow_zeros=allow_zeros)
    sector_to_mult = compute_december_multipliers(a_tr=a_tr, eps=1e-9, min_dec_obs=1, clip_low=clip_low, clip_high=clip_high)
    a_pred = apply_december_bump(a_pred=a_pred, sector_to_mult=sector_to_mult)
    for sector in a_pred.columns:
        a_pred[sector] += 0.32*month[sector-1]
    submission = build_submission_df(a_pred=a_pred, test_raw=test, month_codes=month_codes)
    return a_tr, a_pred, submission
a_tr, a_pred, submission = generate_submission_with_december_bump(
    alpha=0.5,
    n_lags=12,
    t2=10,
    allow_zeros=2, 
    clip_low=0.85,
    clip_high=1.4
)

print('Submission with December seasonality saved to /kaggle/working/submission.csv')
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission
Submission with December seasonality saved to /kaggle/working/submission.csv
id	new_house_transaction_amount
0	2024 Aug_sector 1	10732.788632
1	2024 Aug_sector 2	3791.517824
2	2024 Aug_sector 3	8138.380720
3	2024 Aug_sector 4	81771.761074
4	2024 Aug_sector 5	2571.213703
...	...	...
1147	2025 Jul_sector 92	27689.824431
1148	2025 Jul_sector 93	18571.671145
1149	2025 Jul_sector 94	14115.925455
1150	2025 Jul_sector 95	0.000000
1151	2025 Jul_sector 96	122.466664
1152 rows × 2 columns

 

 # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv
/kaggle/input/china-real-estate-demand-prediction/test.csv
/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv
/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv
/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv
/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv
/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv
/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv
# --- Imports ---
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesRegressor

# --- Paths ---
pth = "/kaggle/input/china-real-estate-demand-prediction"

# --- Load data ---
ci = (pl.read_csv(f"{pth}/train/city_indexes.csv")
        .head(6).fill_null(-1)
        .drop("total_fixed_asset_investment_10k")
        .rename(lambda c: ("" if c in ["sector","month"] else "ci_") + c))

csi = pl.read_csv(f"{pth}/train/city_search_index.csv")  # optional

sp = (pl.read_csv(f"{pth}/train/sector_POI.csv")
        .fill_null(-1)
        .rename(lambda c: ("" if c in ["sector","month"] else "sp_") + c))

train_lt   = pl.read_csv(f"{pth}/train/land_transactions.csv", infer_schema_length=10000
             ).rename(lambda c: ("" if c in ["sector","month"] else "lt_") + c)

train_ltns = pl.read_csv(f"{pth}/train/land_transactions_nearby_sectors.csv"
             ).rename(lambda c: ("" if c in ["sector","month"] else "ltns_") + c)

train_pht  = pl.read_csv(f"{pth}/train/pre_owned_house_transactions.csv"
             ).rename(lambda c: ("" if c in ["sector","month"] else "pht_") + c)

train_phtns = pl.read_csv(f"{pth}/train/pre_owned_house_transactions_nearby_sectors.csv"
              ).rename(lambda c: ("" if c in ["sector","month"] else "phtns_") + c)

train_nht  = pl.read_csv(f"{pth}/train/new_house_transactions.csv"
             ).rename(lambda c: ("" if c in ["sector","month"] else "nht_") + c)

train_nhtns = pl.read_csv(f"{pth}/train/new_house_transactions_nearby_sectors.csv"
              ).rename(lambda c: ("" if c in ["sector","month"] else "nhtns_") + c)

test = pl.read_csv(f"{pth}/test.csv").with_columns(
    id_=pl.col("id").str.split("_")
).with_columns(
    month=pl.col("id_").list.get(0),
    sector=pl.col("id_").list.get(1)
).drop("id_")

month_codes = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
}

# --- Join & feature prep ---
data = (pl.DataFrame(train_nht["month"].unique())
    .join(pl.DataFrame(train_nht["sector"].unique().to_list()+["sector 95"]).rename({"column_0":"sector"}), how="cross")
    .with_columns(
        sector_id=pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8),
        year=pl.col("month").str.split("-").list.get(0).cast(pl.Int16),
        month_num=pl.col("month").str.split("-").list.get(1).replace(month_codes).cast(pl.Int8),
    )
    .with_columns(time=((pl.col("year")-2019)*12 + pl.col("month_num") - 1).cast(pl.Int8))
    .sort("sector_id","time")
    .join(train_nht,   on=["sector","month"], how="left").fill_null(0)
    .join(train_nhtns, on=["sector","month"], how="left").fill_null(-1)
    .join(train_pht,   on=["sector","month"], how="left").fill_null(-1)
    .join(train_phtns, on=["sector","month"], how="left").fill_null(-1)
    .join(ci.rename({"ci_city_indicator_data_year":"year"}), on=["year"], how="left").fill_null(-1)
    .join(sp, on=["sector"], how="left").fill_null(-1)
    .join(train_lt,   on=["sector","month"], how="left").fill_null(-1)
    .join(train_ltns, on=["sector","month"], how="left").fill_null(-1)
    .with_columns(cs.float().cast(pl.Float32))
)

# downcast ints, drop all-zero int columns
for col in data.columns:
    if data[col].dtype == pl.Int64:
        cmin, cmax = data[col].min(), data[col].max()
        if cmin == 0 and cmax == 0:
            data = data.drop(col)
        elif cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
            data = data.with_columns(pl.col(col).cast(pl.Int8))
        elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
            data = data.with_columns(pl.col(col).cast(pl.Int16))
        elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
            data = data.with_columns(pl.col(col).cast(pl.Int32))

data = data.drop("month","sector","year")

# simple temporal joins (as in your script)
data2 = data.sort("time","sector_id")
for m in [1,2,12]:
    data2 = data2.join(
        data.drop("month_num").with_columns(pl.col("time")+m),
        on=["sector_id","time"], how="left", suffix=f"_{m}"
    )
data2 = data2.sort("time","sector_id")

# label & seasonality
lag = -1
data3 = (data2.with_columns(
            pl.col("nht_amount_new_house_transactions").shift(lag).over("sector_id").alias("label"),
            cs=((pl.col("month_num")-1)/6*np.pi).cos(),
            sn=((pl.col("month_num")-1)/6*np.pi).sin(),
            cs6=((pl.col("month_num")-1)/3*np.pi).cos(),
            sn6=((pl.col("month_num")-1)/3*np.pi).sin(),
            cs3=((pl.col("month_num")-1)/1.5*np.pi).cos(),
            sn3=((pl.col("month_num")-1)/1.5*np.pi).sin(),
         )
        # .drop_nulls(subset=["label"])  # keep NA for CatBoost handling via fill
)
data3 = data3.drop("sector_id")  # matches your original

# --- Splits / pools ---
cat_features = ["month_num"]
border  = 66+lag-1
border1 = 6*3

train_mask = (pl.col("time")<=border) & (pl.col("time")>border1)
val_mask   = (pl.col("time")>border) & (pl.col("time")<=66+lag)
t66_mask   = (pl.col("time")==66)

# CatBoost pools
trainPool = Pool(
    data = data3.filter(train_mask).drop(["label"]).to_pandas().fillna(-2),
    label= data3.filter(train_mask)["label"].to_pandas(),
    cat_features=cat_features
)
testPool = Pool(
    data = data3.filter(val_mask).drop(["label"]).to_pandas().fillna(-2),
    label= data3.filter(val_mask)["label"].to_pandas(),
    cat_features=cat_features
)
testPool2 = Pool(  # time==66
    data = data3.filter(t66_mask).drop(["label"]).to_pandas().fillna(-2),
    cat_features=cat_features
)

# Pandas frames for ExtraTrees + stacking
X_train_df = data3.filter(train_mask).drop(["label"]).to_pandas().fillna(-2)
y_train    = data3.filter(train_mask)["label"].to_pandas().values.ravel()

X_val_df   = data3.filter(val_mask).drop(["label"]).to_pandas().fillna(-2)
y_val      = data3.filter(val_mask)["label"].to_pandas().values.ravel()

X_t66_df   = data3.filter(t66_mask).drop(["label"]).to_pandas().fillna(-2)

# --- Custom metric for blending selection ---
def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0: return 0.0
    ape = np.abs((y_true - np.maximum(y_pred,0)) / np.maximum(y_true, eps))
    bad_rate = np.mean(ape > 1.0)
    if bad_rate > 0.30: return 0.0
    good_ape = ape[ape <= 1.0]
    if good_ape.size == 0: return 0.0
    mape = np.mean(good_ape)
    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    return max(0.0, 1.0 - scaled_mape)

class CustomMetric:
    def is_max_optimal(self): return True
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        score = custom_score(target, approxes[0])
        return score, 1
    def get_final_error(self, error, weight): return error

class CustomObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        result = []
        for i in range(len(targets)):
            # non-convex custom: amplify when underpredicting heavily
            der1 = np.sign(targets[i] - approxes[i]) if (2*targets[i] - approxes[i]) < 0 else np.sign(targets[i] - approxes[i]) * 5
            result.append((der1, 0.0))
        return result

# --- Train CatBoost ---
cb = CatBoostRegressor(
    iterations=21000,
    learning_rate=0.0125,
    one_hot_max_size=256,
    custom_metric=["RMSE","MAPE","SMAPE","MAE"],
    loss_function=CustomObjective(),
    eval_metric=CustomMetric(),
    l2_leaf_reg=0.3,
    random_seed=4,
    verbose=1000
)
cb.fit(trainPool, eval_set=testPool, verbose=1000)

# --- Train ExtraTrees (stack component) ---
et = ExtraTreesRegressor(
    n_estimators=800,
    n_jobs=-1,
    random_state=42
)
et.fit(X_train_df, y_train)

# --- Blend weight tuned on validation by custom_score ---
cb_val = cb.predict(testPool)
et_val = et.predict(X_val_df)

best_alpha, best_score = 0.5, -1.0
for a in np.linspace(0.0, 1.0, 21):
    blend_val = a*cb_val + (1.0-a)*et_val
    sc = custom_score(y_val, blend_val)
    if sc > best_score:
        best_score, best_alpha = sc, a

print(f"[Blend] best_alpha={best_alpha:.2f} | val_score={best_score:.6f} | "
      f"cat_val={custom_score(y_val, cb_val):.6f} | et_val={custom_score(y_val, et_val):.6f}")

# --- Final predictions for time==66 (BLENDED) ---
cb_t66 = cb.predict(testPool2)
et_t66 = et.predict(X_t66_df)
month = np.maximum(best_alpha*cb_t66 + (1.0-best_alpha)*et_t66, 0.0)

# Safety: ensure not identical to CatBoost-only (prevents accidental overwrite)
assert not np.allclose(month, cb_t66), "Stacked blend was not used; 'month' equals CatBoost-only."

# --- Index pruning (as in your script) ---
for i in [11,38,40,43,48,51,52,57,71,72,73,74,81,86,88,94,95]:
    month[i] = 0

# --- Write submission ---
sub = pd.read_csv(f"{pth}/sample_submission.csv")
for m in range(12):
    sub.loc[[i + m*96 for i in range(96)], "new_house_transaction_amount"] = month
sub.to_csv("submission.csv", index=False)

print("submission.csv written with CatBoost + ExtraTrees blended predictions.")
/usr/local/lib/python3.11/dist-packages/catboost/core.py:1790: UserWarning: Failed to optimize method "evaluate" in the passed object:
Failed in nopython mode pipeline (step: nopython frontend)
Untyped global name 'custom_score': Cannot determine Numba type of <class 'function'>

File "../../tmp/ipykernel_13/1446721793.py", line 168:
<source missing, REPL/exec in use?>

  self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
0:	learn: 0.0000000	test: 0.0000000	best: 0.0000000 (0)	total: 1.63s	remaining: 9h 29m 14s
1000:	learn: 0.5039899	test: 0.4830896	best: 0.4977141 (988)	total: 47.7s	remaining: 15m 53s
2000:	learn: 0.5402593	test: 0.5275462	best: 0.5275462 (2000)	total: 1m 34s	remaining: 14m 55s
3000:	learn: 0.5642219	test: 0.5177333	best: 0.5356251 (2409)	total: 2m 20s	remaining: 14m 5s
4000:	learn: 0.5827448	test: 0.5189696	best: 0.5356251 (2409)	total: 3m 7s	remaining: 13m 16s
5000:	learn: 0.5969942	test: 0.5484097	best: 0.5659240 (4246)	total: 3m 54s	remaining: 12m 30s
6000:	learn: 0.6096808	test: 0.5853091	best: 0.5894812 (5927)	total: 4m 41s	remaining: 11m 44s
7000:	learn: 0.6169048	test: 0.5676445	best: 0.5997168 (6075)	total: 5m 27s	remaining: 10m 54s
8000:	learn: 0.6294979	test: 0.5987088	best: 0.6011210 (7940)	total: 6m 13s	remaining: 10m 6s
9000:	learn: 0.6427181	test: 0.6045021	best: 0.6045361 (8992)	total: 6m 59s	remaining: 9m 19s
10000:	learn: 0.6525037	test: 0.6031792	best: 0.6071935 (9645)	total: 7m 43s	remaining: 8m 30s
11000:	learn: 0.6644573	test: 0.6093246	best: 0.6110193 (10821)	total: 8m 28s	remaining: 7m 42s
12000:	learn: 0.6733714	test: 0.6063326	best: 0.6126249 (11394)	total: 9m 14s	remaining: 6m 55s
13000:	learn: 0.6835417	test: 0.6065582	best: 0.6126249 (11394)	total: 10m	remaining: 6m 9s
14000:	learn: 0.6909731	test: 0.6012277	best: 0.6177883 (13556)	total: 10m 50s	remaining: 5m 25s
15000:	learn: 0.6976927	test: 0.6035575	best: 0.6177883 (13556)	total: 11m 37s	remaining: 4m 39s
16000:	learn: 0.7090310	test: 0.6102371	best: 0.6177883 (13556)	total: 12m 28s	remaining: 3m 53s
17000:	learn: 0.7171831	test: 0.6109518	best: 0.6177883 (13556)	total: 13m 15s	remaining: 3m 7s
18000:	learn: 0.7272085	test: 0.6091775	best: 0.6177883 (13556)	total: 14m	remaining: 2m 20s
19000:	learn: 0.7349961	test: 0.6076191	best: 0.6177883 (13556)	total: 14m 46s	remaining: 1m 33s
20000:	learn: 0.7420185	test: 0.6092236	best: 0.6214477 (19232)	total: 15m 32s	remaining: 46.6s
20999:	learn: 0.7508729	test: 0.6117854	best: 0.6226570 (20754)	total: 16m 18s	remaining: 0us

bestTest = 0.622656999
bestIteration = 20754

Shrink model to first 20755 iterations.
[Blend] best_alpha=1.00 | val_score=0.622657 | cat_val=0.622657 | et_val=0.000000
submission.csv written with CatBoost + ExtraTrees blended predictions.
/tmp/ipykernel_13/1446721793.py:233: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '[ 13741.58299206   4975.70820976   8415.33312434  80157.03967594
   4184.58322941  15279.70795386   7000.04149282   2926.45826065
  10594.04140356  33328.91583893   4430.62488996      0.
   2852.49992916  20881.99948139  91772.08105415  66324.79001947
   1452.58329726  20928.91614689   3424.91658161   8193.08312986
  23616.16608015  19755.16617604  26764.74933529  65681.58170211
  18385.33287673   3855.08323759   4014.3749003   34869.624134
  24827.29105007  33333.0825055    6236.58317845  38780.20737022
  12008.49970177  77567.08140693   9110.33310708 104704.58073297
  12638.99968611  39261.04069161      0.           9789.08309022
      0.           3480.95824688  12921.1246791       0.
  68771.74829203   5296.04153514  21011.5828115   24827.95771672
      0.          17515.91623165  14863.41629753      0.
      0.          48219.33213579  94728.66431405   3375.37491617
  23968.6660714       0.          21076.04114324   9630.66642749
  64672.49839384  13201.29133881   7515.70814668  16352.9579272
   9828.6247559   51457.99872203   1665.24995864  16801.87458272
   3584.70824431   1992.12495053   8442.58312366      0.
      0.              0.              0.          27066.70766112
  53863.16532896  16730.66625116  37636.79073195  21094.16614279
  14667.2913024       0.          53024.83201645   7021.24982563
 111523.16389696  40939.29064993      0.           3147.24992184
      0.           5207.16653735  30915.12423221  21804.04112516
  14802.9579657   13634.62466138      0.              0.        ]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  sub.loc[[i + m*96 for i in range(96)], "new_house_transaction_amount"] = month


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv
/kaggle/input/china-real-estate-demand-prediction/test.csv
/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv
/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv
/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv
/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv
/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv
/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv
/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv
import polars as pl
import polars.selectors as cs

from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv

import numpy as np
#import matplotlib.pyplot as plt
#from colorama import Fore, Style

from sklearn.model_selection import TimeSeriesSplit
%%time
pth = "/kaggle/input/china-real-estate-demand-prediction"

ci = pl.read_csv(pth+'/train/city_indexes.csv'
                ).head(6).fill_null(-1).drop("total_fixed_asset_investment_10k"
                                            ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "ci_") + column_name) # one row per year

csi = pl.read_csv(pth+'/train/city_search_index.csv') # several rows per training month

sp = pl.read_csv(pth+'/train/sector_POI.csv'
                ).fill_null(-1).rename(lambda column_name:("" if column_name in ["sector","month"] else  "sp_") + column_name) # at most one row per sector

train_lt = pl.read_csv(pth+'/train/land_transactions.csv', infer_schema_length=10000
                      ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "lt_") + column_name)

train_ltns = pl.read_csv(pth+'/train/land_transactions_nearby_sectors.csv'
                        ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "ltns_") + column_name)

train_pht = pl.read_csv(pth+'/train/pre_owned_house_transactions.csv'
                       ).rename(lambda column_name:("" if column_name in ["sector","month"] else  "pht_") + column_name)

train_phtns = pl.read_csv(pth+'/train/pre_owned_house_transactions_nearby_sectors.csv'
                         ).rename(lambda column_name: ("" if column_name in ["sector","month"] else "phtns_") + column_name)

train_nht = pl.read_csv(pth+'/train/new_house_transactions.csv'
                       ).rename(lambda column_name: ("" if column_name in ["sector","month"] else "nht_") + column_name)

train_nhtns = pl.read_csv(pth+'/train/new_house_transactions_nearby_sectors.csv'
                         ).rename(lambda column_name: ("" if column_name in ["sector","month"] else "nhtns_") + column_name)

test = pl.read_csv(pth+'/test.csv')

test = test.with_columns(id_=pl.col("id").str.split("_")
          ).with_columns(month=pl.col("id_").list.get(0),
                         sector=pl.col("id_").list.get(1)
          ).drop("id_")
test 
train_nhtns
train_nht
CPU times: user 129 ms, sys: 73.6 ms, total: 203 ms
Wall time: 375 ms
shape: (5_433, 11)
month	sector	nht_num_new_house_transactions	nht_area_new_house_transactions	nht_price_new_house_transactions	nht_amount_new_house_transactions	nht_area_per_unit_new_house_transactions	nht_total_price_per_unit_new_house_transactions	nht_num_new_house_available_for_sale	nht_area_new_house_available_for_sale	nht_period_new_house_sell_through
str	str	i64	i64	i64	f64	i64	f64	i64	i64	f64
"2019-Jan"	"sector 1"	52	4906	28184	13827.14	94	265.91	159	15904	3.78
"2019-Jan"	"sector 2"	145	15933	17747	28277.73	110	195.02	1491	175113	12.29
"2019-Jan"	"sector 4"	6	725	28004	1424.21	127	356.05	40	6826	5.95
"2019-Jan"	"sector 5"	2	212	37432	792.1	106	396.05	161	17173	83.95
"2019-Jan"	"sector 6"	5	773	15992	607.94	95	151.99	189	19696	14.27
…	…	…	…	…	…	…	…	…	…	…
"2024-Jul"	"sector 91"	70	7921	40967	32450.06	113	463.57	2133	341192	51.82
"2024-Jul"	"sector 92"	211	22084	13949	30804.74	105	145.99	5908	636696	34.76
"2024-Jul"	"sector 93"	62	8136	27452	22335.3	131	360.25	1323	150862	27.74
"2024-Jul"	"sector 94"	44	5078	26367	13389.41	115	304.3	2027	215821	38.62
"2024-Jul"	"sector 96"	1	140	40079	561.19	140	561.19	1	195	1.39
month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}
%%time
data = pl.DataFrame( train_nht["month"].unique() 
            ).join(pl.DataFrame( train_nht["sector"].unique().to_list()+["sector 95"] 
                               ).rename({"column_0":"sector"}), 
                   how="cross" 
            ).with_columns(sector_id=pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8),
                      year=pl.col("month").str.split("-").list.get(0).cast(pl.Int16),
                      month_num=pl.col("month").str.split("-").list.get(1).replace(month_codes).cast(pl.Int8),
            ).with_columns(
                      time=( (pl.col("year") - 2019) * 12 + pl.col("month_num") - 1 ).cast(pl.Int8)
            ).sort("sector_id","time"
                   
            ).join(train_nht, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(0
            ).join(train_nhtns, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).join(train_pht, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).join(train_phtns, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
                       
            ).join(ci.rename({"ci_city_indicator_data_year":"year"}), 
                   on=["year"],
                   how="left"
            ).fill_null(-1
            ).join(sp, 
                   on=["sector"],
                   how="left"
            ).fill_null(-1
                        
            ).join(train_lt, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).join(train_ltns, 
                   on=["sector","month"],
                   how="left"
            ).fill_null(-1
            ).with_columns(cs.float().cast(pl.Float32)
            )
data.max()
CPU times: user 94.4 ms, sys: 82.6 ms, total: 177 ms
Wall time: 136 ms
shape: (1, 253)
month	sector	sector_id	year	month_num	time	nht_num_new_house_transactions	nht_area_new_house_transactions	nht_price_new_house_transactions	nht_amount_new_house_transactions	nht_area_per_unit_new_house_transactions	nht_total_price_per_unit_new_house_transactions	nht_num_new_house_available_for_sale	nht_area_new_house_available_for_sale	nht_period_new_house_sell_through	nhtns_num_new_house_transactions_nearby_sectors	nhtns_area_new_house_transactions_nearby_sectors	nhtns_price_new_house_transactions_nearby_sectors	nhtns_amount_new_house_transactions_nearby_sectors	nhtns_area_per_unit_new_house_transactions_nearby_sectors	nhtns_total_price_per_unit_new_house_transactions_nearby_sectors	nhtns_num_new_house_available_for_sale_nearby_sectors	nhtns_area_new_house_available_for_sale_nearby_sectors	nhtns_period_new_house_sell_through_nearby_sectors	pht_area_pre_owned_house_transactions	pht_amount_pre_owned_house_transactions	pht_num_pre_owned_house_transactions	pht_price_pre_owned_house_transactions	phtns_num_pre_owned_house_transactions_nearby_sectors	phtns_area_pre_owned_house_transactions_nearby_sectors	phtns_amount_pre_owned_house_transactions_nearby_sectors	phtns_price_pre_owned_house_transactions_nearby_sectors	ci_year_end_registered_population_10k	ci_total_households_10k	ci_year_end_resident_population_10k	ci_year_end_total_employed_population_10k	ci_year_end_urban_non_private_employees_10k	…	sp_community_winner_malls_dense	sp_key_focus_malls_dense	sp_transportation_facilities_service_bus_station_dense	sp_transportation_facilities_service_subway_station_dense	sp_transportation_facilities_service_airport_related_dense	sp_transportation_facilities_service_port_terminal_dense	sp_transportation_facilities_service_train_station_dense	sp_transportation_facilities_service_light_rail_station_dense	sp_transportation_facilities_service_long_distance_bus_station_dense	sp_leisure_entertainment_entertainment_venue_game_arcade_dense	sp_leisure_entertainment_entertainment_venue_party_house_dense	sp_leisure_entertainment_cultural_venue_cultural_palace_dense	sp_office_building_industrial_building_industrial_building_dense	sp_medical_health_dense	sp_medical_health_specialty_hospital_dense	sp_medical_health_tcm_hospital_dense	sp_medical_health_physical_examination_institution_dense	sp_medical_health_veterinary_station_dense	sp_medical_health_pharmaceutical_healthcare_dense	sp_medical_health_rehabilitation_institution_dense	sp_medical_health_first_aid_center_dense	sp_medical_health_blood_donation_station_dense	sp_medical_health_disease_prevention_institution_dense	sp_medical_health_general_hospital_dense	sp_medical_health_clinic_dense	sp_education_training_school_education_middle_school_dense	sp_education_training_school_education_primary_school_dense	sp_education_training_school_education_kindergarten_dense	sp_education_training_school_education_research_institution_dense	lt_num_land_transactions	lt_construction_area	lt_planned_building_area	lt_transaction_amount	ltns_num_land_transactions_nearby_sectors	ltns_construction_area_nearby_sectors	ltns_planned_building_area_nearby_sectors	ltns_transaction_amount_nearby_sectors
str	str	i8	i16	i8	i8	i64	i64	i64	f32	i64	f32	i64	i64	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i64	f32	i64	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	…	f32	i64	f32	f32	f32	f32	f32	i64	f32	f32	f32	f32	i64	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i64	f32	f32	f32	f32	f32	f32	f32
"2024-May"	"sector 96"	96	2024	12	66	2669	294430	208288	606407.625	2003	7803.600098	12048	1220617	274.26001	990.200012	101748.75	107817.835938	315836.8125	286.662506	2231.21875	6158.0	631131.75	100.26667	126073	224737.0	1277	149937.15625	440.5	43329.0	97268.570312	74599.570312	1034.910034	336.193512	1881.060059	1163.439941	426.940002	…	0.000004	0	0.001685	0.00068	7.0400e-8	0.000232	0.000575	0	0.000127	0.000291	0.000209	0.000116	0	0.008659	0.000814	0.000042	0.000046	0.000092	0.005056	0.001453	0.000012	0.000031	0.000094	0.000232	0.000872	0.000467	0.000697	0.001162	0.000936	5	465071.0625	1.715928e6	1.876041e6	2.6	155813.40625	571976.0	504823.1875
#Do you have any ideas how to use csi?
csi
shape: (4_020, 4)
month	keyword	source	search_volume
str	str	str	i64
"2019-Jan"	"买房"	"PC端"	1914
"2019-Jan"	"买房"	"移动端"	2646
"2019-Jan"	"二手房市场"	"PC端"	192
"2019-Jan"	"二手房市场"	"移动端"	204
"2019-Jan"	"公积金"	"PC端"	9160
…	…	…	…
"2024-Jul"	"限售"	"移动端"	40
"2024-Jul"	"限购"	"PC端"	203
"2024-Jul"	"限购"	"移动端"	155
"2024-Jul"	"首付"	"PC端"	739
"2024-Jul"	"首付"	"移动端"	1471
%%time
for col in data.columns:
    if data[col].dtype==pl.Int64:
        c_min = data[col].min()
        c_max = data[col].max()
        if c_min == 0 and c_max == 0:
            data = data.drop(col)
            print(col, "0"*20 )
        elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            data = data.with_columns( pl.col(col).cast(pl.Int8) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            data = data.with_columns( pl.col(col).cast(pl.Int16) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            data = data.with_columns( pl.col(col).cast(pl.Int32) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            data = data.with_columns( pl.col(col).cast(pl.Int64) )
            print(col, data[col].dtype, data[col].min(), data[col].max() )
            
data = data.drop("month","sector","year")            
data
nht_num_new_house_transactions Int16 0 2669
nht_area_new_house_transactions Int32 0 294430
nht_price_new_house_transactions Int32 0 208288
nht_area_per_unit_new_house_transactions Int16 0 2003
nht_num_new_house_available_for_sale Int16 0 12048
nht_area_new_house_available_for_sale Int32 0 1220617
pht_area_pre_owned_house_transactions Int32 -1 126073
pht_num_pre_owned_house_transactions Int16 -1 1277
ci_national_year_end_total_population_10k Int32 -1 141260
ci_gdp_per_capita_yuan Int32 -1 156427
ci_annual_average_wage_urban_non_private_employees_yuan Int32 -1 147947
ci_annual_average_wage_urban_non_private_on_duty_employees_yuan Int32 -1 152324
ci_number_of_universities Int8 -1 84
ci_number_of_middle_schools Int16 -1 555
ci_number_of_primary_schools Int16 -1 992
ci_number_of_kindergartens Int16 -1 2223
ci_hospitals_health_centers Int16 -1 6159
ci_number_of_operating_bus_lines Int8 -1 18
ci_operating_bus_line_length_km Int16 -1 643
ci_number_of_industrial_enterprises_above_designated_size Int16 -1 6878
ci_total_current_assets_10k Int32 -1 159482630
ci_total_fixed_assets_10k Int32 -1 39934058
ci_main_business_taxes_and_surcharges_10k Int32 -1 4862051
ci_real_estate_development_investment_completed_10k Int32 -1 31022573
ci_residential_development_investment_completed_10k Int32 -1 20870742
ci_science_expenditure_10k Int32 -1 2439456
ci_education_expenditure_10k Int32 -1 6269391
sp_population_scale Int32 -1 31077700
sp_residential_area Int16 -1 24964
sp_office_building Int16 -1 13187
sp_commercial_area Int16 -1 3681
sp_resident_population Int32 -1 17099643
sp_office_population Int32 -1 26152000
sp_number_of_shops Int32 -1 1267755
sp_catering Int32 -1 350067
sp_retail Int32 -1 776801
sp_hotel Int32 -1 64186
sp_transportation_station Int32 -1 75064
sp_education Int16 -1 24366
sp_leisure_and_entertainment Int32 -1 52882
sp_bus_station_cnt Int16 -1 17281
sp_subway_station_cnt Int16 -1 418
sp_rentable_shops Int32 -1 43378
sp_leisure_entertainment_entertainment_venue_game_arcade Int16 -1 453
sp_leisure_entertainment_entertainment_venue_party_house Int16 -1 558
sp_leisure_entertainment_cultural_venue_cultural_palace Int16 -1 186
sp_office_building_industrial_building_industrial_building Int8 -1 0
sp_education_training_school_education_middle_school Int16 -1 867
sp_education_training_school_education_primary_school Int16 -1 1289
sp_education_training_school_education_kindergarten Int16 -1 4189
sp_education_training_school_education_research_institution Int16 -1 1302
sp_medical_health Int32 -1 55785
sp_medical_health_specialty_hospital Int16 -1 3430
sp_medical_health_tcm_hospital Int8 -1 63
sp_medical_health_physical_examination_institution Int8 -1 123
sp_medical_health_veterinary_station Int8 -1 124
sp_medical_health_pharmaceutical_healthcare Int32 -1 34583
sp_medical_health_rehabilitation_institution Int16 -1 5249
sp_medical_health_first_aid_center Int8 -1 120
sp_medical_health_blood_donation_station Int16 -1 186
sp_medical_health_disease_prevention_institution Int16 -1 422
sp_medical_health_general_hospital Int16 -1 2613
sp_medical_health_clinic Int16 -1 9248
sp_transportation_facilities_service_bus_station Int16 -1 15291
sp_transportation_facilities_service_subway_station Int16 -1 2377
sp_transportation_facilities_service_airport_related Int8 -1 4
sp_transportation_facilities_service_port_terminal Int16 -1 240
sp_transportation_facilities_service_train_station Int16 -1 589
sp_transportation_facilities_service_light_rail_station Int8 -1 0
sp_transportation_facilities_service_long_distance_bus_station Int16 -1 308
sp_number_of_leisure_and_entertainment_stores Int8 -1 15
sp_number_of_other_stores Int8 -1 27
sp_number_of_other_anchor_stores Int8 -1 9
sp_number_of_home_appliance_stores Int8 -1 113
sp_number_of_skincare_cosmetics_stores Int8 -1 106
sp_number_of_fashion_stores Int16 -1 622
sp_number_of_service_stores Int8 -1 67
sp_number_of_jewelry_stores Int8 -1 109
sp_number_of_lifestyle_leisure_stores Int8 -1 2
sp_number_of_supermarket_convenience_stores Int8 -1 69
sp_number_of_catering_food_stores Int16 -1 238
sp_number_of_residential_commercial Int8 -1 5
sp_number_of_office_building_commercial Int8 -1 5
sp_number_of_commercial_buildings Int8 -1 8
sp_number_of_hypermarkets Int8 -1 3
sp_number_of_department_stores Int8 -1 4
sp_number_of_shopping_centers Int8 -1 13
sp_number_of_hotel_commercial Int8 -1 1
sp_number_of_third_tier_shopping_malls_in_business_district Int8 -1 1
sp_number_of_second_tier_shopping_malls_in_business_district Int8 -1 1
sp_number_of_city_winner_malls Int8 -1 3
sp_number_of_shopping_malls_with_street_facing_shops Int8 -1 0
sp_number_of_unranked_malls Int8 -1 27
sp_number_of_community_malls Int8 -1 3
sp_number_of_community_winner_malls Int8 -1 3
sp_number_of_key_focus_malls Int8 -1 0
sp_shopping_malls_with_street_facing_shops_dense Int8 -1 0
sp_key_focus_malls_dense Int8 -1 0
sp_transportation_facilities_service_light_rail_station_dense Int8 -1 0
sp_office_building_industrial_building_industrial_building_dense Int8 -1 0
lt_num_land_transactions Int8 -1 5
CPU times: user 25.3 ms, sys: 6.01 ms, total: 31.3 ms
Wall time: 32.3 ms
shape: (6_432, 250)
sector_id	month_num	time	nht_num_new_house_transactions	nht_area_new_house_transactions	nht_price_new_house_transactions	nht_amount_new_house_transactions	nht_area_per_unit_new_house_transactions	nht_total_price_per_unit_new_house_transactions	nht_num_new_house_available_for_sale	nht_area_new_house_available_for_sale	nht_period_new_house_sell_through	nhtns_num_new_house_transactions_nearby_sectors	nhtns_area_new_house_transactions_nearby_sectors	nhtns_price_new_house_transactions_nearby_sectors	nhtns_amount_new_house_transactions_nearby_sectors	nhtns_area_per_unit_new_house_transactions_nearby_sectors	nhtns_total_price_per_unit_new_house_transactions_nearby_sectors	nhtns_num_new_house_available_for_sale_nearby_sectors	nhtns_area_new_house_available_for_sale_nearby_sectors	nhtns_period_new_house_sell_through_nearby_sectors	pht_area_pre_owned_house_transactions	pht_amount_pre_owned_house_transactions	pht_num_pre_owned_house_transactions	pht_price_pre_owned_house_transactions	phtns_num_pre_owned_house_transactions_nearby_sectors	phtns_area_pre_owned_house_transactions_nearby_sectors	phtns_amount_pre_owned_house_transactions_nearby_sectors	phtns_price_pre_owned_house_transactions_nearby_sectors	ci_year_end_registered_population_10k	ci_total_households_10k	ci_year_end_resident_population_10k	ci_year_end_total_employed_population_10k	ci_year_end_urban_non_private_employees_10k	ci_private_individual_and_other_employees_10k	ci_private_individual_ratio	ci_national_year_end_total_population_10k	…	sp_community_winner_malls_dense	sp_key_focus_malls_dense	sp_transportation_facilities_service_bus_station_dense	sp_transportation_facilities_service_subway_station_dense	sp_transportation_facilities_service_airport_related_dense	sp_transportation_facilities_service_port_terminal_dense	sp_transportation_facilities_service_train_station_dense	sp_transportation_facilities_service_light_rail_station_dense	sp_transportation_facilities_service_long_distance_bus_station_dense	sp_leisure_entertainment_entertainment_venue_game_arcade_dense	sp_leisure_entertainment_entertainment_venue_party_house_dense	sp_leisure_entertainment_cultural_venue_cultural_palace_dense	sp_office_building_industrial_building_industrial_building_dense	sp_medical_health_dense	sp_medical_health_specialty_hospital_dense	sp_medical_health_tcm_hospital_dense	sp_medical_health_physical_examination_institution_dense	sp_medical_health_veterinary_station_dense	sp_medical_health_pharmaceutical_healthcare_dense	sp_medical_health_rehabilitation_institution_dense	sp_medical_health_first_aid_center_dense	sp_medical_health_blood_donation_station_dense	sp_medical_health_disease_prevention_institution_dense	sp_medical_health_general_hospital_dense	sp_medical_health_clinic_dense	sp_education_training_school_education_middle_school_dense	sp_education_training_school_education_primary_school_dense	sp_education_training_school_education_kindergarten_dense	sp_education_training_school_education_research_institution_dense	lt_num_land_transactions	lt_construction_area	lt_planned_building_area	lt_transaction_amount	ltns_num_land_transactions_nearby_sectors	ltns_construction_area_nearby_sectors	ltns_planned_building_area_nearby_sectors	ltns_transaction_amount_nearby_sectors
i8	i8	i8	i16	i32	i32	f32	i16	f32	i16	i32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i32	f32	i16	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i32	…	f32	i8	f32	f32	f32	f32	f32	i8	f32	f32	f32	f32	i8	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i8	f32	f32	f32	f32	f32	f32	f32
1	1	0	52	4906	28184	13827.139648	94	265.910004	159	15904	3.78	29.444445	3532.444336	51992.519531	18366.068359	119.96981	623.753296	350.25	49809.875	29.696251	9163	40994.699219	111	44739.386719	6.75	733.0	1247.037964	17012.796875	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	0.0	0	0.000243	0.000095	0.0	0.0	0.0	0	0.0	0.000027	0.000015	0.000014	0	0.000563	0.000032	0.0	0.0	0.0	0.000339	0.000113	0.0	0.0	8.6000e-7	0.000041	0.000038	0.000015	0.000028	0.000063	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	2	1	24	2526	34846	8802.80957	105	366.779999	158	15814	4.24	13.0	1597.333374	45760.832031	7309.529785	122.871796	562.271545	361.0	45425.0	14.716666	3191	10191.0	50	31936.697266	6.25	709.0	1315.150024	18549.365234	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	0.0	0	0.000243	0.000095	0.0	0.0	0.0	0	0.0	0.000027	0.000015	0.000014	0	0.000563	0.000032	0.0	0.0	0.0	0.000339	0.000113	0.0	0.0	8.6000e-7	0.000041	0.000038	0.000015	0.000028	0.000063	0.000014	0	0.0	0.0	0.0	0.125	3515.625	12367.375	46334.5
1	3	2	68	6732	34589	23283.480469	99	342.399994	151	14767	3.68	23.222221	2838.444336	50319.347656	14282.868164	122.229668	615.051697	284.666656	40527.109375	20.26111	7115	30176.300781	75	42412.226562	9.5	1023.5	1859.599976	18169.027344	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	0.0	0	0.000243	0.000095	0.0	0.0	0.0	0	0.0	0.000027	0.000015	0.000014	0	0.000563	0.000032	0.0	0.0	0.0	0.000339	0.000113	0.0	0.0	8.6000e-7	0.000041	0.000038	0.000015	0.000028	0.000063	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	4	3	69	6935	38392	26626.679688	101	385.890015	141	12936	2.83	25.0	2611.777832	51527.359375	13457.800781	104.471107	538.312073	300.125	41819.875	16.213751	9228	39411.101562	106	42708.171875	6.75	855.25	1718.0	20087.693359	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	0.0	0	0.000243	0.000095	0.0	0.0	0.0	0	0.0	0.000027	0.000015	0.000014	0	0.000563	0.000032	0.0	0.0	0.0	0.000339	0.000113	0.0	0.0	8.6000e-7	0.000041	0.000038	0.000015	0.000028	0.000063	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	5	4	47	3829	22587	8649.419922	81	184.029999	141	12936	2.67	27.111111	3129.444336	48388.820312	15143.012695	115.430328	558.553711	315.75	43923.875	21.745001	9899	45385.398438	115	45848.46875	7.75	896.0	1874.550049	20921.316406	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	0.0	0	0.000243	0.000095	0.0	0.0	0.0	0	0.0	0.000027	0.000015	0.000014	0	0.000563	0.000032	0.0	0.0	0.0	0.000339	0.000113	0.0	0.0	8.6000e-7	0.000041	0.000038	0.000015	0.000028	0.000063	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…
96	3	62	0	0	0	0.0	0	0.0	0	0	0.0	24.0	3095.375	88046.53125	27253.703125	128.973953	1135.570923	499.571442	72388.289062	34.974285	3167	14179.0	38	44771.078125	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000046	0.000054	0.0	0.0	0.0	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
96	4	63	0	0	0	0.0	0	0.0	0	0	0.0	18.875	2823.0	88205.25	24900.341797	149.562912	1319.223389	787.142883	96695.859375	43.792858	2568	11670.0	32	45443.925781	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000046	0.000054	0.0	0.0	0.0	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
96	5	64	0	0	0	0.0	0	0.0	0	0	0.0	19.625	2835.375	87010.3125	24670.6875	144.477707	1257.105103	708.25	87527.75	42.272499	3085	14007.0	36	45403.566406	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000046	0.000054	0.0	0.0	0.0	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
96	6	65	0	0	0	0.0	0	0.0	0	0	0.0	27.0	3730.875	84430.859375	31500.097656	138.180557	1166.670288	541.142883	77793.429688	38.218571	3485	14737.0	42	42286.945312	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000046	0.000054	0.0	0.0	0.0	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
96	7	66	1	140	40079	561.190002	140	561.190002	1	195	1.39	14.875	2050.125	82984.53125	17012.867188	137.823532	1143.722046	799.285706	99867.710938	42.80143	2969	12605.0	36	42455.371094	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000046	0.000054	0.0	0.0	0.0	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
%%time
data2=data
data2=data2.sort("time","sector_id"
        )

for col in data2.columns[3:]:
    for p in [#3,
              #6,
              #12,
             ]:
            print(col)
            data2=data2.with_columns(
                pl.col(col).rolling_mean(window_size=p).alias(f"{col}_mean{p}"), 
                pl.col(col).rolling_min(window_size=p).alias(f"{col}_min{p}"), 
                pl.col(col).rolling_max(window_size=p).alias(f"{col}_max{p}"), 
            #).with_columns(
            #    ( pl.col(f"{col}_max{p}")-pl.col(f"{col}_min{p}") ).alias(f"{col}_maxmin{p}")
            )

for m in [1,2,12]:
    data2=data2.join(data.drop("month_num").with_columns(pl.col("time")+m),
              on=["sector_id","time"],
              how="left",
              suffix=f"_{m}"
        )
data2=data2.sort("time","sector_id"
        )
data2
CPU times: user 111 ms, sys: 74.5 ms, total: 185 ms
Wall time: 58.6 ms
shape: (6_432, 991)
sector_id	month_num	time	nht_num_new_house_transactions	nht_area_new_house_transactions	nht_price_new_house_transactions	nht_amount_new_house_transactions	nht_area_per_unit_new_house_transactions	nht_total_price_per_unit_new_house_transactions	nht_num_new_house_available_for_sale	nht_area_new_house_available_for_sale	nht_period_new_house_sell_through	nhtns_num_new_house_transactions_nearby_sectors	nhtns_area_new_house_transactions_nearby_sectors	nhtns_price_new_house_transactions_nearby_sectors	nhtns_amount_new_house_transactions_nearby_sectors	nhtns_area_per_unit_new_house_transactions_nearby_sectors	nhtns_total_price_per_unit_new_house_transactions_nearby_sectors	nhtns_num_new_house_available_for_sale_nearby_sectors	nhtns_area_new_house_available_for_sale_nearby_sectors	nhtns_period_new_house_sell_through_nearby_sectors	pht_area_pre_owned_house_transactions	pht_amount_pre_owned_house_transactions	pht_num_pre_owned_house_transactions	pht_price_pre_owned_house_transactions	phtns_num_pre_owned_house_transactions_nearby_sectors	phtns_area_pre_owned_house_transactions_nearby_sectors	phtns_amount_pre_owned_house_transactions_nearby_sectors	phtns_price_pre_owned_house_transactions_nearby_sectors	ci_year_end_registered_population_10k	ci_total_households_10k	ci_year_end_resident_population_10k	ci_year_end_total_employed_population_10k	ci_year_end_urban_non_private_employees_10k	ci_private_individual_and_other_employees_10k	ci_private_individual_ratio	ci_national_year_end_total_population_10k	…	sp_community_winner_malls_dense_12	sp_key_focus_malls_dense_12	sp_transportation_facilities_service_bus_station_dense_12	sp_transportation_facilities_service_subway_station_dense_12	sp_transportation_facilities_service_airport_related_dense_12	sp_transportation_facilities_service_port_terminal_dense_12	sp_transportation_facilities_service_train_station_dense_12	sp_transportation_facilities_service_light_rail_station_dense_12	sp_transportation_facilities_service_long_distance_bus_station_dense_12	sp_leisure_entertainment_entertainment_venue_game_arcade_dense_12	sp_leisure_entertainment_entertainment_venue_party_house_dense_12	sp_leisure_entertainment_cultural_venue_cultural_palace_dense_12	sp_office_building_industrial_building_industrial_building_dense_12	sp_medical_health_dense_12	sp_medical_health_specialty_hospital_dense_12	sp_medical_health_tcm_hospital_dense_12	sp_medical_health_physical_examination_institution_dense_12	sp_medical_health_veterinary_station_dense_12	sp_medical_health_pharmaceutical_healthcare_dense_12	sp_medical_health_rehabilitation_institution_dense_12	sp_medical_health_first_aid_center_dense_12	sp_medical_health_blood_donation_station_dense_12	sp_medical_health_disease_prevention_institution_dense_12	sp_medical_health_general_hospital_dense_12	sp_medical_health_clinic_dense_12	sp_education_training_school_education_middle_school_dense_12	sp_education_training_school_education_primary_school_dense_12	sp_education_training_school_education_kindergarten_dense_12	sp_education_training_school_education_research_institution_dense_12	lt_num_land_transactions_12	lt_construction_area_12	lt_planned_building_area_12	lt_transaction_amount_12	ltns_num_land_transactions_nearby_sectors_12	ltns_construction_area_nearby_sectors_12	ltns_planned_building_area_nearby_sectors_12	ltns_transaction_amount_nearby_sectors_12
i8	i8	i8	i16	i32	i32	f32	i16	f32	i16	i32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i32	f32	i16	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i32	…	f32	i8	f32	f32	f32	f32	f32	i8	f32	f32	f32	f32	i8	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i8	f32	f32	f32	f32	f32	f32	f32
1	1	0	52	4906	28184	13827.139648	94	265.910004	159	15904	3.78	29.444445	3532.444336	51992.519531	18366.068359	119.96981	623.753296	350.25	49809.875	29.696251	9163	40994.699219	111	44739.386719	6.75	733.0	1247.037964	17012.796875	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null
2	1	0	145	15933	17747	28277.730469	110	195.020004	1491	175113	12.29	51.0	5830.5	18152.220703	10583.652344	114.323532	207.522598	1366.666626	170501.0	17.093334	192	315.0	2	16406.25	64.181816	5339.0	20880.242188	39108.902344	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null
3	1	0	0	0	0	0.0	0	0.0	0	0	0.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	-1	-1.0	77.714287	7457.143066	17376.214844	23301.4375	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null
4	1	0	6	725	28004	1424.209961	127	356.049988	40	6826	5.95	106.666664	10736.333008	21105.521484	22659.591797	100.653122	212.43367	2045.333374	238398.828125	29.178333	0	0.0	0	12794.69043	57.666668	5109.666504	19021.152344	37225.820312	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null
5	1	0	2	212	37432	792.099976	106	396.049988	161	17173	83.949997	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	-1	-1.0	45.42857	3763.5	15800.741211	41984.167969	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null
…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…
92	7	66	211	22084	13949	30804.740234	105	145.990005	5908	636696	34.759998	121.833336	11804.833008	18898.710938	22309.613281	96.893295	183.115845	3912.75	394516.25	26.975	15207	17992.640625	124	11831.81543	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	1.0500e-7	0	0.000003	3.1600e-7	0.0	0.0	0.0	0	0.0	1.7600e-7	3.5100e-8	2.8100e-7	0	0.000017	4.9200e-7	0.0	3.5100e-8	0.0	0.000011	0.000002	7.0200e-8	7.0200e-8	7.0200e-8	5.6200e-7	0.000002	3.8600e-7	7.3800e-7	0.000002	1.4000e-7	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
93	7	66	62	8136	27452	22335.300781	131	360.25	1323	150862	27.74	215.666672	40734.332031	27112.361328	110440.398438	188.876358	512.088379	2439.666748	270202.65625	20.293333	4892	9734.0	44	19897.792969	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.00001	0.000004	0.0	0.0	0.0	0	0.0	0.0	0.0	0.000002	0	0.000033	0.000004	0.0	0.0	0.0	0.000021	0.000004	0.0	0.0	0.0	0.0	0.000005	7.7600e-7	0.000002	0.000008	0.0	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
94	7	66	44	5078	26367	13389.410156	115	304.299988	2027	215821	38.619999	47.0	4594.600098	18871.40625	8670.65625	97.757446	184.48204	1458.0	149751.40625	26.068001	7067	13092.0	78	18525.541016	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000003	5.3800e-7	0.0	0.0	0.0	0	0.0	0.0	0.0	0.0	0	9.6900e-7	0.0	0.0	0.0	0.0	5.3800e-7	1.0800e-7	0.0	0.0	0.0	2.1500e-7	1.0800e-7	2.1500e-7	1.0800e-7	3.2300e-7	1.0800e-7	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
95	7	66	0	0	0	0.0	0	0.0	0	0	0.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	-1	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000283	0.00002	0.0	0.00002	0.0	0	0.0	0.0	0.0	0.00002	0	0.000748	0.000081	0.00002	0.0	0.0	0.000384	0.000141	0.0	0.0	0.0	0.000061	0.000061	0.0	0.0	0.00004	0.000061	0	0.0	0.0	0.0	-1.0	-1.0	-1.0	-1.0
96	7	66	1	140	40079	561.190002	140	561.190002	1	195	1.39	14.875	2050.125	82984.53125	17012.867188	137.823532	1143.722046	799.285706	99867.710938	42.80143	2969	12605.0	36	42455.371094	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	…	0.0	0	0.000046	0.000054	0.0	0.0	0.0	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
lag=-1
%%time
data3 = data2.with_columns(pl.col("nht_amount_new_house_transactions").shift(lag).over("sector_id").alias("label"),
                           #pl.col("amount_new_house_transactions").shift(lag+1).over("sector_id").alias("label1"),
                           #pl.col("amount_new_house_transactions").shift(lag+2).over("sector_id").alias("label2"),
                           #pl.col("amount_new_house_transactions").shift(lag+3).over("sector_id").alias("label3"),
                           #pl.col("amount_new_house_transactions").shift(lag+4).over("sector_id").alias("label4"),
                           #pl.col("amount_new_house_transactions").shift(lag+5).over("sector_id").alias("label5"),
                           
                           cs=((pl.col("month_num")-1)/6*np.pi).cos(),
                           sn=((pl.col("month_num")-1)/6*np.pi).sin(),
                           cs6=((pl.col("month_num")-1)/3*np.pi).cos(),
                           sn6=((pl.col("month_num")-1)/3*np.pi).sin(),
                           cs3=((pl.col("month_num")-1)/1.5*np.pi).cos(),
                           sn3=((pl.col("month_num")-1)/1.5*np.pi).sin(),
#                          ).with_columns( -pl.col("label")+1
                          )#.drop_nulls(subset=["label"])
data3 = data3.drop("sector_id")
data3
CPU times: user 8.51 ms, sys: 2.11 ms, total: 10.6 ms
Wall time: 14.7 ms
shape: (6_432, 997)
month_num	time	nht_num_new_house_transactions	nht_area_new_house_transactions	nht_price_new_house_transactions	nht_amount_new_house_transactions	nht_area_per_unit_new_house_transactions	nht_total_price_per_unit_new_house_transactions	nht_num_new_house_available_for_sale	nht_area_new_house_available_for_sale	nht_period_new_house_sell_through	nhtns_num_new_house_transactions_nearby_sectors	nhtns_area_new_house_transactions_nearby_sectors	nhtns_price_new_house_transactions_nearby_sectors	nhtns_amount_new_house_transactions_nearby_sectors	nhtns_area_per_unit_new_house_transactions_nearby_sectors	nhtns_total_price_per_unit_new_house_transactions_nearby_sectors	nhtns_num_new_house_available_for_sale_nearby_sectors	nhtns_area_new_house_available_for_sale_nearby_sectors	nhtns_period_new_house_sell_through_nearby_sectors	pht_area_pre_owned_house_transactions	pht_amount_pre_owned_house_transactions	pht_num_pre_owned_house_transactions	pht_price_pre_owned_house_transactions	phtns_num_pre_owned_house_transactions_nearby_sectors	phtns_area_pre_owned_house_transactions_nearby_sectors	phtns_amount_pre_owned_house_transactions_nearby_sectors	phtns_price_pre_owned_house_transactions_nearby_sectors	ci_year_end_registered_population_10k	ci_total_households_10k	ci_year_end_resident_population_10k	ci_year_end_total_employed_population_10k	ci_year_end_urban_non_private_employees_10k	ci_private_individual_and_other_employees_10k	ci_private_individual_ratio	ci_national_year_end_total_population_10k	ci_resident_registered_ratio	…	sp_transportation_facilities_service_light_rail_station_dense_12	sp_transportation_facilities_service_long_distance_bus_station_dense_12	sp_leisure_entertainment_entertainment_venue_game_arcade_dense_12	sp_leisure_entertainment_entertainment_venue_party_house_dense_12	sp_leisure_entertainment_cultural_venue_cultural_palace_dense_12	sp_office_building_industrial_building_industrial_building_dense_12	sp_medical_health_dense_12	sp_medical_health_specialty_hospital_dense_12	sp_medical_health_tcm_hospital_dense_12	sp_medical_health_physical_examination_institution_dense_12	sp_medical_health_veterinary_station_dense_12	sp_medical_health_pharmaceutical_healthcare_dense_12	sp_medical_health_rehabilitation_institution_dense_12	sp_medical_health_first_aid_center_dense_12	sp_medical_health_blood_donation_station_dense_12	sp_medical_health_disease_prevention_institution_dense_12	sp_medical_health_general_hospital_dense_12	sp_medical_health_clinic_dense_12	sp_education_training_school_education_middle_school_dense_12	sp_education_training_school_education_primary_school_dense_12	sp_education_training_school_education_kindergarten_dense_12	sp_education_training_school_education_research_institution_dense_12	lt_num_land_transactions_12	lt_construction_area_12	lt_planned_building_area_12	lt_transaction_amount_12	ltns_num_land_transactions_nearby_sectors_12	ltns_construction_area_nearby_sectors_12	ltns_planned_building_area_nearby_sectors_12	ltns_transaction_amount_nearby_sectors_12	label	cs	sn	cs6	sn6	cs3	sn3
i8	i8	i16	i32	i32	f32	i16	f32	i16	i32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i32	f32	i16	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i32	f32	…	i8	f32	f32	f32	f32	i8	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	f32	i8	f32	f32	f32	f32	f32	f32	f32	f32	f64	f64	f64	f64	f64	f64
1	0	52	4906	28184	13827.139648	94	265.910004	159	15904	3.78	29.444445	3532.444336	51992.519531	18366.068359	119.96981	623.753296	350.25	49809.875	29.696251	9163	40994.699219	111	44739.386719	6.75	733.0	1247.037964	17012.796875	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	1.604863	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	8802.80957	1.0	0.0	1.0	0.0	1.0	0.0
1	0	145	15933	17747	28277.730469	110	195.020004	1491	175113	12.29	51.0	5830.5	18152.220703	10583.652344	114.323532	207.522598	1366.666626	170501.0	17.093334	192	315.0	2	16406.25	64.181816	5339.0	20880.242188	39108.902344	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	1.604863	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	12868.219727	1.0	0.0	1.0	0.0	1.0	0.0
1	0	0	0	0	0.0	0	0.0	0	0	0.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	-1	-1.0	77.714287	7457.143066	17376.214844	23301.4375	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	1.604863	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	0.0	1.0	0.0	1.0	0.0	1.0	0.0
1	0	6	725	28004	1424.209961	127	356.049988	40	6826	5.95	106.666664	10736.333008	21105.521484	22659.591797	100.653122	212.43367	2045.333374	238398.828125	29.178333	0	0.0	0	12794.69043	57.666668	5109.666504	19021.152344	37225.820312	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	1.604863	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	1522.030029	1.0	0.0	1.0	0.0	1.0	0.0
1	0	2	212	37432	792.099976	106	396.049988	161	17173	83.949997	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	-1	-1.0	45.42857	3763.5	15800.741211	41984.167969	953.719971	313.84549	1530.589966	1125.890015	400.220001	725.671997	0.644532	140005	1.604863	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	409.130005	1.0	0.0	1.0	0.0	1.0	0.0
…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…
7	66	211	22084	13949	30804.740234	105	145.990005	5908	636696	34.759998	121.833336	11804.833008	18898.710938	22309.613281	96.893295	183.115845	3912.75	394516.25	26.975	15207	17992.640625	124	11831.81543	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	…	0	0.0	1.7600e-7	3.5100e-8	2.8100e-7	0	0.000017	4.9200e-7	0.0	3.5100e-8	0.0	0.000011	0.000002	7.0200e-8	7.0200e-8	7.0200e-8	5.6200e-7	0.000002	3.8600e-7	7.3800e-7	0.000002	1.4000e-7	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	null	-1.0	1.2246e-16	1.0	-2.4493e-16	1.0	-4.8986e-16
7	66	62	8136	27452	22335.300781	131	360.25	1323	150862	27.74	215.666672	40734.332031	27112.361328	110440.398438	188.876358	512.088379	2439.666748	270202.65625	20.293333	4892	9734.0	44	19897.792969	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	…	0	0.0	0.0	0.0	0.000002	0	0.000033	0.000004	0.0	0.0	0.0	0.000021	0.000004	0.0	0.0	0.0	0.0	0.000005	7.7600e-7	0.000002	0.000008	0.0	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	null	-1.0	1.2246e-16	1.0	-2.4493e-16	1.0	-4.8986e-16
7	66	44	5078	26367	13389.410156	115	304.299988	2027	215821	38.619999	47.0	4594.600098	18871.40625	8670.65625	97.757446	184.48204	1458.0	149751.40625	26.068001	7067	13092.0	78	18525.541016	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	…	0	0.0	0.0	0.0	0.0	0	9.6900e-7	0.0	0.0	0.0	0.0	5.3800e-7	1.0800e-7	0.0	0.0	0.0	2.1500e-7	1.0800e-7	2.1500e-7	1.0800e-7	3.2300e-7	1.0800e-7	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	null	-1.0	1.2246e-16	1.0	-2.4493e-16	1.0	-4.8986e-16
7	66	0	0	0	0.0	0	0.0	0	0	0.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	-1	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	…	0	0.0	0.0	0.0	0.00002	0	0.000748	0.000081	0.00002	0.0	0.0	0.000384	0.000141	0.0	0.0	0.0	0.000061	0.000061	0.0	0.0	0.00004	0.000061	0	0.0	0.0	0.0	-1.0	-1.0	-1.0	-1.0	null	-1.0	1.2246e-16	1.0	-2.4493e-16	1.0	-4.8986e-16
7	66	1	140	40079	561.190002	140	561.190002	1	195	1.39	14.875	2050.125	82984.53125	17012.867188	137.823532	1143.722046	799.285706	99867.710938	42.80143	2969	12605.0	36	42455.371094	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1	-1.0	…	0	0.0	0.00004	0.000021	0.000004	0	0.000441	0.000085	0.0	0.000006	0.0	0.000197	0.000048	0.000002	0.000008	0.000004	0.000014	0.000079	0.00001	0.00001	0.000033	0.000014	0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	null	-1.0	1.2246e-16	1.0	-2.4493e-16	1.0	-4.8986e-16
%%time
cat_features = [#"sector_id",
                "month_num"
               ]
border=66+lag-12*0-1
#border1=border//2 #-1
border1=border-6*6
border1=6*3
trainPool = Pool(data=data3.filter(pl.col("time")<=border
                                  #).filter(pl.col("label")>10
                                  ).filter(pl.col("time")>border1
                                          ).drop(["label",
                                                  #"label1","label2","label3","label4","label5",
                                                 ]).to_pandas().fillna(-2),
                 label=data3.filter(pl.col("time")<=border
                                  #).filter(pl.col("label")>10
                                   ).filter(pl.col("time")>border1
                                           )["label",
                                             #"label1","label2","label3","label4","label5",
                                            ].to_pandas(),
                 cat_features = cat_features, 
                 #text_features = text_features, 
                )

testPool = Pool(data=data3.filter(pl.col("time")>border
                                  ).filter(pl.col("time")<=66+lag
                                  #).filter(pl.col("label")>10
                                 ).drop(["label",
                                         #"label1","label2","label3","label4","label5",
                                        ]).to_pandas().fillna(-2),
                 label=data3.filter(pl.col("time")>border
                                  ).filter(pl.col("time")<=66+lag
                                  #).filter(pl.col("label")>10
                                   )["label",
                                     #"label1","label2","label3","label4","label5",
                                    ].to_pandas(),
                 cat_features = cat_features, 
                 #text_features = text_features, 
                )
trainPool
CPU times: user 244 ms, sys: 75.6 ms, total: 319 ms
Wall time: 261 ms
<catboost.core.Pool at 0x7bed4332b830>
#custom eval metric for catboost

# based on https://www.kaggle.com/competitions/china-real-estate-demand-prediction/discussion/598174

def custom_score(y_true, y_pred, eps=1e-12):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return 0.0

    ape = np.abs((y_true - np.maximum(y_pred,0) ) / np.maximum(y_true, eps))

    bad_rate = np.mean(ape > 1.0)
    if bad_rate > 0.30:
        return 0.0

    mask = ape <= 1.0
    good_ape = ape[mask]

    if good_ape.size == 0:
        return 0.0

    mape = np.mean(good_ape)

    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    score = max(0.0, 1.0 - scaled_mape)
    return score

class CustomMetric:
    def is_max_optimal(self):
        return True # greater is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        y_pred = approx
        y_true = target

        output_weight = 1 # weight is not used

        score = custom_score( target, approx )
 
        return score, output_weight

    def get_final_error(self, error, weight):
        return error
#custom loss for catboost
class CustomObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        result = []
        delta = 10**(-6)
        for i in range(len(targets)):
            diff = (targets[i] - approxes[i])#/(targets[i]+delta)
            #der1 = np.sign(diff)
            der1 = np.sign(diff) if (2*targets[i] - approxes[i]) < 0 else np.sign(diff)*5
            der2 = 0
            result.append((der1, der2))
        return result  
testPool2 = Pool(data=data3.filter(pl.col("time")==66
                                 ).drop(["label"]).to_pandas().fillna(-2),
                 cat_features = cat_features, 
                )

cb = CatBoostRegressor(iterations=21000,
                            learning_rate=0.0125,
                            one_hot_max_size=256,
                            custom_metric=[                            
                                           "RMSE",
                                           "MAPE",
                                           "SMAPE",
                                           "MAE",
                            ],
                            loss_function=CustomObjective(),
                            eval_metric=CustomMetric(),
                            l2_leaf_reg=0.3,
                            random_seed=4,
                           )
cb.fit(trainPool,
           eval_set=testPool,
           #early_stopping_rounds=4000,
           #use_best_model=False,
           #plot=True,
           verbose=1000,           
           )
month=np.maximum(cb.predict(testPool2),0)
month
/usr/local/lib/python3.11/dist-packages/catboost/core.py:1790: UserWarning: Failed to optimize method "evaluate" in the passed object:
Failed in nopython mode pipeline (step: nopython frontend)
Untyped global name 'custom_score': Cannot determine Numba type of <class 'function'>

File "../../tmp/ipykernel_13/2665501567.py", line 47:
<source missing, REPL/exec in use?>

  self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
0:	learn: 0.0000000	test: 0.0000000	best: 0.0000000 (0)	total: 1.65s	remaining: 9h 35m 59s
1000:	learn: 0.5039899	test: 0.4830896	best: 0.4977141 (988)	total: 46.8s	remaining: 15m 35s
2000:	learn: 0.5402593	test: 0.5275462	best: 0.5275462 (2000)	total: 1m 31s	remaining: 14m 29s
3000:	learn: 0.5642219	test: 0.5177333	best: 0.5356251 (2409)	total: 2m 16s	remaining: 13m 36s
4000:	learn: 0.5827448	test: 0.5189696	best: 0.5356251 (2409)	total: 3m 1s	remaining: 12m 49s
5000:	learn: 0.5969942	test: 0.5484097	best: 0.5659240 (4246)	total: 3m 45s	remaining: 12m
6000:	learn: 0.6096808	test: 0.5853091	best: 0.5894812 (5927)	total: 4m 29s	remaining: 11m 14s
7000:	learn: 0.6169048	test: 0.5676445	best: 0.5997168 (6075)	total: 5m 14s	remaining: 10m 28s
8000:	learn: 0.6294979	test: 0.5987088	best: 0.6011210 (7940)	total: 6m	remaining: 9m 45s
9000:	learn: 0.6427181	test: 0.6045021	best: 0.6045361 (8992)	total: 6m 44s	remaining: 8m 59s
10000:	learn: 0.6525037	test: 0.6031792	best: 0.6071935 (9645)	total: 7m 29s	remaining: 8m 14s
11000:	learn: 0.6644573	test: 0.6093246	best: 0.6110193 (10821)	total: 8m 13s	remaining: 7m 28s
12000:	learn: 0.6733714	test: 0.6063326	best: 0.6126249 (11394)	total: 8m 57s	remaining: 6m 43s
13000:	learn: 0.6835417	test: 0.6065582	best: 0.6126249 (11394)	total: 9m 41s	remaining: 5m 57s
14000:	learn: 0.6909731	test: 0.6012277	best: 0.6177883 (13556)	total: 10m 26s	remaining: 5m 13s
15000:	learn: 0.6976927	test: 0.6035575	best: 0.6177883 (13556)	total: 11m 10s	remaining: 4m 28s
16000:	learn: 0.7090310	test: 0.6102371	best: 0.6177883 (13556)	total: 11m 54s	remaining: 3m 43s
17000:	learn: 0.7171831	test: 0.6109518	best: 0.6177883 (13556)	total: 12m 38s	remaining: 2m 58s
18000:	learn: 0.7272085	test: 0.6091775	best: 0.6177883 (13556)	total: 13m 22s	remaining: 2m 13s
19000:	learn: 0.7349961	test: 0.6076191	best: 0.6177883 (13556)	total: 14m 6s	remaining: 1m 29s
20000:	learn: 0.7420185	test: 0.6092236	best: 0.6214477 (19232)	total: 14m 50s	remaining: 44.5s
20999:	learn: 0.7508729	test: 0.6117854	best: 0.6226570 (20754)	total: 15m 33s	remaining: 0us

bestTest = 0.622656999
bestIteration = 20754

Shrink model to first 20755 iterations.
array([1.37415830e+04, 4.97570821e+03, 8.41533312e+03, 8.01570397e+04,
       4.18458323e+03, 1.52797080e+04, 7.00004149e+03, 2.92645826e+03,
       1.05940414e+04, 3.33289158e+04, 4.43062489e+03, 1.32574997e+03,
       2.85249993e+03, 2.08819995e+04, 9.17720811e+04, 6.63247900e+04,
       1.45258330e+03, 2.09289161e+04, 3.42491658e+03, 8.19308313e+03,
       2.36161661e+04, 1.97551662e+04, 2.67647493e+04, 6.56815817e+04,
       1.83853329e+04, 3.85508324e+03, 4.01437490e+03, 3.48696241e+04,
       2.48272911e+04, 3.33330825e+04, 6.23658318e+03, 3.87802074e+04,
       1.20084997e+04, 7.75670814e+04, 9.11033311e+03, 1.04704581e+05,
       1.26389997e+04, 3.92610407e+04, 1.20749997e+02, 9.78908309e+03,
       8.67499978e+01, 3.48095825e+03, 1.29211247e+04, 1.16220414e+04,
       6.87717483e+04, 5.29604154e+03, 2.10115828e+04, 2.48279577e+04,
       5.31291653e+02, 1.75159162e+04, 1.48634163e+04, 0.00000000e+00,
       1.66595829e+03, 4.82193321e+04, 9.47286643e+04, 3.37537492e+03,
       2.39686661e+04, 1.89166662e+01, 2.10760411e+04, 9.63066643e+03,
       6.46724984e+04, 1.32012913e+04, 7.51570815e+03, 1.63529579e+04,
       9.82862476e+03, 5.14579987e+04, 1.66524996e+03, 1.68018746e+04,
       3.58470824e+03, 1.99212495e+03, 8.44258312e+03, 1.34591663e+03,
       1.04520831e+03, 6.53208317e+02, 4.84958321e+02, 2.70667077e+04,
       5.38631653e+04, 1.67306663e+04, 3.76367907e+04, 2.10941661e+04,
       1.46672913e+04, 1.20329164e+03, 5.30248320e+04, 7.02124983e+03,
       1.11523164e+05, 4.09392906e+04, 0.00000000e+00, 3.14724992e+03,
       3.14249992e+02, 5.20716654e+03, 3.09151242e+04, 2.18040411e+04,
       1.48029580e+04, 1.36346247e+04, 0.00000000e+00, 3.82708324e+02])
#%%time
importance = cb.get_feature_importance(prettified=True,data=testPool)
importance.to_csv("importance.csv")
importance.head(60)
Feature Id	Importances
0	nht_amount_new_house_transactions	21.626483
1	nht_amount_new_house_transactions_1	10.311945
2	nht_amount_new_house_transactions_2	3.348693
3	nht_area_per_unit_new_house_transactions	2.862383
4	sp_lifestyle_leisure_stores_dense_2	1.883345
5	nht_num_new_house_available_for_sale	1.841162
6	sp_transportation_facilities_service_bus_stati...	1.704617
7	nht_area_new_house_available_for_sale	1.592881
8	nht_area_new_house_transactions_2	0.990010
9	nht_price_new_house_transactions_2	0.879488
10	nht_area_per_unit_new_house_transactions_1	0.839089
11	sp_number_of_catering_food_stores_2	0.813124
12	nht_total_price_per_unit_new_house_transactions_1	0.787746
13	nht_amount_new_house_transactions_12	0.738164
14	nht_period_new_house_sell_through	0.732147
15	nhtns_total_price_per_unit_new_house_transacti...	0.717655
16	pht_num_pre_owned_house_transactions_2	0.698623
17	sp_sector_coverage	0.652341
18	sp_office_population_2	0.642088
19	nht_period_new_house_sell_through_12	0.621470
20	nht_period_new_house_sell_through_2	0.593950
21	nht_num_new_house_transactions_1	0.525859
22	nhtns_period_new_house_sell_through_nearby_sec...	0.516621
23	nht_num_new_house_transactions	0.511917
24	sp_transportation_facilities_service_train_sta...	0.507355
25	nhtns_period_new_house_sell_through_nearby_sec...	0.492335
26	sp_transportation_facilities_service_subway_st...	0.492331
27	nhtns_amount_new_house_transactions_nearby_sec...	0.489805
28	sp_education_training_school_education_primary...	0.483157
29	nht_area_new_house_transactions	0.481334
30	nht_num_new_house_transactions_2	0.468887
31	nht_period_new_house_sell_through_1	0.461470
32	nht_area_new_house_transactions_1	0.457777
33	nhtns_amount_new_house_transactions_nearby_sec...	0.432178
34	nht_price_new_house_transactions_1	0.420273
35	phtns_amount_pre_owned_house_transactions_near...	0.412781
36	sp_number_of_home_appliance_stores	0.403222
37	sp_community_winner_malls_dense_1	0.402780
38	phtns_num_pre_owned_house_transactions_nearby_...	0.385676
39	pht_amount_pre_owned_house_transactions	0.375988
40	nhtns_period_new_house_sell_through_nearby_sec...	0.359227
41	sp_residential_area_dense_12	0.358539
42	nhtns_area_per_unit_new_house_transactions_nea...	0.354187
43	nht_area_new_house_available_for_sale_2	0.350446
44	nhtns_area_per_unit_new_house_transactions_nea...	0.345487
45	phtns_amount_pre_owned_house_transactions_near...	0.343437
46	nht_area_per_unit_new_house_transactions_12	0.336879
47	phtns_price_pre_owned_house_transactions_nearb...	0.336650
48	phtns_price_pre_owned_house_transactions_nearb...	0.335792
49	nht_area_per_unit_new_house_transactions_2	0.323856
50	nht_area_new_house_available_for_sale_1	0.317959
51	sp_office_population_dense_1	0.316831
52	phtns_area_pre_owned_house_transactions_nearby...	0.314394
53	sp_catering_dense	0.310081
54	nht_price_new_house_transactions	0.309761
55	nht_num_new_house_available_for_sale_2	0.298061
56	sp_home_appliance_stores_dense_1	0.296898
57	nhtns_amount_new_house_transactions_nearby_sec...	0.293056
58	sp_transportation_station_1	0.286658
59	nht_price_new_house_transactions_12	0.277545
#importance.tail(60)
#based on https://www.kaggle.com/code/ambrosm/red-explained-baseline/notebook
for i in [11,38,40,43,48,51,52,57,71,72,73,74,81,86,88,94,95]:
    month[i]=0
month
array([ 13741.58299206,   4975.70820976,   8415.33312434,  80157.03967594,
         4184.58322941,  15279.70795386,   7000.04149282,   2926.45826065,
        10594.04140356,  33328.91583893,   4430.62488996,      0.        ,
         2852.49992916,  20881.99948139,  91772.08105415,  66324.79001947,
         1452.58329726,  20928.91614689,   3424.91658161,   8193.08312986,
        23616.16608015,  19755.16617604,  26764.74933529,  65681.58170211,
        18385.33287673,   3855.08323759,   4014.3749003 ,  34869.624134  ,
        24827.29105007,  33333.0825055 ,   6236.58317845,  38780.20737022,
        12008.49970177,  77567.08140693,   9110.33310708, 104704.58073297,
        12638.99968611,  39261.04069161,      0.        ,   9789.08309022,
            0.        ,   3480.95824688,  12921.1246791 ,      0.        ,
        68771.74829203,   5296.04153514,  21011.5828115 ,  24827.95771672,
            0.        ,  17515.91623165,  14863.41629753,      0.        ,
            0.        ,  48219.33213579,  94728.66431405,   3375.37491617,
        23968.6660714 ,      0.        ,  21076.04114324,   9630.66642749,
        64672.49839384,  13201.29133881,   7515.70814668,  16352.9579272 ,
         9828.6247559 ,  51457.99872203,   1665.24995864,  16801.87458272,
         3584.70824431,   1992.12495053,   8442.58312366,      0.        ,
            0.        ,      0.        ,      0.        ,  27066.70766112,
        53863.16532896,  16730.66625116,  37636.79073195,  21094.16614279,
        14667.2913024 ,      0.        ,  53024.83201645,   7021.24982563,
       111523.16389696,  40939.29064993,      0.        ,   3147.24992184,
            0.        ,   5207.16653735,  30915.12423221,  21804.04112516,
        14802.9579657 ,  13634.62466138,      0.        ,      0.        ])
sub = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv")
sub
id	new_house_transaction_amount
0	2024 Aug_sector 1	100000
1	2024 Aug_sector 2	100000
2	2024 Aug_sector 3	100000
3	2024 Aug_sector 4	100000
4	2024 Aug_sector 5	100000
...	...	...
1147	2025 Jul_sector 92	100000
1148	2025 Jul_sector 93	100000
1149	2025 Jul_sector 94	100000
1150	2025 Jul_sector 95	100000
1151	2025 Jul_sector 96	100000
1152 rows × 2 columns

for m in range(12):
    sub.loc[[i+m*96 for i in range(96)], "new_house_transaction_amount"]=month
sub.to_csv("submission.csv", index=False)
sub
/tmp/ipykernel_13/628683564.py:2: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '[ 13741.58299206   4975.70820976   8415.33312434  80157.03967594
   4184.58322941  15279.70795386   7000.04149282   2926.45826065
  10594.04140356  33328.91583893   4430.62488996      0.
   2852.49992916  20881.99948139  91772.08105415  66324.79001947
   1452.58329726  20928.91614689   3424.91658161   8193.08312986
  23616.16608015  19755.16617604  26764.74933529  65681.58170211
  18385.33287673   3855.08323759   4014.3749003   34869.624134
  24827.29105007  33333.0825055    6236.58317845  38780.20737022
  12008.49970177  77567.08140693   9110.33310708 104704.58073297
  12638.99968611  39261.04069161      0.           9789.08309022
      0.           3480.95824688  12921.1246791       0.
  68771.74829203   5296.04153514  21011.5828115   24827.95771672
      0.          17515.91623165  14863.41629753      0.
      0.          48219.33213579  94728.66431405   3375.37491617
  23968.6660714       0.          21076.04114324   9630.66642749
  64672.49839384  13201.29133881   7515.70814668  16352.9579272
   9828.6247559   51457.99872203   1665.24995864  16801.87458272
   3584.70824431   1992.12495053   8442.58312366      0.
      0.              0.              0.          27066.70766112
  53863.16532896  16730.66625116  37636.79073195  21094.16614279
  14667.2913024       0.          53024.83201645   7021.24982563
 111523.16389696  40939.29064993      0.           3147.24992184
      0.           5207.16653735  30915.12423221  21804.04112516
  14802.9579657   13634.62466138      0.              0.        ]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  sub.loc[[i+m*96 for i in range(96)], "new_house_transaction_amount"]=month
id	new_house_transaction_amount
0	2024 Aug_sector 1	13741.582992
1	2024 Aug_sector 2	4975.708210
2	2024 Aug_sector 3	8415.333124
3	2024 Aug_sector 4	80157.039676
4	2024 Aug_sector 5	4184.583229
...	...	...
1147	2025 Jul_sector 92	21804.041125
1148	2025 Jul_sector 93	14802.957966
1149	2025 Jul_sector 94	13634.624661
1150	2025 Jul_sector 95	0.000000
1151	2025 Jul_sector 96	0.000000
1152 rows × 2 columns