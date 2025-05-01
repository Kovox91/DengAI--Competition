import pandas as pd
import numpy as np


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime format
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    # Extract date components
    df["month"] = df["week_start_date"].dt.month
    df["weekofyear"] = df["week_start_date"].dt.isocalendar().week
    df["dayofweek"] = df["week_start_date"].dt.dayofweek

    # Cyclical encoding
    def encode_cyclical(col, period):
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        # df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    encode_cyclical("month", max(df["month"]))
    encode_cyclical("weekofyear", max(df["weekofyear"]))
    encode_cyclical("dayofweek", 7)
    # df.drop(["month", "weekofyear", "dayofweek"], axis=1, inplace=True)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lag_features: list[str] = [
        "station_max_temp_c",
        "station_min_temp_c",
        "station_avg_temp_c",
        "station_precip_mm",
        "station_diur_temp_rng_c",
        "precipitation_amt_mm",
        "reanalysis_sat_precip_amt_mm",
        "reanalysis_dew_point_temp_k",
        "reanalysis_air_temp_k",
        "reanalysis_relative_humidity_percent",
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_precip_amt_kg_per_m2",
        "reanalysis_max_air_temp_k",
        "reanalysis_min_air_temp_k",
        "reanalysis_avg_temp_k",
        "reanalysis_tdtr_k",
        "ndvi_se",
        "ndvi_sw",
        "ndvi_ne",
        "ndvi_nw",
    ],
    lags: list[int] = [1, 2, 3, 4],
    roll_windows: list[int] = [3, 5],
) -> pd.DataFrame:
    # Sort before creating features
    df = df.sort_values(["city", "week_start_date"])

    for col in lag_features:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)
        for window in roll_windows:
            df[f"{col}_rollmean{window}"] = df.groupby("city")[col].transform(
                lambda x: x.rolling(window).mean()
            )

    # Sort back if needed
    df = df.sort_values(by=["city", "week_start_date"], ascending=[False, True])
    df.drop(lag_features, axis=1, inplace=True)
    print(df.head())

    return df


def impute_with_mean(df, target_col):
    """
    Replace missing values in all columns except the target column
    with the mean of their respective columns.

    Args:
        df (pd.DataFrame): The input DataFrame with missing values.
        target_column (str): The name of the column to exclude from imputation.

    Returns:
        pd.DataFrame: A new DataFrame with imputed values.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        if col != target_col and df_copy[col].dtype in [float, int]:
            mean_value = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_value)

    return df_copy


def create_favorable_temp_cols(df: pd.DataFrame):
    df["good_larva_4"] = (
        (df["station_avg_temp_c_lag4"] >= 15) & (df["station_avg_temp_c_lag4"] <= 35)
    ).astype(int)
    df["good_larva_3"] = (
        (df["station_avg_temp_c_lag3"] >= 15) & (df["station_avg_temp_c_lag3"] <= 35)
    ).astype(int)

    df["good_larva_4-2"] = (
        (df["reanalysis_air_temp_k_lag4"] >= 15)
        & (df["reanalysis_air_temp_k_lag4"] <= 35)
    ).astype(int)
    df["good_larva_3-2"] = (
        (df["reanalysis_air_temp_k_lag3"] >= 15)
        & (df["reanalysis_air_temp_k_lag3"] <= 35)
    ).astype(int)

    df["good_adult_2"] = (
        (df["station_avg_temp_c_lag2"] >= 10) & (df["station_avg_temp_c_lag2"] <= 39)
    ).astype(int)
    df["good_adult_1"] = (
        (df["station_avg_temp_c_lag1"] >= 10) & (df["station_avg_temp_c_lag1"] <= 39)
    ).astype(int)

    df["good_adult_2"] = (
        (df["reanalysis_air_temp_k_lag2"] >= 10)
        & (df["reanalysis_air_temp_k_lag2"] <= 39)
    ).astype(int)
    df["good_adult_1"] = (
        (df["reanalysis_air_temp_k_lag1"] >= 10)
        & (df["reanalysis_air_temp_k_lag1"] <= 39)
    ).astype(int)
    return df
