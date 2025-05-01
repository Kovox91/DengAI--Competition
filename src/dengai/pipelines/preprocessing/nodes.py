import pandas as pd
import numpy as np
from typing import List


def merge_dataframes(
    dengue_features_train: pd.DataFrame,
    dengue_labels_train: pd.DataFrame,
    dengue_features_test: pd.DataFrame,
):
    """Merges the train features and labels, subsequently add the test data.

    Args:
        dengue_features_train (pd.DataFrame)
        dengue_labels_train (pd.DataFrame)
        dengue_features_test (pd.DataFrame)

    Returns:
        pd.DataFrame: Merged train and test data
    """

    dengue_train = pd.merge(
        dengue_features_train, dengue_labels_train, on=["city", "weekofyear", "year"]
    )
    return pd.concat([dengue_train, dengue_features_test], axis=0, ignore_index=True)


def impute_with_mean(merged_df):
    """
    Replace missing values in all columns except the target column
    with the mean of their respective columns.

    Args:
        df (pd.DataFrame): The input DataFrame with missing values.
        target_column (str): The name of the column to exclude from imputation.

    Returns:
        pd.DataFrame: A new DataFrame with imputed values.
    """
    df_copy = merged_df.copy()

    for col in df_copy.columns:
        if col != "total_cases" and df_copy[col].dtype in [float, int]:
            mean_value = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_value)

    return df_copy


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
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    encode_cyclical("month", 12)
    encode_cyclical("weekofyear", 52)
    encode_cyclical("dayofweek", 7)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, lag_features: List[str], lags: List[int], roll_windows: List[int]
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
    return df
