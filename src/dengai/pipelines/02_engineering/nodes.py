import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pdb


def removal_nonnumeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    This fucntion removes the non- numeric columns and encodes city with 1 and 2

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:output dataframe
    """
    # Column to keep (even if non-numeric)
    city_column = "city"

    df["city"] = df["city"].replace({"sj": 1, "iq": 2})

    # Filter numeric columns and add the keep_column
    numeric_data = df.select_dtypes(include="number")  # keeps only numeric columns
    if city_column in df.columns:
        numeric_data[city_column] = df[city_column]

    return numeric_data


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

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, lag_features: list[str], lags: list[int], roll_windows: list[int]
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
    return df


def add_favourable_conditions(df: pd.DataFrame, params: dict):
    conditions = params["conditions"]
    temperature_sources = params["temperature_sources"]

    for label, cfg in conditions.items():
        for lag in cfg["lags"]:
            for col_pattern, suffix in temperature_sources.items():
                col_name = col_pattern.format(lag)
                new_col = f"{label}_{lag}{suffix}"
                df[new_col] = (
                    (df[col_name] >= cfg["min_temp"])
                    & (df[col_name] <= cfg["max_temp"])
                ).astype(int)
    return df


def split_data(df: pd.DataFrame):
    """
    This function accepts the complete data and split into validation and train and test dataset
    df (pd.DataFrame): The input DataFrame.
    train_df, test_df,validation_df  (pd.DataFrame): The  output retrun the train ,test,validation dataframe.

    """

    # 1. Separate validation set (rows with missing target)
    validation_data = df[df["total_cases"].isna()]

    validation_data = validation_data.drop("total_cases", axis=1)

    # 2. Keep only rows with available target values
    labeled_df = df[df["total_cases"].notna()]

    # 3. Split labeled data into X and y
    X = labeled_df.drop("total_cases", axis=1)
    y = labeled_df["total_cases"]

    # 4. Split labeled data into train (80%) and test (20%)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, validation_data


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
