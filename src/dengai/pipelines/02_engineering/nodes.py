import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def removal_nonnumeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes non-numeric columns from the DataFrame, retaining and encoding the 'city' column.

    The 'city' column is encoded with values: 'sj' → 1 and 'iq' → 2, and is preserved in the output.

    Args:
        df (pd.DataFrame): Input DataFrame possibly containing non-numeric columns.

    Returns:
        pd.DataFrame: A DataFrame containing only numeric columns and the encoded 'city' column.
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
    """
    Adds cyclical (sinusoidal) features to a DataFrame based on the 'week_start_date' column.

    Extracts month, ISO week number, and day of the week from the date,
    then applies sine transformation to capture their cyclical nature.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'week_start_date' column.

    Returns:
        pd.DataFrame: The original DataFrame with added cyclical features.
    """
    # Ensure the date column is in datetime format
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    # Extract temporal features
    df["month"] = df["week_start_date"].dt.month
    df["weekofyear"] = df["week_start_date"].dt.isocalendar().week
    df["dayofweek"] = df["week_start_date"].dt.dayofweek

    # Apply sine encoding to reflect cyclical nature of time-based features
    def encode_cyclical(col, period):
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        # Cosine component can be added for 2D embedding if needed
        # df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    encode_cyclical("month", max(df["month"]))
    encode_cyclical("weekofyear", max(df["weekofyear"]))
    encode_cyclical("dayofweek", 7)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, lag_features: list[str], lags: list[int], roll_windows: list[int]
) -> pd.DataFrame:
    """
    Adds lag and rolling mean features to specified columns, grouped by 'city'.

    For each column in `lag_features`, the function creates:
    - Lag features at the specified `lags`
    - Rolling mean features over the specified `roll_windows`

    The original columns are dropped after feature creation.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least 'city' and 'week_start_date'.
        lag_features (list[str]): Column names to create lag and rolling features from.
        lags (list[int]): Lag periods to apply.
        roll_windows (list[int]): Window sizes for computing rolling means.

    Returns:
        pd.DataFrame: DataFrame with new lag and rolling mean features added.
    """
    # Sort to ensure correct temporal order within each city
    df = df.sort_values(["city", "week_start_date"])

    for col in lag_features:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)
        for window in roll_windows:
            df[f"{col}_rollmean{window}"] = df.groupby("city")[col].transform(
                lambda x: x.rolling(window).mean()
            )

    # Optional sort to restore desired order (descending city, ascending date)
    df = df.sort_values(by=["city", "week_start_date"], ascending=[False, True])

    # Remove original columns to avoid leakage or redundancy
    df.drop(lag_features, axis=1, inplace=True)

    return df


def add_favourable_conditions(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds binary indicators for whether temperature conditions fall within specified favourable ranges.

    For each defined condition label, and for each lag and temperature source, the function checks
    whether the temperature is within the [min_temp, max_temp] interval. A new binary column is added
    with 1 if the condition is met, otherwise 0.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing lagged temperature columns.
        params (dict): Dictionary with:
            - "conditions": dict[label -> {"lags": [...], "min_temp": float, "max_temp": float}]
            - "temperature_sources": dict[str -> str], where keys are string templates with `{}` for lag

    Returns:
        pd.DataFrame: DataFrame with additional binary columns indicating favourable conditions.
    """
    conditions = params["conditions"]
    temperature_sources = params["temperature_sources"]

    for label, cfg in conditions.items():
        for lag in cfg["lags"]:
            for col_pattern, suffix in temperature_sources.items():
                col_name = col_pattern.format(
                    lag
                )  # e.g., "temperature_lag{}" -> "temperature_lag1"
                new_col = f"{label}_{lag}{suffix}"
                # 1 if temp is within [min_temp, max_temp], else 0
                df[new_col] = (
                    (df[col_name] >= cfg["min_temp"])
                    & (df[col_name] <= cfg["max_temp"])
                ).astype(int)

    return df


def split_data(df: pd.DataFrame):
    """
    Splits the input DataFrame into training, testing, and validation sets.

    The function separates the rows with missing target (`total_cases`) as the validation set.
    The remaining labeled data is split into training and test sets (80/20 split).

    Parameters:
        df (pd.DataFrame): Input DataFrame that must contain a 'total_cases' column.

    Returns:
        X_train (pd.DataFrame): Features for training.
        X_test (pd.DataFrame): Features for testing.
        y_train (pd.Series): Target values for training.
        y_test (pd.Series): Target values for testing.
        validation_data (pd.DataFrame): Unlabeled data to use for validation (without 'total_cases').
    """
    # Separate validation set: rows where target is missing
    validation_data = df[df["total_cases"].isna()].drop("total_cases", axis=1)

    # Keep labeled data: rows with target present
    labeled_df = df[df["total_cases"].notna()]

    # Split into features and target
    X = labeled_df.drop("total_cases", axis=1)
    y = labeled_df["total_cases"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, validation_data


def impute_with_mean(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in all numeric columns except 'total_cases' using column-wise means.

    Parameters:
        merged_df (pd.DataFrame): Input DataFrame potentially containing missing values.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with missing values imputed.
    """
    df_copy = merged_df.copy()

    for col in df_copy.columns:
        # Skip the target column and non-numeric columns
        if col != "total_cases" and df_copy[col].dtype in [float, int]:
            mean_value = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_value)

    return df_copy
