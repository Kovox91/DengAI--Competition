import pandas as pd


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


import pandas as pd
from sklearn.model_selection import train_test_split
import pdb


def split_data(df: pd.DataFrame):
    """
     This fucntion accepts the complete data and split into validation and train and test dataset
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
