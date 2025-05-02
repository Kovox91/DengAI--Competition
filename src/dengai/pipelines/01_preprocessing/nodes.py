import pandas as pd
import numpy as np


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


def impute_interpolation(df: pd.DataFrame, except_col: list[str] = ["total_cases"]):
    """Imputes missing values in a DataFrame using linear interpolation, excluding specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing missing values to be imputed.
        except_col (list[str]): List of column names to exclude from interpolation.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with missing values linearly interpolated,
                      excluding the specified columns.
    """
    df_interpolated = df.copy()
    df_interpolated.loc[:, df.columns.difference(["total_cases"])] = df[
        df.columns.difference(except_col)
    ].interpolate(method="linear")
    return df_interpolated
