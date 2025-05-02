import pandas as pd
import numpy as np
from typing import List
import pdb


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


def impute_interpolation(df: pd.DataFrame):
    df_interpolated = df.copy()
    df_interpolated.loc[:, df.columns.difference(["total_cases"])] = df[
        df.columns.difference(["total_cases"])
    ].interpolate(method="linear")
    return df_interpolated
