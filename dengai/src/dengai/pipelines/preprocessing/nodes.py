import pandas as pd


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
