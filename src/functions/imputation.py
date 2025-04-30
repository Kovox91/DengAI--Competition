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
        if col != target_column and df_copy[col].dtype in [float, int]:
            mean_value = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_value)
    
    return df_copy