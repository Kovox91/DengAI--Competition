import pandas as pd

def merge_dataframes(train_features:pd.DataFrame, train_labels:pd.DataFrame, test:pd.DataFrame on:str):
    """
    Merge two DataFrames on common columns.

    Args:
        df1, df2 : The two DataFrames to merge.
        on (list): Column(s) to merge on.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """    
    train = pd.merge(train_features, train_labels, on=on)
    return pd.concat([train, test], axis=0)