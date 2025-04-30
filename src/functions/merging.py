import pandas as pd

def merge(df1, df2, on):
    """
    Merge two DataFrames on common columns.

    Args:
        df1, df2 : The two DataFrames to merge.
        on (list): Column(s) to merge on.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """    
    return pd.merge(df1, df2, on=on)