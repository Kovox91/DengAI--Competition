import pandas as pd

def remerge(train, test):
    """
    Concatenate train adn test datasets for final training.

    Args:
        train, test (pd.DataFrame): The input DataFrames.

    Returns:
        pd.DataFrame: A new concatenated DataFrame.
    """
    return pd.concat([train,test])