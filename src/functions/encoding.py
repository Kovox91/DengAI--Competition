# removes all non-numeric columns Except the city column

import pandas as pd

def removal_nonnumeric_columns(df:pd.DataFrame)-> pd.DataFrame:
    """
    This fucntion removes the non- numeric (except city)columns as not required.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:output dataframe
    """
    # Column to keep (even if non-numeric)
    city_column = 'city'

    df['city'] = df['city'].replace({'sj': 1, 'iq': 2})

    # Filter numeric columns and add the keep_column
    numeric_df = df.select_dtypes(include='number')  # keeps only numeric columns
    if city_column in df.columns:
        numeric_df[city_column] = df[city_column]
    
    return numeric_df

