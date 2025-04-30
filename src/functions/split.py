# split into train, validation and test sets
# validation = all without target values
# train: 80% of remaining
# test: 20% of remaining


import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df:pd.DataFrame):

    """
    This fucntion accepts the complete data and split into validation and train and test dataset
    df (pd.DataFrame): The input DataFrame.
   train_df, test_df,validation_df  (pd.DataFrame): The  output retrun the train ,test,validation dataframe.

    """

    # 1. Separate validation set (rows with missing target)
    validation_df = df[df['total_cases'].isna()]

    # 2. Keep only rows with available target values
    labeled_df = df[df['total_cases'].notna()]

    # 3. Split labeled data into train (80%) and test (20%)
    train_df, test_df = train_test_split(labeled_df, test_size=0.2, random_state=42)

    return train_df, test_df,validation_df
