import pandas as pd

def make_prediction(df: pd.DataFrame, model)->pd.Series:
    """
    This fucntion accepts the train/test data and make the prediction based on the model 
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    model (Data model): The model used for prediction

    Returns:output dataframe(pd.DataFrame)
    """
    X=df
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name='total_cases')