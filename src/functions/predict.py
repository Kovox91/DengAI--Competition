# predicts cases on validation set
import pandas as pd

def make_prediction(df: pd.DataFrame, model)=>pd.series:
    """
    This fucntion accepts the train/test data and make the prediction based on the model 
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    model (Data model): The model used for prediction

    Returns:output dataframe(pd.DataFrame)
    """
    """
    Predict using a model and a subset of features from the DataFrame.

    Args:
        model: Trained model with a .predict() method (e.g., sklearn model).
        df (pd.DataFrame): Input data.
        feature_columns (list): List of column names to use as features.

    Returns:
        pd.Series: Model predictions.
    """
    # Ensure all required columns are in the DataFrame
    #missing = set(feature_columns) - set(df.columns)
    #if missing:
       # raise ValueError(f"Missing required columns: {missing}")

    #X = df[feature_columns]
    X=df
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name='Prediction')