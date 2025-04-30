import pandas as pd
from sklearn.metrics import mean_absolute_error

def test_model(model:object, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """
    Evaluate a models performance based on the mean absolute error.
    Args:
        model (object): The trained model to evaluate.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.DataFrame): The test data target variable.
    Returns:
        MAE (float): The mean absolute error of the model predictions.
    """
    test_predict = model.predict(X_test)
    MAE = mean_absolute_error(y_test, test_predict)
    return MAE
    