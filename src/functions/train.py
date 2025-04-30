# fits a basic linear regression on training data
import pandas as pd
from sklearn.linear_model import LinearRegression
import pdb

def train_model(X_train:pd.DataFrame, y_train:pd.DataFrame):
    """
    Trains a linear regression model on the training data.
    Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data target variable.
    Returns:
        model (LinearRegression): The trained linear regression model.
    """

    # initialize the model
    model = LinearRegression(fit_intercept=False)

    model.fit(X_train, y_train)

    return model