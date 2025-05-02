import pandas as pd
from xgboost import XGBRegressor

# from lightgbm import LGBMRegressor
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import csv


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    """
    Trains a linear regression model on the training data.
    Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data target variable.
    Returns:
        model (XGBRegressor): The trained XGBoost regression model.
    """

    # initialize the model
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=100, learning_rate=0.1
    )

    # fit
    model.fit(X_train, y_train)

    return model


def log_to_csv(filepath, value):
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), value])
    return


def test_model(model: object, X_test: pd.DataFrame, y_test: pd.DataFrame):
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
    log_to_csv("data/06_model_output/mean_absolute_errors.csv", MAE)
    print(f"Mean absolute Error is {MAE}")

    return


def remerge(train, test):
    """
    Concatenate train and test datasets for final training.

    Args:
        train, test (pd.DataFrame): The input DataFrames.

    Returns:
        pd.DataFrame: A new concatenated DataFrame.
    """
    return pd.concat([train, test])
