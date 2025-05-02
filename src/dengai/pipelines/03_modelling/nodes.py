import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import csv


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    """
    Trains an XGBoost regression model on the provided training data.

    Parameters:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.DataFrame): Target values for training.
        params (dict): Hyperparameters for the XGBoost model (currently unused).

    Returns:
        model (XGBRegressor): Trained XGBoost regression model.
    """
    # Initialize the model with fixed hyperparameters
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=100, learning_rate=0.1
    )

    # Fit the model to training data
    model.fit(X_train, y_train)

    return model


def log_to_csv(filepath: str, value: str) -> None:
    """
    Logs a value to a CSV file with the current timestamp.

    Parameters:
        filepath (str): Path to the CSV file where the log entry will be saved.
        value (str): The value to log.

    Returns:
        None
    """
    # Open the file in append mode and write the timestamped value
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), value])

    return


def test_model(model: object, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Evaluates the performance of a model based on the Mean Absolute Error (MAE).

    Parameters:
        model (object): The trained model to evaluate.
        X_test (pd.DataFrame): Feature matrix for the test data.
        y_test (pd.DataFrame): True target values for the test data.

    Returns:
        None: Logs MAE to a CSV file and prints it.
    """
    test_predict = model.predict(X_test)

    MAE = mean_absolute_error(y_test, test_predict)

    log_to_csv("data/06_model_output/mean_absolute_errors.csv", MAE)


def remerge(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate train and test datasets for final training.

    Parameters:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    return pd.concat([train, test])
