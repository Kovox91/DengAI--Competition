import pandas as pd
from sklearn.metrics import mean_absolute_error
import csv
from datetime import datetime


def log_to_csv(filepath, value):
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), value])


def get_min_from_csv(filepath):
    values = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty rows
                values.append(float(row[1]))  # assuming value is in 2nd column
    return min(values) if values else None


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
    print(f"Mean absolute Error is {MAE}")
    # log_to_csv("logs/MAEs.csv", MAE)

    return MAE
