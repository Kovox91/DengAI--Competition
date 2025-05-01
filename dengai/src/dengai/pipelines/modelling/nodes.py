import pandas as pd
from lightgbm import LGBMRegressor
from datetime import datetime
from sklearn.metrics import mean_absolute_error


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    """
    Trains a linear regression model on the training data.
    Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data target variable.
    Returns:
        model (LGBMRegressor): The trained linear regression model.
    """

    # initialize the model
    model = LGBMRegressor(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        random_state=42,
        max_depth=params["max_depth"],
        verbose=-1,
        forcedsplits_filename=params["split_file"],
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
    print(f"Mean absolute Error is {MAE}")
    log_to_csv("logs/MAEs.csv", MAE)

    return


def remerge(train, test):
    """
    Concatenate train adn test datasets for final training.

    Args:
        train, test (pd.DataFrame): The input DataFrames.

    Returns:
        pd.DataFrame: A new concatenated DataFrame.
    """
    return pd.concat([train, test])


def make_predictions(model: object, validation: pd.DataFrame):
    """Predicts the target variable with the retrained model

    Args:
        model (object): A trained regression model
        validation (pd.DataFrame): A prepared validation DataFrame

    Returns:
        _type_: _description_
    """
    return model.predict(validation)


def create_submission(predictions: pd.DataFrame, validation_data: pd.DataFrame):
    """
    Creates a submission file for the competition.
    Args:
        predictions (pd.DataFrame): The predictions made by the model.
        validation_data (pd.DataFrame): The validation data used to make the predictions.
    """
    # Create a submission DataFrame
    submission_df = pd.DataFrame(
        {
            "city": validation_data["city"],
            "year": validation_data["year"],
            "weekofyear": validation_data["weekofyear"],
            "total_cases": predictions,
        }
    )

    submission_df["city"] = submission_df["city"].replace({1: "sj", 2: "iq"})

    # transform total cases float to int
    submission_df["total_cases"] = submission_df["total_cases"].astype(int)

    return submission_df
