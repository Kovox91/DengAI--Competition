import pandas as pd


def make_predictions(model: object, validation: pd.DataFrame) -> pd.Series:
    """
    Predicts the target variable using a trained model on the validation data.

    Parameters:
        model (object): A trained regression model.
        validation (pd.DataFrame): The prepared validation dataset.

    Returns:
        pd.Series: The predicted values for the validation data.
    """
    preds = model.predict(validation)
    return pd.Series(preds)


def create_submission(predictions, validation_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a submission DataFrame for the competition.

    Parameters:
        predictions (pd.Series or pd.DataFrame): Model predictions for the validation data.
        validation_data (pd.DataFrame): The validation data used for making predictions.

    Returns:
        pd.DataFrame: A DataFrame formatted for submission.
    """
    # Flatten predictions if necessary
    if isinstance(predictions, (pd.Series, pd.DataFrame)):
        predictions = predictions.squeeze()

    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()

    # Ensure predictions length matches validation data
    if len(predictions) != len(validation_data):
        raise ValueError(
            f"Length of predictions ({len(predictions)}) does not match length of validation_data ({len(validation_data)})."
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "city": validation_data["city"].values,
            "year": validation_data["year"].values,
            "weekofyear": validation_data["weekofyear"].values,
            "total_cases": predictions,
        }
    )
    submission_df["city"] = submission_df["city"].replace({1: "sj", 2: "iq"})

    # Convert total_cases to integers
    submission_df["total_cases"] = submission_df["total_cases"].astype(int)

    return submission_df
