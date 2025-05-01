import pandas as pd


def make_predictions(model: object, validation: pd.DataFrame):
    """Predicts the target variable with the retrained model

    Args:
        model (object): A trained regression model
        validation (pd.DataFrame): A prepared validation DataFrame

    Returns:
        _type_: _description_
    """
    preds = model.predict(validation)
    return pd.Series(preds)


def create_submission(predictions, validation_data: pd.DataFrame):
    """
    Creates a submission file for the competition.
    Args:
        predictions (pd.DataFrame): The predictions made by the model.
        validation_data (pd.DataFrame): The validation data used to make the predictions.
    """

    # Check if predictions is a pandas Series or DataFrame and squeeze it to remove extra dimensions
    if isinstance(predictions, (pd.Series, pd.DataFrame)):
        predictions = predictions.squeeze()

    # Ensure that predictions are a 1D array (flatten if it's a 2D array with 1 column)
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()

    # Ensure indices of validation_data and predictions align
    if len(predictions) != len(validation_data):
        raise ValueError(
            f"Length of predictions ({len(predictions)}) does not match length of validation_data ({len(validation_data)})."
        )

    # Create a submission DataFrame
    submission_df = pd.DataFrame(
        {
            "city": validation_data[
                "city"
            ].values,  # Ensure it's using the correct values
            "year": validation_data["year"].values,
            "weekofyear": validation_data["weekofyear"].values,
            "total_cases": predictions,
        }
    )
    submission_df["city"] = submission_df["city"].replace({1: "sj", 2: "iq"})

    # transform total cases float to int
    submission_df["total_cases"] = submission_df["total_cases"].astype(int)

    return submission_df
