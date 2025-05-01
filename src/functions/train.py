import pandas as pd
from lightgbm import LGBMRegressor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=7,
    n_estimators=100,
):
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
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        random_state=42,
        verbose=-1,
    )

    # fit
    model.fit(X_train, y_train)

    return model
