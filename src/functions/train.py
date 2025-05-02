import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sktime.regression.interval_based import TimeSeriesForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pdb


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    learning_rate=0.05,
    num_leaves=50,
    max_depth=-1,
    n_estimators=1000,
    split_file="../../data/04_additions/forced_splits.json",
):
    """
    Trains a linear regression model on the training data.
    Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data target variable.
    Returns:
        model (LGBMRegressor): The trained linear regression model.
    """

    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=100, learning_rate=0.1
    )

    model.fit(X_train, y_train)

    return model
