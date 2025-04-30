import pandas as pd
import lightgbm as lgb

def train_model(X_train:pd.DataFrame, y_train:pd.DataFrame, learning_rate=.09, max_depth=15, num_leaves=10):
    """
    Trains a linear regression model on the training data.
    Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data target variable.
    Returns:
        model (RegLGBMClassifier): The trained linear regression model.
    """
    # initialize the model
    model = lgb.LGBMClassifier(learning_rate=learning_rate,max_depth=max_depth,num_leaves=num_leaves,random_state=42)

    # fit
    model.fit(X_train,y_train)

    return model