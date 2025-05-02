from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=add_cyclical_features,
                inputs="imputed_data_interpol",
                outputs="cyclical_imputed",
            ),
            node(
                func=add_lag_and_rolling_features,
                inputs=dict(
                    df="cyclical_imputed",
                    lag_features="params:lag_features",
                    lags="params:lags",
                    roll_windows="params:roll_windows",
                ),
                outputs="lag_features_added",
            ),
            node(
                func=removal_nonnumeric_columns,
                inputs="imputed_data_mean",
                outputs="numeric_data",
            ),
            node(
                func=split_data,
                inputs="numeric_data",
                outputs=["X_train", "X_test", "y_train", "y_test", "validation_data"],
            ),
        ]
    )
