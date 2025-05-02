from kedro.pipeline import node, Pipeline, pipeline
from .nodes import (
    add_cyclical_features,
    add_lag_and_rolling_features,
    impute_with_mean,
    removal_nonnumeric_columns,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=add_cyclical_features,
                inputs="merged_data",
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
                func=impute_with_mean,
                inputs="lag_features_added",
                outputs="imputed_data_mean",
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
