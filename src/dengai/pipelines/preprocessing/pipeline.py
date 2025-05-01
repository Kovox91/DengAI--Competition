from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_dataframes,
                inputs=[
                    "dengue_features_train",
                    "dengue_labels_train",
                    "dengue_features_test",
                ],
                outputs="merged_data",
            ),
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
                outputs="imputed_data",
            ),
        ]
    )
