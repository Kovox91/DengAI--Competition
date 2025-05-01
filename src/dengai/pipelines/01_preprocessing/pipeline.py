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
                func=impute_interpolation,
                inputs="merged_data",
                outputs="imputed_data",
            ),
        ]
    )
