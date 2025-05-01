"""
This is a boilerplate pipeline 'engineering'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=removal_nonnumeric_columns,
                inputs="imputed_data",
                outputs="numeric_data",
            ),
            node(
                func=split_data,
                inputs="numeric_data",
                outputs=["X_train", "X_test", "y_train", "y_test", "validation_data"],
            ),
        ]
    )
