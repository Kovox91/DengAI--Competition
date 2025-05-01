"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:train_model"],
                outputs="model",
            ),
            node(func=test_model, inputs=["model", "X_test", "y_test"], outputs=None),
            node(func=remerge, inputs=["X_train", "X_test"], outputs="X"),
            node(func=remerge, inputs=["y_train", "y_test"], outputs="y"),
            node(
                func=train_model,
                inputs=["X", "y", "params:train_model"],
                outputs="final_model",
            ),
        ]
    )
