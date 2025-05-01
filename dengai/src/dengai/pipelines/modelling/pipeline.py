"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=train_model, input=["X_train", "y_train"], output="model"),
            node(func=test_model, input=["model", "X_test", "y_test"], output=None),
            node(func=remerge, input=["X_train", "X_test"], output="X"),
            node(func=remerge, input=["y_train", "y_test"], output="y"),
            node(func=train_model, input=["X, y"], output="final_model"),
            node(
                func=make_predictions,
                input=["final_model", "validation_data"],
                output="prediction",
            ),
            node(
                func=create_submission,
                input=["prediction", "validation_data"],
                output="submission",
            ),
        ]
    )
