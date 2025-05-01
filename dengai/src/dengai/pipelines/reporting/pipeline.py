"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=make_predictions,
                inputs=["final_model", "validation_data"],
                outputs="prediction",
            ),
            node(
                func=create_submission,
                inputs=["prediction", "validation_data"],
                outputs="submission",
            ),
        ]
    )
