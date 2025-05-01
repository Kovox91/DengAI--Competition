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
                input="imputed_data",
                output="numeric_data",
            )
        ]
    )
