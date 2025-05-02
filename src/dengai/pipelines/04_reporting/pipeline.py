from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import make_predictions, create_submission


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
