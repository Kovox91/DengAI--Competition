# src/proj/pipeline/master_pipeline.py
from kedro.pipeline import Pipeline
from .preprocessing.pipeline import create_preprocessing_pipeline
from .engineering.pipeline import create_engineering_pipeline
from .modelling.pipeline import create_modelling_pipeline
from .reporting.pipeline import create_reporting_pipeline


def create_pipeline():
    return Pipeline(
        [
            *create_preprocessing_pipeline().nodes,
            *create_engineering_pipeline().nodes,
            *create_modelling_pipeline().nodes,
            *create_reporting_pipeline().nodes,
        ]
    )
