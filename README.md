# DengAI - Mini Competition

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is our approach on the [DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/submissions/).  

Our main idea was to get a model as fast as possible and then iterate over features, models and the raw data.  
Our current approach uses an XGBoosted Decision Tree on some basic features elaborated below.

The Pipeline was created using `kedro 0.19.12`.

## Features and Imputations

The current iteration utilizes sine encoded time, lagged environmental data (temperatures, precipitation, humidity and vegetation) as well as a lagged rolling mean environmeltal data.  
Additionally, the time features (month, week and day) are sine encoded, surprisingly that performed better than sine and cosine encoding. 
Missing values are mean-imputed after the encoding step.

## Current score
![Alt text](fluff/IMG_20250502_053535.jpg)

Based on those we reached a score sub 25, the biggest improvement was the usage of XGB over LGBM.

## Tried and Failed

Here, a brief overview of what we tried so far and could either not improve the model or implement the idea.
- Include favourable and unvavourable development, growth and breeding conditions
    - The idea was to give the model access to information of the conditions for eggs larvae or parents of mosquitos that would have infected people with symptom (e.g. cases) in the current week. For that we took the lagged environmental features, binned them into good or bad for the respective file cycle stage and encoded them binary.  
    Worsened the Score
- Implement recursive forecasting.
    - One Observation was that mosquitos need to drink blood from an invected person to be able to infect another person. Combined with the incubation time of Dengue, our assumption was that the number of cases in the last two weeks would be a strong indicator for the current number of cases.  
    We were not yet able to implement this idea due to various reasons.


## Further ideas

There are more ideas we woul like to try but did not have the resources for:
- Create two independant models.
    - Currently we use one model for both cities wich is properly not ideal. During our experimentations with LGBM we forced the model to do the first split based on the city column, however this did not change the score leading us to believe that the model distinguished by city quite early on its own. Stil trying to work out two independant models might be worthwhile eventually.
- Improve outbreak predictions.
    - Right now the model performs good on regular seasonal case fluctuations but is bad at predicting massive outbreaks. There are several lines to take here our ideas included recursive predictions (see above) and two models, for baseline and outbreak prediction.
- Feature selection and fine-tuning
    - Since we are plainly creating new features, our final sets consists of more than 130, sometimes closely releated features, it might be worthwhile to be a bit more rigorous with selection or combining features to reduce overfitting.


## Project structure

```
.
├── conf
│   ├── base
│   │   ├── catalog.yml
│   │   ├── parameters_dengue_model.yml
│   │   ├── parameters_engineering.yml
│   │   ├── parameters_modelling.yml
│   │   ├── parameters_preprocessing.yml
│   │   ├── parameters_reporting.yml
│   │   └── parameters.yml
│   ├── local
│   ├── logging.yml
│   └── README.md
├── data
│   ├── 01_raw
│   │   ├── dengue_features_test.csv
│   │   ├── dengue_features_train.csv
│   │   ├── dengue_labels_train.csv
│   │   └── forced_splits.json
│   ├── 02_intermediate
│   │   ├── cyclical_imputed.parquet
│   │   ├── fav_temps_added.parquet
│   │   ├── imputed_data_interpol.parquet
│   │   ├── imputed_data_mean.parquet
│   │   ├── imputed_data.parquet
│   │   ├── lag_features_added.parquet
│   │   ├── merged_data.parquet
│   │   └── numeric_data.parquet
│   ├── 03_model_input
│   │   ├── validation_data.parquet
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   ├── 05_models
│   │   ├── final_model.pkl
│   │   └── model.pkl
│   ├── 06_model_output
│   │   ├── mean_absolute_errors.csv
│   │   └── prediction.csv
│   ├── 07_reporting
│   │   └── submission
│   │       └── shortened
│   └── baseline_my_pc.csv
├── docs
│   └── source
│       ├── conf.py
│       └── index.rst
├── fluff
│   └── IMG_20250502_053535.jpg
├── info.log
├── LICENSE
├── notebooks
│   ├── 250501_SM_binned_temperature.ipynb
│   ├── experiment1.ipynb
│   └── exploration.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   └── dengai
│       ├── __init__.py
│       ├── __main__.py
│       ├── pipeline_registry.py
│       ├── pipelines
│       │   ├── 01_preprocessing
│       │   │   ├── __init__.py
│       │   │   ├── nodes.py
│       │   │   ├── pipeline.py
│       │   ├── 02_engineering
│       │   │   ├── __init__.py
│       │   │   ├── nodes.py
│       │   │   ├── pipeline.py
│       │   ├── 03_modelling
│       │   │   ├── __init__.py
│       │   │   ├── nodes.py
│       │   │   ├── pipeline.py
│       │   │   └── __pycache__
│       │   ├── 04_reporting
│       │   │   ├── __init__.py
│       │   │   ├── nodes.py
│       │   │   ├── pipeline.py
│       │   │   └── __pycache__
│       │   ├── __init__.py
│       │   ├── master_pipeline.py
│       │   └── __pycache__
│       │       └── __init__.cpython-313.pyc
│       ├── __pycache__
│       └── settings.py
├── streamlit
│   └── app.py
├── struct.txt
└── tests
    ├── __init__.py
    ├── pipelines
    │   ├── dengue_model
    │   │   ├── __init__.py
    │   │   └── test_pipeline.py
    │   ├── engineering
    │   │   ├── __init__.py
    │   │   └── test_pipeline.py
    │   ├── __init__.py
    │   ├── modelling
    │   │   ├── __init__.py
    │   │   └── test_pipeline.py
    │   ├── preprocessing
    │   │   ├── __init__.py
    │   │   └── test_pipeline.py
    │   └── reporting
    │       ├── __init__.py
    │       └── test_pipeline.py
    └── test_run.py

53 directories, 118 files
```