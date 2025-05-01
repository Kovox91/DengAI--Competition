from functions.loading import load_csv
from functions.merging import merge_dataframes
from functions.imputation import (
    impute_with_mean,
    add_cyclical_features,
    add_lag_and_rolling_features,
    create_favorable_temp_cols,
)
from functions.remerge import remerge
from functions.train import train_model
from functions.test import test_model, get_min_from_csv
from functions.submission import create_submission
from functions.encoding import removal_nonnumeric_columns
from functions.split import split_data
from functions.predict import make_prediction
import random
import pdb

random.seed(42)


def main():

    # read data
    train_features = load_csv("../data/01_raw/dengue_features_train.csv")
    train_labels = load_csv("../data/01_raw/dengue_labels_train.csv")
    test_features = load_csv("../data/01_raw/dengue_features_test.csv")

    df = merge_dataframes(
        train_features, train_labels, test_features, on=["city", "weekofyear", "year"]
    )
    df = add_cyclical_features(df)

    df = add_lag_and_rolling_features(df)
    df = create_favorable_temp_cols(df)
    df = impute_with_mean(df, "total_cases")

    df = removal_nonnumeric_columns(df)
    train_X, test_X, train_y, test_y, validation = split_data(df)
    train_X, test_X, train_y, test_y, validation = split_data(df)

    model = train_model(train_X, train_y)
    best_MAE = get_min_from_csv("logs/MAEs.csv")
    best_MAE = get_min_from_csv("logs/MAEs.csv")

    current_MAE = test_model(model, test_X, test_y)

    if current_MAE >= best_MAE:
        print("Model did not perform better than previous models. Aborting.")
        print("Model did not perform better than previous models. Aborting.")

    else:
        # CHECK IF MAE IS GOOD ENOUGH
        X = remerge(train_X, test_X)
        y = remerge(train_y, test_y)

        final_model = train_model(X, y)

        predictions = make_prediction(validation, final_model)

        create_submission(
            predictions, validation, "../data/03_submissions/naive_model_30_04_1314.csv"
        )
        create_submission(
            predictions, validation, "../data/03_submissions/naive_model_30_04_1314.csv"
        )


if __name__ == "__main__":
    main()
