from functions.loading import load_csv
from functions.merging import merge_dataframes
from functions.imputation import impute_with_mean
from functions.remerge import remerge
from functions.train import train_model
from functions.test import test_model
from functions.submission import create_submission
from functions.encoding import removal_nonnumeric_columns
from functions.split import split_data
from functions.predict import make_prediction
import pdb

def main():

    # read data
    train_features = load_csv("../data/01_raw/dengue_features_train.csv")
    train_labels = load_csv("../data/01_raw/dengue_labels_train.csv")
    test_features = load_csv("../data/01_raw/dengue_features_test.csv")

    df = merge_dataframes(train_features, train_labels, test_features, on=["city", "weekofyear", "year"])

    df = impute_with_mean(df, "total_cases")

    df = removal_nonnumeric_columns(df)

    train, test, validation = split_data(df)

    train_X = train.drop("total_cases", axis=1)
    train_y = train["total_cases"]
    test_X = test.drop("total_cases", axis=1)
    test_y = test["total_cases"]

    model = train_model(train_X, train_y)

    MAE = test_model(model, test_X, test_y)

    # CHECK IF MAE IS GOOD ENOUGH
    X = remerge(train_X, test_X)
    y = remerge(train_y, test_y)

    final_model = train_model(X, y)

    predictions = make_prediction(validation, final_model)

    create_submission(predictions, validation, "../data/03_submissions/naive_model_30_04_1314.csv")


if __name__ == "__main__":
    main()