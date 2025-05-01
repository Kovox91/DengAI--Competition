import pandas as pd


def lagging_temp(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # first sort the column with specific order
    df = df.sort_values(by=["city", "year", "weekofyear"])
    list_columns = [
        "reanalysis_air_temp_k",
        "reanalysis_avg_temp_k",
        "reanalysis_dew_point_temp_k",
        "reanalysis_relative_humidity_percent",
    ]
    # Lagging the all slected  columns for each city by 1 period (1 )

    for col in list_columns:
        df[col + "_lag_1"] = df.groupby(["city"])[col].shift(1).fillna(0)
        df[col + "_lag_2"] = df.groupby(["city"])[col].shift(2).fillna(0)
        df[col + "_lag_3"] = df.groupby(["city"])[col].shift(3).fillna(0)
        df[col + "_lag_4"] = df.groupby(["city"])[col].shift(4).fillna(0)

    # df["reanalysis_air_temp_k_lagged"] = (
    # df.groupby(["city"])["reanalysis_air_temp_k"].shift(1).fillna(0)
    # )

    return df
