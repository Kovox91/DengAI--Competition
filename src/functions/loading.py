import pandas as pd
import os

def load_csv(file_name, sep=","):
    """
    Loads a CSV file and returns a pandas DataFrame.

    Args:
        file_path (str): Relative or absolute path to the CSV file.
        **kwargs: Additional arguments to pass to pd.read_csv()

    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")
    
    return pd.read_csv(file_name, sep=sep)