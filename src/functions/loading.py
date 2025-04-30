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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, sep=sep)