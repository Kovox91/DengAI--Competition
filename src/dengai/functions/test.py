import pandas as pd
import csv
from datetime import datetime


def get_min_from_csv(filepath):
    values = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty rows
                values.append(float(row[1]))  # assuming value is in 2nd column
    return min(values) if values else None
