# filepath: utils/dataframe.py
import re
from typing import List

import numpy as np
import pandas as pd


def replace_value_with_nan(df: pd.DataFrame, value=-2147483648) -> pd.DataFrame:
    """
    Replace a specific value with np.nan in a DataFrame if the value exists.

    Parameters:
    df (pd.DataFrame): The DataFrame in which to replace the value.
    value (int): The value to replace. Defaults to -2147483648.

    Returns:
    pd.DataFrame: The DataFrame with the value replaced.
    """
    if (df.values == value).any():
        df.replace(value, np.nan, inplace=True)
    return df


def get_features(df: pd.DataFrame) -> List[str]:
    """
    Get a list of features i.e., all columns except the last column.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    list: The list of features.
    """
    # Return all columns except the last one
    features = df.columns.tolist()[:-1]
    print(features)
    return features


def sanitize_column_names_for_table(table: pd.DataFrame) -> pd.DataFrame:
    # Sanitize column names
    table.columns = [re.sub("[^0-9a-zA-Z_]", "", col) for col in table.columns]
    table.columns = ["col_" + col if col[0].isdigit() else col for col in table.columns]
    return table
