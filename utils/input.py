import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


def get_user_inputs(df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple:
    # user input batch number through number_input, default to 1
    batch_number = st.number_input("Enter batch number", min_value=1, value=1, step=1)

    # print(metadata)

    # Get optimization type with default from metadata
    optimization_type = metadata["optimization_type"]

    # Get output column names and directions from metadata
    output_column_names = metadata["output_column_names"]
    directions = metadata["directions"]

    # print(df)

    # Get number of parameters
    num_parameters = len(df.columns) - len(output_column_names)

    # Get number of random lines
    num_random_lines = st.number_input(
        "Enter the number of random lines",
        min_value=1,
        max_value=len(df),
        value=len(df),
    )

    parameter_info = df.dtypes[: -len(output_column_names)].to_dict()
    # Define a mapping from pandas dtypes to your desired types
    dtype_mapping = {"int64": "integer", "float64": "float", "O": "object"}

    # Iterate over the items in the dictionary and replace the dtypes
    parameter_info = {
        k: dtype_mapping.get(v.name, v.name) for k, v in parameter_info.items()
    }

    # Get parameter ranges
    parameter_ranges = {}
    to_nearest_values = {}

    for column in df.columns[: -len(output_column_names)]:
        if np.issubdtype(df[column].dtype, np.number):
            min_value = st.number_input(
                f"Enter the min value for {column}", value=df[column].min()
            )
            max_value = st.number_input(
                f"Enter the max value for {column}", value=df[column].max()
            )
            parameter_ranges[column] = (min_value, max_value)

            # Get to_nearest_value for numeric parameters
            if np.issubdtype(df[column].dtype, np.integer):
                to_nearest_value = st.number_input(
                    f"Enter the value to round {column} to",
                    min_value=1,
                    value=1,
                    step=1,
                )
                to_nearest_values[column] = to_nearest_value
            elif np.issubdtype(df[column].dtype, np.floating):
                to_nearest_value = st.number_input(
                    f"Enter the value to round {column} to",
                    min_value=0.01,
                    value=0.01,
                    step=0.01,
                )
                to_nearest_values[column] = to_nearest_value

        elif np.issubdtype(df[column].dtype, object):
            categories = st.text_input(
                f"Enter the categories for {column}",
                value=", ".join(df[column].unique()),
            )
            parameter_ranges[column] = categories.split(", ")

    return (
        batch_number,
        optimization_type,
        output_column_names,
        num_parameters,
        num_random_lines,
        parameter_info,
        parameter_ranges,
        directions,
        to_nearest_values,
    )


def validate_inputs(
    df: pd.DataFrame,
    parameter_ranges: Dict[str, Any],
    output_column_names: list[str] = None,
) -> List[str]:
    validation_errors = []

    if output_column_names is not None:
        df = df[list(parameter_ranges.keys()) + output_column_names]

    # Validate numeric parameters
    for column, range_values in parameter_ranges.items():
        if np.issubdtype(df[column].dtype, np.number):
            min_value, max_value = range_values
            if not min_value <= df[column].max() <= max_value:
                validation_errors.append(
                    f"Values for {column} are not within the specified range."
                )

    # Validate string parameters
    for column, categories in parameter_ranges.items():
        if np.issubdtype(df[column].dtype, object):
            unique_values = df[column].unique()
            if not set(unique_values).issubset(set(categories)):
                validation_errors.append(
                    f"Unique values for {column} are not within the specified categories."
                )

    return validation_errors


def display_dictionary(
    seed: int,
    batch_number: int,
    table_name: str,
    optimization_type: str,
    output_column_names: list[str],
    num_parameters: int,
    num_random_lines: int,
    parameter_info: Dict[str, Any],
    parameter_ranges: tuple,
    directions: Dict[str, Any],
    user_id: str,
    to_nearest: float,
    metadata: Dict[str, Any],
    bucket_name: str = "test-bucket",
) -> Dict[str, Any]:
    metadata.update(
        {
            "seed": seed,
            "batch_number": batch_number,
            "table_name": table_name,
            "optimization_type": optimization_type,
            "batch_type": "batch",
            "output_column_names": output_column_names,
            "num_parameters": num_parameters,
            "num_random_lines": num_random_lines,
            "parameter_info": parameter_info,
            "parameter_ranges": parameter_ranges,
            "learner": "regr.ranger",
            "acquisition_function": "ei",
            "directions": directions,
            "bucket_name": bucket_name,
            "user_id": user_id,
            "to_nearest": to_nearest,
        }
    )
    display_metadata = {
        k: v for k, v in metadata.items() if k != "user_id" and k != "bucket_name"
    }
    with st.expander("Show metadata", expanded=False):
        st.write(display_metadata)
    return metadata
