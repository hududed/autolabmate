# filepath: utils/io.py
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def get_user_inputs(df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple:
    # user input batch number through number_input, default to 1
    batch_number = st.number_input("Enter batch number", min_value=1, value=1, step=1)

    # optimization_type here remains from the upload page ("single" or "multi")
    optimization_type = metadata["optimization_type"]

    # Get output column names and directions from metadata
    output_column_names = metadata["output_column_names"]
    directions = metadata["directions"]

    # Get number of parameters
    num_parameters = len(df.columns) - len(output_column_names)

    # Get number of random lines
    num_random_lines = st.number_input(
        "Enter the number of random lines",
        min_value=1,
        max_value=len(df),
        value=len(df),
    )

    # Build parameter_info from dataframe dtypes using a mapping
    parameter_info = df.dtypes[: -len(output_column_names)].to_dict()
    dtype_mapping = {"int64": "integer", "float64": "float", "O": "object"}
    parameter_info = {
        k: dtype_mapping.get(v.name, v.name) for k, v in parameter_info.items()
    }

    # Get parameter ranges and rounding values
    parameter_ranges = {}
    to_nearest_values = {}

    # Use the new learner_choice field (defaulting to "Random Forest" if not set)
    learner_choice = metadata.get("learner_choice", "Random Forest")

    for column in df.columns[: -len(output_column_names)]:
        if np.issubdtype(df[column].dtype, np.number):
            min_value = st.number_input(
                f"Enter the min value for {column}", value=df[column].min()
            )
            max_value = st.number_input(
                f"Enter the max value for {column}", value=df[column].max()
            )
            parameter_ranges[column] = (min_value, max_value)

            # Get rounding precision for numeric parameters
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
            # If learner_choice is Gaussian Process, categorical parameters are not supported.
            if learner_choice == "Gaussian Process":
                st.error(
                    f"Column '{column}' is categorical and is not supported with Gaussian Process. Please remove or convert this column."
                )
                # Mark unsupported column by setting its range to None.
                parameter_ranges[column] = None
                continue
            else:
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
    learner_choice: str = "Random Forest",
) -> List[str]:
    validation_errors = []

    # Subset df to only include the keys we gathered (plus outputs)
    if output_column_names is not None:
        df = df[list(parameter_ranges.keys()) + output_column_names]

    # Validate numeric parameters
    for column, range_values in parameter_ranges.items():
        if range_values is None:
            # This indicates an unsupported (categorical) column for Gaussian Process.
            continue  # We'll flag it below.
        if np.issubdtype(df[column].dtype, np.number):
            min_value, max_value = range_values
            if not min_value <= df[column].max() <= max_value:
                validation_errors.append(
                    f"Values for {column} are not within the specified range."
                )

    # Validate string parameters (for categorical data)
    for column, range_values in parameter_ranges.items():
        if range_values is None:
            continue  # Already flagged as unsupported.
        if np.issubdtype(df[column].dtype, object):
            unique_values = df[column].unique()
            if not set(unique_values).issubset(set(range_values)):
                validation_errors.append(
                    f"Unique values for {column} are not within the specified categories."
                )

    # If using Gaussian Process, categorical parameters are not allowed.
    if learner_choice == "Gaussian Process":
        for column, range_values in parameter_ranges.items():
            if range_values is None or isinstance(range_values, list):
                validation_errors.append(
                    f"Parameter '{column}' is categorical, but Gaussian Process only supports numeric parameters."
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
    # Set the learner based on the new learner_choice field
    learner_choice = metadata.get("learner_choice", "Random Forest")
    learner = "default_gp" if learner_choice == "Gaussian Process" else "default_rf"
    metadata.update(
        {
            "seed": seed,
            "batch_number": batch_number,
            "table_name": table_name,
            "optimization_type": optimization_type,  # remains "single" or "multi"
            "batch_type": "batch",
            "output_column_names": output_column_names,
            "num_parameters": num_parameters,
            "num_random_lines": num_random_lines,
            "parameter_info": parameter_info,
            "parameter_ranges": parameter_ranges,
            "learner": learner,
            "acquisition_function": "ei",
            "directions": directions,
            "bucket_name": bucket_name,
            "user_id": user_id,
            "to_nearest": to_nearest,
            "learner_choice": learner_choice,
        }
    )
    display_metadata = {
        k: v for k, v in metadata.items() if k not in ["user_id", "bucket_name"]
    }
    with st.expander("Show metadata", expanded=False):
        st.write(display_metadata)
    return metadata


def sanitize_table_name(table_name: str) -> str:
    sanitized_name = table_name.replace(" ", "-")
    sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "", sanitized_name)
    return sanitized_name


def generate_timestamps():
    now = datetime.now()
    filename_timestamp = now.strftime("%Y%m%d-%H%M")
    display_timestamp = now.strftime("%b %d %Y %I:%M%p")
    return filename_timestamp, display_timestamp
