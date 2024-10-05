import streamlit as st
from typing import List, Tuple, Union


def get_parameter_info() -> Tuple[List[str], List[str]]:
    series = st.number_input("Number of parameters:", value=3, key="num_params")
    param_names = []
    param_types = []
    for i in range(series):
        param_name = st.text_input(
            f"Parameter {i+1} name:", f"param{i+1}", key=f"param_name_{i}"
        )
        param_type = st.selectbox(
            f"Parameter {i+1} type:",
            ["Integer", "Float", "Categorical"],
            key=f"param_type_{i}",
        )
        param_names.append(param_name)
        param_types.append(param_type)
    return param_names, param_types


def get_parameter_ranges(
    param_names: List[str], param_types: List[str]
) -> Tuple[List[Union[Tuple[int, int], Tuple[float, float], List[str]]], List[float]]:
    param_ranges = []
    param_intervals = []
    for i in range(len(param_names)):
        if param_types[i] == "Integer":
            to_nearest = st.number_input(
                f"Enter the interval for {param_names[i]} (must be > 0):",
                value=1.0,
                min_value=1.0,
                step=1.0,
                format="%.0f",
                key=f"to_nearest_{i}",
            )
            to_nearest = int(to_nearest)  # Ensure to_nearest is an integer
            min_val = st.number_input(
                f"Minimum value for {param_names[i]}:",
                value=0,
                step=to_nearest,
                key=f"min_val_{i}",
            )
            max_val = st.number_input(
                f"Maximum value for {param_names[i]}:",
                value=100,
                step=to_nearest,
                key=f"max_val_{i}",
            )
            param_ranges.append((min_val, max_val))
            param_intervals.append(to_nearest)
        elif param_types[i] == "Float":
            to_nearest = st.number_input(
                f"Enter the interval for {param_names[i]} (must be > 0):",
                value=0.1,
                min_value=0.01,
                format="%.2f",
                key=f"to_nearest_{i}",
            )
            min_val = st.number_input(
                f"Minimum value for {param_names[i]}:",
                value=0.0,
                step=to_nearest,
                format="%.2f",
                key=f"min_val_{i}",
            )
            max_val = st.number_input(
                f"Maximum value for {param_names[i]}:",
                value=100.0,
                step=to_nearest,
                format="%.2f",
                key=f"max_val_{i}",
            )
            param_ranges.append((min_val, max_val))
            param_intervals.append(to_nearest)
        else:
            categories = st.text_input(
                f"Enter categories for {param_names[i]} (comma-separated):",
                "cat1,cat2,cat3",
                key=f"categories_{i}",
            )
            categories = [cat.strip() for cat in categories.split(",")]
            param_ranges.append(categories)
            param_intervals.append(None)
    return param_ranges, param_intervals
