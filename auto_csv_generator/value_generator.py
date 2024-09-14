from random import randrange, uniform
from pyDOE import lhs
import sobol_seq
from typing import List, Union, Tuple
from .utils import round_to_nearest, format_values


def generate_random_values(
    param_names: List[str],
    param_types: List[str],
    param_ranges: List[Union[Tuple[int, int], Tuple[float, float], List[str]]],
    param_intervals: List[float],
    nr_random_lines: int,
    decimal_places: int,
) -> List[List[Union[int, float, str]]]:
    param_values = []
    for _ in range(nr_random_lines):
        values = []
        for i in range(len(param_types)):
            if param_types[i] == "Integer":
                min_val, max_val = param_ranges[i]
                value = randrange(min_val, max_val + 1)
                value = round_to_nearest(value, param_intervals[i])
                value = int(value)
            elif param_types[i] == "Float":
                min_val, max_val = param_ranges[i]
                value = uniform(min_val, max_val)
                value = round_to_nearest(value, param_intervals[i])
            else:
                categories = param_ranges[i]
                value = categories[randrange(len(categories))]
            values.append(value)
        param_values.append(values)
    return format_values(param_values, decimal_places)


def generate_lhs_values(
    param_names: List[str],
    param_types: List[str],
    param_ranges: List[Union[Tuple[int, int], Tuple[float, float], List[str]]],
    param_intervals: List[float],
    nr_random_lines: int,
    decimal_places: int,
) -> List[List[Union[int, float, str]]]:
    samples = lhs(len(param_names), nr_random_lines)
    param_values = []
    for i in range(nr_random_lines):
        values = []
        for j in range(len(param_names)):
            if param_types[j] == "Integer":
                min_val, max_val = param_ranges[j]
                value = samples[i, j] * (max_val - min_val) + min_val
                value = round_to_nearest(value, param_intervals[j])
                value = int(value)
            elif param_types[j] == "Float":
                min_val, max_val = param_ranges[j]
                value = samples[i, j] * (max_val - min_val) + min_val
                value = round_to_nearest(value, param_intervals[j])
            else:
                categories = param_ranges[j]
                value = categories[int(samples[i, j] * (len(categories) - 1))]
            values.append(value)
        param_values.append(values)
    return format_values(param_values, decimal_places)


def generate_sobol_values(
    param_names: List[str],
    param_types: List[str],
    param_ranges: List[Union[Tuple[int, int], Tuple[float, float], List[str]]],
    param_intervals: List[float],
    nr_random_lines: int,
    decimal_places: int,
) -> List[List[Union[int, float, str]]]:
    samples = sobol_seq.i4_sobol_generate(len(param_names), nr_random_lines)
    param_values = []
    for i in range(nr_random_lines):
        values = []
        for j in range(len(param_names)):
            if param_types[j] == "Integer":
                min_val, max_val = param_ranges[j]
                value = samples[i, j] * (max_val - min_val) + min_val
                value = round_to_nearest(value, param_intervals[j])
                value = int(value)
            elif param_types[j] == "Float":
                min_val, max_val = param_ranges[j]
                value = samples[i, j] * (max_val - min_val) + min_val
                value = round_to_nearest(value, param_intervals[j])
            else:
                categories = param_ranges[j]
                value = categories[int(samples[i, j] * (len(categories) - 1))]
            values.append(value)
        param_values.append(values)
    return format_values(param_values, decimal_places)
