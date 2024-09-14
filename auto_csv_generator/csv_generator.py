import csv
from random import randrange, uniform
import pandas as pd
import streamlit as st
from pyDOE import lhs
import sobol_seq
from typing import Callable, List, Tuple, Union
from .value_generator import (
    generate_random_values,
    generate_lhs_values,
    generate_sobol_values,
)
from decimal import Decimal, getcontext, ROUND_HALF_UP
from dataclasses import dataclass, field

# Set the precision high enough to handle your use case
getcontext().prec = 28

ParameterInfoCallable = Callable[[], Tuple[List[str], List[str]]]
ParameterRangesCallable = Callable[
    [List[str], List[str]],
    Tuple[List[Union[Tuple[int, int], Tuple[float, float], List[str]]], List[float]],
]
ValueGeneratorCallable = Callable[
    [
        List[str],
        List[str],
        List[Union[Tuple[int, int], Tuple[float, float], List[str]]],
        List[float],
        int,
        int,
    ],
    List[List[Union[int, float, str]]],
]
CSVWriterCallable = Callable[[List[str], List[List[Union[int, float, str]]]], None]
CSVDownloaderCallable = Callable[[], None]


@dataclass
class CSVGenerator:
    param_info_func: ParameterInfoCallable
    param_ranges_func: ParameterRangesCallable
    value_generator_func: ValueGeneratorCallable
    csv_writer_func: CSVWriterCallable
    csv_downloader_func: CSVDownloaderCallable
    series: None = None
    nr_random_lines: None = None
    param_names: List[str] = field(default_factory=list)
    param_types: List[str] = field(default_factory=list)
    final_col_name1: str = "output1"
    final_col_name2: str = "output2"
    optimization_type: None = None
    final_col_name: None = None
    param_ranges: List[tuple] = field(default_factory=list)
    param_intervals: List[float] = field(default_factory=list)
    param_values: List[List[float]] = field(default_factory=list)
    data_header: List[str] = field(default_factory=list)

    def get_randomization_type(self) -> None:
        self.randomization_type = st.selectbox(
            "Select randomization type:", ["Random", "Latin Hypercube", "Sobol"]
        )
        if self.randomization_type == "Random":
            self.value_generator_func = generate_random_values
        elif self.randomization_type == "Latin Hypercube":
            self.value_generator_func = generate_lhs_values
        elif self.randomization_type == "Sobol":
            self.value_generator_func = generate_sobol_values

    def get_input_values(self) -> None:
        self.nr_random_lines = st.number_input("Number of random lines:", value=5)

    def get_data_header(self) -> None:
        self.data_header = self.param_names + [
            self.final_col_name1,
            self.final_col_name2,
        ]

    def get_decimal_places(self):
        self.decimal_places = st.number_input(
            "Number of decimal places:", value=2, min_value=1, step=1
        )

    def get_optimization_type(self) -> None:
        self.optimization_type = st.selectbox(
            "Select optimization type:", ["Single", "Multi"]
        )
        if self.optimization_type == "Single":
            self.final_col_name = st.text_input("Name of the objective:", "output")
            self.data_header = self.param_names + [self.final_col_name]
        else:
            self.final_col_name1 = st.text_input(
                "Name of the first objective:", "output1"
            )
            self.final_col_name2 = st.text_input(
                "Name of the second objective:", "output2"
            )
            self.data_header = self.param_names + [
                self.final_col_name1,
                self.final_col_name2,
            ]

    def generate(self) -> None:
        self.get_randomization_type()
        self.get_input_values()
        self.param_names, self.param_types = self.param_info_func()
        self.get_data_header()
        self.get_decimal_places()
        self.param_ranges, self.param_intervals = self.param_ranges_func(
            self.param_names, self.param_types
        )
        self.param_values = self.value_generator_func(
            self.param_names,
            self.param_types,
            self.param_ranges,
            self.param_intervals,
            self.nr_random_lines,
            self.decimal_places,
        )
        self.get_optimization_type()  # must be after get_data_header()
        self.csv_writer_func(self.data_header, self.param_values)
        self.csv_downloader_func()
