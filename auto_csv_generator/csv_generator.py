import csv
from random import randrange, uniform
import pandas as pd
import streamlit as st
from pyDOE import lhs
from scipy.stats import norm
import sobol_seq


class CSVGenerator:
    def __init__(self):
        self.series = None
        self.nr_random_lines = None
        self.param_names = []
        self.param_types = []
        self.final_col_name1 = "output1"
        self.final_col_name2 = "output2"
        self.optimization_type = None
        self.final_col_name = None
        self.param_ranges = []
        self.param_values = []
        self.data_header = []

    def get_randomization_type(self) -> None:
        self.randomization_type = st.selectbox(
            "Select randomization type:", ["Random", "Latin Hypercube", "Sobol"]
        )

    def get_input_values(self) -> None:
        self.series = st.number_input("Number of parameters:", value=3)
        self.nr_random_lines = st.number_input("Number of random lines:", value=10)

    def get_parameter_info(self) -> None:
        for i in range(self.series):
            param_name = st.text_input(f"Parameter {i+1} name:", f"param{i+1}")
            param_type = st.selectbox(
                f"Parameter {i+1} type:", ["Integer", "Float", "Categorical"]
            )
            self.param_names.append(param_name)
            self.param_types.append(param_type)

    def get_data_header(self) -> None:
        self.data_header = self.param_names + [
            self.final_col_name1,
            self.final_col_name2,
        ]

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

    def get_parameter_ranges(self) -> None:
        for i in range(len(self.param_names)):
            if self.param_types[i] == "Integer":
                min_val = st.number_input(
                    f"Minimum value for {self.param_names[i]}:", value=0
                )
                max_val = st.number_input(
                    f"Maximum value for {self.param_names[i]}:", value=100
                )
                self.param_ranges.append((min_val, max_val))
            elif self.param_types[i] == "Float":
                min_val = st.number_input(
                    f"Minimum value for {self.param_names[i]}:", value=0.0
                )
                max_val = st.number_input(
                    f"Maximum value for {self.param_names[i]}:", value=100.0
                )
                self.param_ranges.append((min_val, max_val))
            else:
                categories = st.text_input(
                    f"Enter categories for {self.param_names[i]} (comma-separated):",
                    "cat1,cat2,cat3",
                )
                categories = [cat.strip() for cat in categories.split(",")]
                self.param_ranges.append(categories)

    # def generate_parameter_values(self) -> None:
    #     try:
    #         for i in range(self.nr_random_lines):
    #             values = []
    #             for j in range(self.series):
    #                 if self.param_types[j] == 'Integer':
    #                     min_val, max_val = self.param_ranges[j]
    #                     value = randrange(min_val, max_val+1, 1)
    #                 elif self.param_types[j] == 'Float':
    #                     min_val, max_val = self.param_ranges[j]
    #                     value = format(round(uniform(min_val, max_val), 2), '.2f')
    #                 else:
    #                     categories = self.param_ranges[j]
    #                     value = categories[randrange(len(categories))]
    #                 values.append(value)
    #             self.param_values.append(values)
    #     except Exception as e:
    #         st.error(f'Error generating parameter values: {e}')

    def generate_parameter_values(self) -> None:
        if self.randomization_type == "Random":
            self.generate_random_values()
        elif self.randomization_type == "Latin Hypercube":
            self.generate_lhs_values()
        elif self.randomization_type == "Sobol":
            self.generate_sobol_values()

    def generate_random_values(self) -> None:
        for i in range(self.nr_random_lines):
            values = []
            for j in range(self.series):
                if self.param_types[j] == "Integer":
                    min_val, max_val = self.param_ranges[j]
                    value = randrange(min_val, max_val + 1, 1)
                elif self.param_types[j] == "Float":
                    min_val, max_val = self.param_ranges[j]
                    value = format(round(uniform(min_val, max_val), 2), ".2f")
                else:
                    categories = self.param_ranges[j]
                    value = categories[randrange(len(categories))]
                values.append(value)
            self.param_values.append(values)

    def generate_lhs_values(self) -> None:
        # Generate Latin Hypercube samples
        samples = lhs(self.series, self.nr_random_lines)

        # Convert the samples to parameter values
        for i in range(self.nr_random_lines):
            values = []
            for j in range(self.series):
                if self.param_types[j] == "Integer":
                    min_val, max_val = self.param_ranges[j]
                    value = int(samples[i, j] * (max_val - min_val) + min_val)
                elif self.param_types[j] == "Float":
                    min_val, max_val = self.param_ranges[j]
                    value = format(samples[i, j] * (max_val - min_val) + min_val, ".2f")
                else:
                    categories = self.param_ranges[j]
                    value = categories[int(samples[i, j] * (len(categories) - 1))]
                values.append(value)
            self.param_values.append(values)

    def generate_sobol_values(self) -> None:
        # Generate Sobol sequence samples
        samples = sobol_seq.i4_sobol_generate(self.series, self.nr_random_lines)

        # Convert the samples to parameter values
        for i in range(self.nr_random_lines):
            values = []
            for j in range(self.series):
                if self.param_types[j] == "Integer":
                    min_val, max_val = self.param_ranges[j]
                    value = int(samples[i, j] * (max_val - min_val) + min_val)
                elif self.param_types[j] == "Float":
                    min_val, max_val = self.param_ranges[j]
                    value = format(samples[i, j] * (max_val - min_val) + min_val, ".2f")
                else:
                    categories = self.param_ranges[j]
                    value = categories[int(samples[i, j] * (len(categories) - 1))]
                values.append(value)
            self.param_values.append(values)

    def write_csv_file(self) -> None:
        with open("data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.data_header)
            for values in self.param_values:
                writer.writerow(values)
        df2 = pd.read_csv("data.csv")
        df2.to_csv(
            "data.csv", index=False
        )  # this is what will be read by mlrMBO in the R code

    def download_csv_file(self) -> None:
        st.download_button(
            label="Download CSV",
            data=open("data.csv", "rb").read(),
            file_name="data.csv",
            mime="text/csv",
        )

    def generate(self) -> None:
        self.get_randomization_type()
        self.get_input_values()
        self.get_parameter_info()
        self.get_data_header()
        self.get_optimization_type()  # must be after get_data_header()
        self.get_parameter_ranges()
        self.generate_parameter_values()
        self.write_csv_file()
        self.download_csv_file()
