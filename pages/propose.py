import streamlit as st
from utils import display_table, engine, inspect
from st_pages import hide_pages
import pandas as pd
import numpy as np


st.title("Propose Experiment")


def main():
    # choose table in supabase via streamlit dropdown
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        table_name = st.selectbox('Select a table to display', table_names)

        # Load the selected table
        query = f'SELECT * FROM {table_name};'
        table = pd.read_sql_query(query, engine)

        # Get optimization type
        optimization_type = st.selectbox('Select optimization type', ['single', 'multi'])

        # Get output column names
        if optimization_type == 'single':
            output_column_names = [table.columns[-1]]
        elif optimization_type == 'multi':
            output_column_names = table.columns[-2:]

        # Get number of parameters
        num_parameters = len(table.columns) - len(output_column_names)

        # Get number of random lines
        num_random_lines = st.number_input('Enter the number of random lines', min_value=1, max_value=len(table))

        # Get parameter info
        parameter_info = table.dtypes[:-len(output_column_names)].to_dict()

        # Get parameter ranges
        parameter_ranges = {}
        for column in table.columns[:-len(output_column_names)]:
            if np.issubdtype(table[column].dtype, np.number):
                min_value = st.number_input(f'Enter the min value for {column}', value=table[column].min())
                max_value = st.number_input(f'Enter the max value for {column}', value=table[column].max())
                parameter_ranges[column] = (min_value, max_value)
            elif np.issubdtype(table[column].dtype, object):
                categories = st.text_input(f'Enter the categories for {column}', value=', '.join(table[column].unique()))
                parameter_ranges[column] = categories.split(', ')

        # Generate the dictionary metadata
        metadata = {
            'table_name': table_name,
            'optimization_type': optimization_type,
            'output_column_names': output_column_names,
            'num_parameters': num_parameters,
            'num_random_lines': num_random_lines,
            'parameter_info': parameter_info,
            'parameter_ranges': parameter_ranges,
        }

        st.write(metadata)

if __name__ == "__main__":
    main()


# AIM: create dictionary metadata for table created by inferring from data or user input



# get_optimization_type -> user input: single (last column) or multi objective (2 last columns)

# get_output_column_names -> if single objective, return last column name, multi objective return last 2 column names 

# get_input_values -> infer: number of parameters (total column - output column(s)), user input: number of random lines

# get_parameter_info -> infer: get parameter (column) name (total column - output column(s)), infer: parameter type


# assume batch-optimization for now

# get_parameter_ranges -> infer: data types from column values, if integer or float ->user input: min and max values for integer and float, if strings -> user input: categories for strings

# validate if values are within range for integer and float, validate if unique string values are within categories

# assume learner type is regression (regr.ranger for now )

# assume expected improvement 

# generate_dict_metadata -> dictionary with metadata accompanying data table / sample


