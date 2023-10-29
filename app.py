import streamlit as st
import csv
from random import randrange, uniform
import pandas as pd

def generate():
    series = st.number_input('Number of parameters:', value=3)
    nr_random_lines = st.number_input('Number of random lines:', value=10)

    # Ask the user to enter the parameter names and types
    param_names = []
    param_types = []
    for i in range(series):
        param_name = st.text_input(f'Parameter {i+1} name:', f'param{i+1}')
        param_type = st.selectbox(f'Parameter {i+1} type:', ['Integer', 'Float', 'Categorical'])
        param_names.append(param_name)
        param_types.append(param_type)

    # Initialize the data_header variable with default values
    final_col_name1 = 'output1'
    final_col_name2 = 'output2'
    data_header = param_names + [final_col_name1, final_col_name2]

    # Ask the user to select single or multiobjective optimization
    optimization_type = st.selectbox('Select optimization type:', ['Single', 'Multi'])

    # If single objective optimization is selected, ask the user to enter the name of the objective
    if optimization_type == 'Single':
        final_col_name = st.text_input('Name of the objective:', 'output')
        data_header = param_names + [final_col_name]

    # If multiobjective optimization is selected, ask the user to enter the names of the objectives
    if optimization_type == 'Multi':
        final_col_name1 = st.text_input('Name of the first objective:', 'output1')
        final_col_name2 = st.text_input('Name of the second objective:', 'output2')
        data_header = param_names + [final_col_name1, final_col_name2]

    # Ask the user to enter the minimum and maximum values or categories for each parameter
    param_ranges = []
    for i in range(series):
        if param_types[i] == 'Integer':
            min_val = st.number_input(f'Minimum value for {param_names[i]}:', value=0)
            max_val = st.number_input(f'Maximum value for {param_names[i]}:', value=100)
            param_ranges.append((min_val, max_val))
        elif param_types[i] == 'Float':
            min_val = st.number_input(f'Minimum value for {param_names[i]}:', value=0.0)
            max_val = st.number_input(f'Maximum value for {param_names[i]}:', value=100.0)
            param_ranges.append((min_val, max_val))
        else:
            categories = st.text_input(f'Enter categories for {param_names[i]} (comma-separated):', 'cat1,cat2,cat3')
            categories = [cat.strip() for cat in categories.split(',')]
            param_ranges.append(categories)

    # Generate the parameter values
    param_values = []
    try:
        for i in range(nr_random_lines):
            values = []
            for j in range(series):
                if param_types[j] == 'Integer':
                    min_val, max_val = param_ranges[j]
                    value = randrange(min_val, max_val+1, 1)
                elif param_types[j] == 'Float':
                    min_val, max_val = param_ranges[j]
                    value = format(round(uniform(min_val, max_val), 2), '.2f')
                else:
                    categories = param_ranges[j]
                    value = categories[randrange(len(categories))]
                values.append(value)
            param_values.append(values)
    except Exception as e:
        st.error(f'Error generating parameter values: {e}')

    # Write the parameter values to a CSV file
    with open('dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_header)
        for values in param_values:
            writer.writerow(values)

    df2 = pd.read_csv('dataset.csv')
    df2 = df2.drop_duplicates() #keep only the unique rows
    df2.to_csv('data.csv', index=False) #this is what will be read by mlrMBO in the R code

    st.download_button(
        label="Download CSV",
        data=open('data.csv', 'rb').read(),
        file_name='data.csv',
        mime='text/csv'
    )

def main():
    st.title('CSV Generator')
    generate()

if __name__ == '__main__':
    main()