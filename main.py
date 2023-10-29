import streamlit as st
import csv
from random import randrange
import pandas as pd

def generate():
    series = st.number_input('Number of parameters:', value=3)
    nr_random_lines = st.number_input('Number of random lines:', value=10)

    # Ask the user to enter the parameter names and ranges
    param_names = []
    param_ranges = []
    for i in range(series):
        param_name = st.text_input(f'Parameter {i+1} name:', f'param{i+1}')
        min_val = st.number_input(f'Minimum value for {param_name}:', value=0)
        max_val = st.number_input(f'Maximum value for {param_name}:', value=100)
        param_names.append(param_name)
        param_ranges.append((min_val, max_val))

    # Ask the user to enter the name of the objective
    final_col_name = st.text_input('Name of the objective:', 'output')

    # Generate the parameter values
    param_values = []
    for i in range(nr_random_lines):
        values = []
        for j in range(series):
            min_val, max_val = param_ranges[j]
            value = randrange(min_val, max_val+1, 1)
            values.append(value)
        param_values.append(values)

    # Write the parameter values to a CSV file
    data_header = param_names + [final_col_name]
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