import streamlit as st
from utils import (
    engine,
    inspect,
    get_user_inputs,
    validate_inputs,
    display_dictionary,
)
import pandas as pd


st.title("Propose Experiment")


def main():
    # choose table in supabase via streamlit dropdown
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        table_name = st.selectbox("Select a table to display", table_names)

        # Load the selected table
        query = f"SELECT * FROM {table_name};"
        table = pd.read_sql_query(query, engine)
        (
            optimization_type,
            output_column_names,
            num_parameters,
            num_random_lines,
            parameter_info,
            parameter_ranges,
        ) = get_user_inputs(table, table_name)

        # Add validate button
        if st.button("Validate"):
            validation_errors = validate_inputs(table, parameter_ranges)

            # Display validation errors or metadata
            if validation_errors:
                for error in validation_errors:
                    st.write(error)
            else:
                st.write("Validation passed.")
                display_dictionary(
                    table_name,
                    optimization_type,
                    output_column_names,
                    num_parameters,
                    num_random_lines,
                    parameter_info,
                    parameter_ranges,
                )


if __name__ == "__main__":
    main()
