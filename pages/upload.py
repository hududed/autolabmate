import pandas as pd
import streamlit as st

from db.crud.data import insert_data
from dependencies.authentication import check_authentication, initialize_session_state
from utils.dataframe import sanitize_column_names_for_table
from utils.file import save_to_local, upload_local_to_bucket
from utils.io import sanitize_table_name

st.title("Upload first batch CSV")


def main():
    initialize_session_state()
    check_authentication()

    user_id = st.session_state.user_id
    metadata = {
        "directions": {},
        "table_name": "",
        "X_columns": [],
        "output_column_names": [],
        "optimization_type": "",
    }

    st.write("Please upload your first batch CSV file.")
    file = st.file_uploader("Upload first batch CSV", type="csv")
    if file is not None:
        st.write("Please enter a name for the new table.")

        df = pd.read_csv(file)
        with st.form(key="create_table_form"):
            table_name = st.text_input("Table Name")
            confirm_upload = st.checkbox(
                "I confirm that I have both inputs and outputs filled."
            )
            if (
                st.form_submit_button("Create Table")
                and table_name != ""
                and confirm_upload
            ):
                # Sanitize the table name
                table_name = sanitize_table_name(table_name)
                st.session_state.table_name = table_name

                # st.dataframe(df)
        if st.session_state.table_name != "":
            df = sanitize_column_names_for_table(df)
            st.write(
                "Column headers sanitized. The following table will be updated in the database."
            )
            # Let the user select the X columns
            X_columns = st.multiselect(
                "Select the X columns",
                list(df.columns),
            )

            # Let the user select the y columns
            y_columns = st.multiselect(
                "Select the y columns",
                list(set(df.columns) - set(X_columns)),
            )
            # Let the user select the optimization direction for each y column
            y_directions = {}
            for column in y_columns:
                y_directions[column] = st.selectbox(
                    f"Should {column} be optimized for min or max?",
                    options=["minimize", "maximize"],
                    key=column,
                )

            # Add y_directions to metadata
            metadata["directions"] = y_directions

            # Add table_name, X_columns, and y_columns to metadata
            metadata["table_name"] = st.session_state.table_name
            metadata["X_columns"] = X_columns
            metadata["output_column_names"] = y_columns

            # Detect optimization type
            if len(y_columns) == 1:
                metadata["optimization_type"] = "single"
            else:
                metadata["optimization_type"] = "multi"

            # Rearrange the DataFrame
            df = df[X_columns + y_columns]

            st.dataframe(df)

        # Confirm drops button
        st.warning("Warning: The changes you are about to make are permanent.")
        if st.button("Confirm drops and insert"):
            # Display a warning message

            st.session_state.update_clicked = True
            insert_data(st.session_state.table_name, df, user_id, metadata)

            bucket_name = "test-bucket"
            file.seek(0)  # Reset the file pointer to the beginning
            df = pd.read_csv(file)
            output_file_name = "raw-data.csv"
            save_to_local(
                bucket_name, user_id, st.session_state.table_name, output_file_name, df
            )
            upload_local_to_bucket(
                bucket_name, user_id, st.session_state.table_name, file_extension=".csv"
            )
            st.write("Head to `dashboard` to see your data! :fire:")


if __name__ == "__main__":
    main()
