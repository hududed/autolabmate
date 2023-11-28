import streamlit as st
from utils import (
    insert_data,
    upload_local_to_bucket,
    save_to_local,
    sanitize_column_names,
)
import pandas as pd


st.title("Upload first batch CSV")


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()

    if "table_name" not in st.session_state:
        st.session_state.table_name = ""
    user_id = st.session_state.user_id

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
                st.session_state.table_name = table_name

                # st.dataframe(df)
        if st.session_state.table_name != "":
            df = sanitize_column_names(df)
            st.write(
                "Column headers sanitized. The following table will be updated in the database."
            )
            drop_columns = st.multiselect(
                "Select a column to drop before inserting into database",
                list(df.columns),
            )
            for drop_column in drop_columns:
                # Drop the column from the dataframe
                df = df.drop(columns=[drop_column])
                st.write(f"Dropped column: {drop_column}")

            st.dataframe(df)

        # Confirm drops button
        st.warning("Warning: The changes you are about to make are permanent.")
        if st.button("Confirm drops and insert"):
            # Display a warning message

            st.session_state.update_clicked = True

            insert_data(table_name, df, user_id)

            bucket_name = "test-bucket"
            file.seek(0)  # Reset the file pointer to the beginning
            df = pd.read_csv(file)
            output_file_name = "raw-data.csv"
            save_to_local(bucket_name, user_id, table_name, output_file_name, df)
            upload_local_to_bucket(
                bucket_name, user_id, table_name, file_extension=".csv"
            )
            st.write("Head to `dashboard` to see your data! :fire:")


if __name__ == "__main__":
    main()
