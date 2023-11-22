import streamlit as st
from utils import (
    insert_data,
    upload_local_to_bucket,
    save_to_local,
)
from streamlit_extras.switch_page_button import switch_page
from time import sleep
import pandas as pd


st.title("Upload first batch CSV")


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()

    st.write("Please upload your first batch CSV file to create a new table.")
    file = st.file_uploader("Upload first batch CSV", type="csv")
    if file is not None:
        st.write("Please enter a name for the new table.")
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

                insert_data(table_name, file, st.session_state.user_id)

                bucket_name = "test-bucket"
                file.seek(0)  # Reset the file pointer to the beginning
                df = pd.read_csv(file)
                output_file_name = "raw-data.csv"
                save_to_local(bucket_name, table_name, output_file_name, df)
                upload_local_to_bucket(bucket_name, table_name, file_extension=".csv")
                st.write("Time for some preprocessing: switching to `Clean`.")

                sleep(2)
                # Switch to the clean page
                switch_page("clean")


if __name__ == "__main__":
    main()
