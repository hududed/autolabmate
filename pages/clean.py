from sqlalchemy import MetaData, Table, inspect
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils import engine, save_to_local, upload_local_to_bucket
import re
from time import sleep
from st_pages import hide_pages


def drop_column_from_table(table_name, column_name):
    # Reflect the table
    metadata = MetaData()
    db_table = Table(table_name, metadata, autoload_with=engine)

    # Check if the column exists
    if column_name in db_table.c:
        # Drop the column
        with engine.connect() as connection:
            db_table._columns.remove(db_table.c[column_name])
        st.write(f"Dropped column {column_name} from table {table_name}")
    else:
        st.write(f'Column "{column_name}" does not exist in table "{table_name}"')


def sanitize_column_names(table):
    # Sanitize column names
    table.columns = [re.sub("[^0-9a-zA-Z_]", "", col) for col in table.columns]
    table.columns = ["col_" + col if col[0].isdigit() else col for col in table.columns]
    return table


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()
    # choose table in supabase via streamlit dropdown
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        default_table = (
            st.session_state.table_name.lower()
            if "table_name" in st.session_state
            else table_names[0].lower()
        )
        table_name = st.selectbox(
            "Select a table",
            [name.lower() for name in table_names],
            index=[name.lower() for name in table_names].index(default_table),
        )

        # Load the table
        if "table" not in st.session_state or st.session_state.table_name != table_name:
            st.session_state.table = pd.read_sql_table(table_name, engine)
            st.session_state.table_name = table_name

        # Dropdown to select column to drop
        drop_column = st.selectbox(
            "Select a column to drop from database",
            ["None"] + list(st.session_state.table.columns),
        )
        if drop_column != "None":
            # Drop the column from the dataframe
            st.session_state.table = st.session_state.table.drop(columns=[drop_column])
            st.write(f"Dropped column: {drop_column}")

            # Drop the column from the database table
            drop_column_from_table(table_name, drop_column)

        # Confirm drops button
        if st.button("Confirm drops"):
            # Sanitize column names
            st.session_state.table = sanitize_column_names(st.session_state.table)

            # Display a warning message
            st.warning("Warning: The changes you are about to make are permanent.")
            st.write(
                "Column headers sanitized. The following table will be updated in the database:"
            )
            st.session_state.update_clicked = True

        # Display the updated table
        st.dataframe(st.session_state.table)

        if (
            "update_clicked" in st.session_state
            and st.session_state.update_clicked
            and st.button("Update Table")
        ):
            # Replace the table in the database with the new table
            st.session_state.table.to_sql(
                table_name, engine, if_exists="replace", index=False
            )
            bucket_name = "test-bucket"
            output_file_name = "cleaned-data.csv"
            save_to_local(
                bucket_name, table_name, output_file_name, st.session_state.table
            )
            upload_local_to_bucket(bucket_name, table_name, file_extension=".csv")

            st.write(f"Table {table_name} has been updated.")
            st.session_state.update_clicked = False
            sleep(2)
            switch_page("dashboard")


if __name__ == "__main__":
    main()
