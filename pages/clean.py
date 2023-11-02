from sqlalchemy import MetaData, Table, inspect
import pandas as pd
import streamlit as st
from utils import engine


def drop_column_from_table(table_name, column_name):
    # Reflect the table
    metadata = MetaData()
    db_table = Table(table_name, metadata, autoload_with=engine)

    # Check if the column exists
    if column_name in db_table.c:
        # Drop the column
        with engine.connect() as connection:
            db_table._columns.remove(db_table.c[column_name])
        st.write(f"Dropped column {column_name} from database table {table_name}")
    else:
        st.write(f'Column "{column_name}" does not exist in table "{table_name}"')

def main():
    # choose table in supabase via streamlit dropdown
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        table_name = st.selectbox("Select a table", table_names)

        # Load the table
        if 'table' not in st.session_state or st.session_state.table_name != table_name:
            st.session_state.table = pd.read_sql_table(table_name, engine)
            st.session_state.table_name = table_name

        # Display the table
        # st.dataframe(st.session_state.table)

        # Dropdown to select column to drop
        drop_column = st.selectbox("Select a column to drop from database", ["None"] + list(st.session_state.table.columns))
        if drop_column != "None":
            # Drop the column from the dataframe
            st.session_state.table = st.session_state.table.drop(columns=[drop_column])
            st.write(f"Dropped column: {drop_column}")

            # Drop the column from the database tab le
            drop_column_from_table(table_name, drop_column)

        # Display the updated table
        st.dataframe(st.session_state.table)

        # Validate button
        if st.button('Validate'):
            # Replace the table in the database with the new table
            st.session_state.table.to_sql(table_name, engine, if_exists='replace', index=False)
            st.write(f"Table {table_name} has been updated.")

if __name__ == '__main__':
    main()