import streamlit as st
from utils import display_table, engine, inspect
from st_pages import hide_pages


st.title("Dashboard")


def main():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        default_table = (
            st.session_state.table_name
            if "table_name" in st.session_state
            else table_names[0]
        )
        table_name = st.selectbox(
            "Select a table", table_names, index=table_names.index(default_table)
        )
        # table_name = st.selectbox('Select a table to display', table_names)
        display_table(table_name)
    else:
        st.write("No tables found in the database.")


if __name__ == "__main__":
    main()
