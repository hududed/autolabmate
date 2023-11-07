import streamlit as st
from utils import (
    show_dashboard,
    show_interaction_pdp,
    engine,
    inspect,
    get_features,
    query_table,
)


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

        show_dashboard(table_name)

        # Create a list of features i.e., all columns except the target column
        features = get_features(table_name)

        # User input multibox select exactly 2 features to compare from the list of features
        selected_features = st.multiselect(
            "Select exactly 2 features to compare",
            options=features,
            default=features[:2],
        )

        # Check if exactly 2 features are selected
        if len(selected_features) != 2:
            st.error("Please select exactly 2 features.")
        else:
            # Once selected, the tuple of two features is added as pair_param
            pair_param = [tuple(selected_features)]
            df = query_table(table_name)
            # print(df)
            show_interaction_pdp(df, pair_param)

    else:
        st.write("No tables found in the database.")


if __name__ == "__main__":
    main()
