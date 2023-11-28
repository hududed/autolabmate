import streamlit as st
from utils import (
    show_dashboard,
    show_interaction_pdp,
    get_features,
    get_table_names,
    get_latest_row,
    train_model,
    feature_importance,
)

st.title("Dashboard")


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()

    user_id = st.session_state.user_id

    table_names = get_table_names(user_id)
    if not table_names:
        st.write("No tables found.")
        return

    default_table = (
        st.session_state.table_name
        if "table_name" in st.session_state
        and st.session_state.table_name in table_names
        else table_names[0]
    )
    selected_table = st.selectbox(
        "Select a table", table_names, index=table_names.index(default_table)
    )

    if selected_table:
        df = get_latest_row(user_id, selected_table)
        df = df.dropna()
        model = train_model(df)
        show_dashboard(df, model)
        feature_importance(df, model)

        # Create a list of features i.e., all columns except the target column
        features = get_features(df)

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
            # df = query_table(table_name)
            # print(df)
            show_interaction_pdp(df, pair_param, model, overlay=True)

    # else:
    #     st.write("No tables found in the database.")


if __name__ == "__main__":
    main()
