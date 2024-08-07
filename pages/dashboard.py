from typing import Any, Dict
import streamlit as st
from utils import (
    show_dashboard,
    show_dashboard_multi,
    show_interaction_pdp,
    show_interaction_pdp_multi,
    get_features,
    get_table_names,
    get_latest_data_for_table,
    get_latest_data_and_metadata,
    train_model,
    train_model_multi,
    feature_importance,
    feature_importance_multi,
    plot_output_with_confidence,
    create_dashboard_report_multi,
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

    # Get the latest metadata
    df_with_preds, metadata, latest_table = get_latest_data_and_metadata(user_id)
    columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
    df = df_with_preds[columns_to_keep]
    # print(f"########### {latest_table}")
    # print("df to display:", df)

    default_table = latest_table
    selected_table = st.selectbox(
        "Select a table", table_names, index=table_names.index(default_table)
    )

    if selected_table != default_table:
        df_with_preds, metadata = get_latest_data_for_table(user_id, selected_table)
        columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
        df = df_with_preds[columns_to_keep]
        # print(f"Selected table {selected_table} is different from default table {default_table}")
        # print("df to display:", df)

    if metadata is None:
        raise ValueError("metadata is None. Please upload a new table.")

    if selected_table:
        df = df.dropna()

        # Get the first N columns based on the length of session X_columns
        features = get_features(df)
        if len(metadata["output_column_names"]) == 2:
            features = features[:-1]

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

            directions: Dict[str, Any] = metadata["directions"]
            output_columns: list[str] = metadata["output_column_names"]

            # TODO: save metadata to db, currently switching between single and multi will not work
            if len(output_columns) == 2:
                models = train_model_multi(df)
                show_dashboard_multi(
                    df,
                    models,
                    directions,
                    output_columns,
                )
                feature_importance_multi(df, models, output_columns)
                show_interaction_pdp_multi(
                    df,
                    pair_param,
                    models,
                    output_columns,
                    overlay=True,
                )
                plot_output_with_confidence(df_with_preds, output_columns, metadata)

            else:
                model = train_model(df)
                show_dashboard(df, model, directions, output_columns)
                feature_importance(df, model)
                show_interaction_pdp(df, pair_param, model, overlay=True)

            # Add the button for downloading the Matplotlib plot
            if st.button("Generate Summary Report for Download"):
                buf = create_dashboard_report_multi(
                    df, df_with_preds, models, output_columns, metadata
                )
                st.download_button(
                    label="Download Matplotlib Plot as PDF",
                    data=buf,
                    file_name="summary_report.pdf",
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
