# filepath: pages/propose.py
import os

import pandas as pd
import rpy2.robjects as ro
import streamlit as st
from rpy2.robjects import pandas2ri

from db.crud.data import (
    get_latest_data_metadata_by_user_id_table,
    get_latest_data_metadata_table_by_user_id,
    insert_data,
)
from db.crud.table import get_table_names_by_user_id
from dependencies.authentication import (
    check_authentication,
    clear_session_state,
    initialize_session_state_basic,
)
from dependencies.navigation import authenticate_and_show_nav
from utils.dataframe import replace_value_with_nan
from utils.file import (
    compress_files,
    retrieve_and_download_files,
    save_metadata,
    save_to_local,
    upload_local_to_bucket,
    upload_metadata_to_bucket,
)
from utils.io import (
    display_dictionary,
    generate_timestamps,
    get_user_inputs,
    validate_inputs,
)
from utils.rpy2_utils import py_dict_to_r_list

# Set page config first
st.set_page_config(page_title="Upload | Autolabmate", page_icon="⬆️")

authenticate_and_show_nav()
initialize_session_state_basic()

if "update_page_loaded" in st.session_state:
    del st.session_state.update_page_loaded

st.title("Propose Experiment")

if not st.session_state.propose_page_loaded:
    st.session_state.propose_page_loaded = True
    clear_session_state(
        [
            "df_no_preds",
            "messages",
            "zip_buffer",
            "output_zip",
            "update_clicked",
            "expander_what_in_file",
            "expander_usage_examples",
        ]
    )


def main():
    pandas2ri.activate()
    check_authentication()
    if "button_start_ml" not in st.session_state or st.session_state.button_start_ml:
        st.session_state.button_start_ml = False

    user_id = st.session_state.user_id

    table_names = get_table_names_by_user_id(user_id)
    if not table_names:
        st.write("No tables found.")
        return

    df_with_preds, metadata, latest_table = get_latest_data_metadata_table_by_user_id(
        user_id
    )
    columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
    df = df_with_preds[columns_to_keep]

    default_table = latest_table
    selected_table = st.selectbox(
        "Select a table", table_names, index=table_names.index(default_table)
    )
    seed = st.number_input("Enter a seed", value=42, step=1)

    if selected_table != default_table:
        df_with_preds, metadata = get_latest_data_metadata_by_user_id_table(
            user_id, selected_table
        )
        columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
        df = df_with_preds[columns_to_keep]

    # NEW: Let the user choose the learner type without affecting optimization_type
    learner_choice = st.selectbox(
        "Choose Learner", ["Random Forest", "Gaussian Process"], index=0
    )
    metadata["learner_choice"] = learner_choice

    if selected_table:
        (
            batch_number,
            optimization_type,  # "single" or "multi"
            output_column_names,
            num_parameters,
            num_random_lines,
            parameter_info,
            parameter_ranges,
            directions,
            to_nearest,
        ) = get_user_inputs(df, metadata)

        if st.button("Validate"):
            # Pass learner_choice instead of optimization_type for learner-specific validation
            validation_errors = validate_inputs(
                df,
                parameter_ranges,
                output_column_names,
                learner_choice=metadata["learner_choice"],
            )

            if validation_errors:
                for error in validation_errors:
                    st.write(error)
            else:
                st.write("Validation passed.")
                metadata = display_dictionary(
                    seed,
                    batch_number,
                    selected_table,
                    optimization_type,
                    output_column_names,
                    num_parameters,
                    num_random_lines,
                    parameter_info,
                    parameter_ranges,
                    directions,
                    user_id,
                    to_nearest,
                    metadata,
                )
                insert_data(metadata["table_name"], df, user_id, metadata)

        if "button_start_ml" not in st.session_state:
            st.session_state.button_start_ml = False

        if st.button("Start ML"):
            # Re-run validation before starting ML
            errors = validate_inputs(
                df,
                parameter_ranges,
                output_column_names,
                learner_choice=metadata["learner_choice"],
            )
            if errors:
                st.error(
                    "Cannot start ML. Please fix the following errors first:\n"
                    + "\n".join(errors)
                )
            else:
                st.session_state.button_start_ml = not st.session_state.button_start_ml

        if st.session_state.button_start_ml:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            metadata["base_dir"] = base_dir
            r_dir = os.path.join(base_dir, "R")

            with ro.conversion.localconverter(
                ro.default_converter + pandas2ri.converter
            ):
                ro.r.source(os.path.join(r_dir, "propose.R"))

            # Gaussian Process requires columns to be float
            for col in metadata["output_column_names"]:
                df[col] = df[col].astype(float)

            converter = ro.default_converter + pandas2ri.converter
            with converter.context():
                r_data = ro.conversion.get_conversion().py2rpy(df)
                r_metadata = py_dict_to_r_list(metadata)
                rsum = ro.r["experiment"]
                result = rsum(r_data, r_metadata)
                result_no_preds = result["data_no_preds"]
                result_with_preds = result["data_with_preds"]

                upload_metadata_to_bucket(metadata, batch_number)
                df_no_preds = ro.conversion.get_conversion().rpy2py(result_no_preds)
                df_with_preds = ro.conversion.get_conversion().rpy2py(result_with_preds)

            for col in metadata["parameter_info"]:
                if metadata["parameter_info"][col] == "integer":
                    df_no_preds[col] = df_no_preds[col].astype(pd.Int64Dtype())
                    df_with_preds[col] = df_with_preds[col].astype(pd.Int64Dtype())

            replace_value_with_nan(df_no_preds)
            replace_value_with_nan(df_with_preds)

            save_metadata(metadata, user_id, selected_table, batch_number)

            bucket_name = metadata["bucket_name"]
            batch_number = metadata["batch_number"]

            upload_local_to_bucket(
                bucket_name,
                user_id,
                selected_table,
                batch_number,
                file_extension=".rds",
            )
            filename_timestamp, display_timestamp = generate_timestamps()

            filename_no_preds = f"{filename_timestamp}_{batch_number}-data.csv"
            filename_with_preds = (
                f"{filename_timestamp}_{batch_number}-data-with-preds.csv"
            )

            save_to_local(
                bucket_name,
                user_id,
                selected_table,
                filename_no_preds,
                df_no_preds,
                batch_number,
            )
            save_to_local(
                bucket_name,
                user_id,
                selected_table,
                filename_with_preds,
                df_with_preds,
                batch_number,
            )
            upload_local_to_bucket(
                bucket_name,
                user_id,
                selected_table,
                batch_number,
                file_extension=".csv",
            )

            downloaded_files = retrieve_and_download_files(
                bucket_name, user_id, selected_table, batch_number
            )
            output_zip = f"{filename_timestamp}_{batch_number}_data.zip"
            zip_buffer = compress_files(downloaded_files)

            st.session_state.zip_buffer = zip_buffer
            st.session_state.output_zip = output_zip
            st.session_state.df_no_preds = df_no_preds

            st.session_state.update_clicked = False
            st.session_state.button_start_ml = False

            st.session_state.messages.append(
                "Your next batch of experiments to run are ready! :fire: \n Remember to check your data in `dashboard` before running the next campaign. Happy experimenting!"
            )
            st.session_state.messages.append(
                "Run the proposed batch of experiments and proceed to update the model."
            )

            st.session_state.expander_what_in_file = True
            st.session_state.expander_usage_examples = True

    for message in st.session_state.messages:
        st.write(message)
    if st.session_state.df_no_preds is not None:
        st.write(st.session_state.df_no_preds)

    # Add the download button outside the if block to ensure it is always available
    if (
        st.session_state.zip_buffer is not None
        and st.session_state.output_zip is not None
    ):
        st.download_button(
            label="Download compressed data",
            data=st.session_state.zip_buffer.getvalue(),
            file_name=st.session_state.output_zip,
            mime="application/zip",
        )
    if st.session_state.expander_what_in_file:
        with st.expander("❓ What's in the file ?", expanded=False):
            st.write("""
            **1. `archive-{}.rds` (Optimization Archive)**
            - **Contains:** Evaluated hyperparameter configurations (`xdt`), objective function values (`ydt`), search space definitions, and codomain constraints.
            - **Purpose:** Stores the history of evaluations for analysis, debugging, and incremental updates. It acts as the main record for tracking Bayesian optimization progress.

            **2. `acqopt-{}.rds` (Acquisition Optimizer)**
            - **Contains:** The optimizer responsible for searching new candidate points, including the optimization method (`random_search`, `DIRECT`, etc.) and termination criteria.
            - **Purpose:** Ensures efficient candidate search in the hyperparameter space based on acquisition function guidance.

            **3. `acqf-{}.rds` (Acquisition Function)**
            - **Contains:** The acquisition function type (`ei`, `ehvi`, etc.) and surrogate model predictions.
            - **Purpose:** Determines the most promising points to evaluate, balancing exploration (searching new areas) and exploitation (refining best-known regions).
            """)
    if st.session_state.expander_usage_examples:
        with st.expander("💻 Usage examples", expanded=False):
            st.write("""
            ```r
            # Load and inspect the archive
            archive <- readRDS("path/to/archive-20250131-1200.rds")
            print(archive$data)  # View stored hyperparameter configurations
            best_config <- archive$best()
            print(best_config)  # Extract best-performing configuration

            # Use the acquisition function
            acq_function <- readRDS("path/to/acqf-20250131-1200.rds")
            acq_function$surrogate$update()
            acq_function$update()
            candidate_score <- acq_function$eval_dt(new_candidate)
            print(candidate_score)  # Evaluate candidate

            # Optimize using the acquisition optimizer
            acq_optimizer <- readRDS("path/to/acqopt-20250131-1200.rds")
            candidate <- acq_optimizer$optimize()
            print(candidate)  # Get next candidate
            ```
            """)


if __name__ == "__main__":
    main()
