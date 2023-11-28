import streamlit as st
from utils import (
    validate_inputs,
    save_and_upload_results,
    load_metadata,
    py_dict_to_r_list,
    save_to_local,
    upload_local_to_bucket,
    save_metadata,
    replace_value_with_nan,
    get_table_names,
    get_latest_row,
    insert_data,
)
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

st.title("Update Experiment")


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()
    # Reset st.session_state.button_start_ml to False when the page is loaded
    if "button_start_ml" not in st.session_state or st.session_state.button_start_ml:
        st.session_state.button_start_ml = False

    st.warning("This section is still under development.")

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
        st.write("Loading data from previous batch: ")
        df = get_latest_row(user_id, selected_table)
        st.dataframe(df)

        # Query user for batch number with st.number_input and save as batch_number
        batch_number = st.number_input(
            "Enter batch number", min_value=2, value=2, step=1
        )
        try:
            # Try to load the metadata
            metadata = load_metadata(user_id, selected_table, batch_number - 1)
        except FileNotFoundError:
            # If a FileNotFoundError is raised, display a message and return
            st.write(
                f"No metadata found for table {selected_table}, and previous batch {batch_number - 1}."
            )
            return

        # st.write("Loading metadata from previous batch: ")
        with st.expander("Show metadata from previous batch", expanded=False):
            st.write(metadata)

        bucket_name = metadata["bucket_name"]
        parameter_ranges = metadata["parameter_ranges"]

        # User validates
        uploaded_file = st.file_uploader("Upload new data", type=["csv"])
        if st.button("Validate"):
            validation_errors = validate_inputs(df, parameter_ranges)

            # Display validation errors or metadata
            if validation_errors:
                for error in validation_errors:
                    st.write(error)

            else:
                st.write("Validation successful!")
                if uploaded_file is not None:
                    st.session_state.new_data = pd.read_csv(uploaded_file)
                    st.dataframe(st.session_state.new_data)

                    # TODO: data is then saved save_to_local with user input batch_number. path should be e.g. test-bucket/{table_name}/{batch_number}/*.csv

                    ro.r(
                        """
                    library(mlr3mbo)
                    library(mlr3)
                    library(mlr3learners)
                    library(bbotk)
                    library(data.table)
                    library(tibble)
                    library(R.utils)

                    load_file <- function(files, pattern) {
                        # Check if there are files that match the pattern
                        matched_files = grep(pattern, files, value = TRUE)
                        if (length(matched_files) == 0) {
                            stop(paste("No", pattern, "files found"))
                        }

                        # Filter for the latest file
                        latest_file = max(matched_files)

                        # Check if latest_file is a valid file
                        if (!file.exists(latest_file)) {
                            stop(paste("File does not exist:", latest_file))
                        }

                        # Load the file
                        loaded_file = readRDS(latest_file)

                        return(loaded_file)
                    }

                    # load_archive must be modified to load the latest metadata-*.json 
                    load_archive <- function(metadata) {
                        # Get a list of all *.rds files in the bucket
                        files = list.files(path = paste0(metadata$bucket_name, "/", metadata$user_id, "/", metadata$table_name, "/", metadata$batch_number), pattern = "*.rds", full.names = TRUE)

                        # Load the acqf-, acqopt-, and archive- files
                        acqf = load_file(files, "acqf-")
                        acqopt = load_file(files, "acqopt-")
                        archive = load_file(files, "archive-")

                        acqf$surrogate$archive = archive
                        return(list(archive, acqf, acqopt))
                    }

                    save_archive <- function(archive, acq_function, acq_optimizer, metadata) {
                        # Get the current timestamp
                        timestamp = format(Sys.time(), "%Y%m%d%H%M%S")

                        new_batch_number = as.integer(metadata$batch_number) + 1

                        # Define the directory path
                        dir_path = paste0(metadata$bucket_name, "/", metadata$user_id, "/", metadata$table_name, "/", new_batch_number)
                        
                        # Create the directory if it doesn't exist
                        if (!dir.exists(dir_path)) {
                            dir.create(dir_path, recursive = TRUE)
                        }
                        
                        # Save the objects to files
                        saveRDS(archive, paste0(dir_path, "/archive-", timestamp, ".rds"))
                        saveRDS(acq_function, paste0(dir_path, "/acqf-", timestamp, ".rds"))
                        saveRDS(acq_optimizer, paste0(dir_path, "/acqopt-", timestamp, ".rds"))
                    }

                    update_and_optimize <- function(acq_function, acq_optimizer, tmp_archive, candidate_new, lie) {
                        acq_function$surrogate$update()
                        acq_function$update()
                        tmp_archive$add_evals(xdt = candidate_new, xss_trafoed = transform_xdt_to_xss(candidate_new, tmp_archive$search_space), ydt = lie)
                        candidate_new = acq_optimizer$optimize()
                        return(candidate_new)
                    }

                    add_evals_to_archive <- function(archive, acq_function, acq_optimizer, data, q, metadata) {
                        lie = data.table()
                        liar = min
                        acq_function$surrogate$update()
                        acq_function$update()
                        candidate = acq_optimizer$optimize()
                        print(candidate)
                        tmp_archive = archive$clone(deep = TRUE)
                        acq_function$surrogate$archive = tmp_archive
                        lie[, archive$cols_y := liar(archive$data[[archive$cols_y]])]
                        candidate_new = candidate

                        # Check if lie is a data.table
                        if (!is.data.table(lie)) {
                            stop("lie is not a data.table")
                        }
                        candidate_new = candidate
                        for (i in seq_len(q)[-1L]) {
                            candidate_new = update_and_optimize(acq_function, acq_optimizer, tmp_archive, candidate_new, lie)
                            candidate = rbind(candidate, candidate_new)
                        }
                        candidate_new = update_and_optimize(acq_function, acq_optimizer, tmp_archive, candidate_new, lie)
                        # Iterate over each column in candidate
                        for (col in names(candidate_new)) {
                            # If the column is numeric, round and format it
                            if (is.double(candidate_new[[col]])) {
                                candidate_new[[col]] <- format(round(candidate_new[[col]], 2), nsmall = 2)
                            }
                        }
                        
                        save_archive(archive, acq_function, acq_optimizer, metadata)
                        return(list(candidate, archive, acq_function))
                        }
                    experiment <- function(data, metadata) {
                        set.seed(42)
                        result = load_archive(metadata)
                        full_data = as.data.table(data)
                        # print(data)
                        data <- tail(full_data, n=metadata$num_random_lines)
                        print(data)
                            
                        archive = result[[1]]
                        acq_function = result[[2]]
                        acq_optimizer = result[[3]]
                        
                        # Check if metadata$output_column_names is NULL or empty
                        if (is.null(metadata$output_column_names) || length(metadata$output_column_names) == 0) {
                            stop("metadata$output_column_names is NULL or empty")
                        }

                        # Check if all output_column_names exist in data
                        if (!all(metadata$output_column_names %in% names(data))) {
                            stop("Some names in metadata$output_column_names do not exist in data")
                        }

                        # print(class(archive))
                        # print(methods(class=class(archive)))


                        # Now you can safely call the add_evals method
                        archive$add_evals(xdt = data[, names(metadata$parameter_info), with=FALSE], ydt = data[, metadata$output_column_names, with=FALSE])
                        print("Model archive so far: ")
                        print(archive)
                        q = metadata$num_random_lines
                        result = add_evals_to_archive(archive, acq_function, acq_optimizer, data, q, metadata)

                        candidate = result[[1]]
                        archive = result[[2]]
                        acq_function = result[[3]]

                        print(result)

                        x2 <- candidate[, names(metadata$parameter_info), with=FALSE]
                        print("New candidates: ")
                        print(x2)
                        print("New archive: ")
                        print(archive)

                        x2_dt <- as.data.table(x2)
                        full_data <- rbindlist(list(full_data, x2_dt), fill = TRUE)
                        print(full_data)
                        print("Returning data to streamlit")
                        return(full_data)

                        }
                        """
                    )

        # Display the updated table

        if "button_start_ml" not in st.session_state:
            st.session_state.button_start_ml = False

        if st.button("Start ML"):
            st.session_state.button_start_ml = not st.session_state.button_start_ml

        if st.session_state.button_start_ml:
            pandas2ri.activate()

            converter = ro.default_converter + pandas2ri.converter
            with converter.context():
                r_data = ro.conversion.get_conversion().py2rpy(
                    st.session_state.new_data
                )
                r_metadata = py_dict_to_r_list(metadata)

                # Call the R function
                rsum = ro.r["experiment"]
                result = rsum(r_data, r_metadata)

                metadata["batch_number"] = batch_number

                save_and_upload_results(metadata, batch_number)

                # Convert R data frame to pandas data frame
                df = ro.conversion.get_conversion().rpy2py(result)

                # Replace -2147483648 with np.nan if -2147483648 exists in the DataFrame
                replace_value_with_nan(df)

                save_metadata(metadata, user_id, selected_table, batch_number)
                st.write("Saving metadata to local file")

                upload_local_to_bucket(
                    bucket_name, user_id, selected_table, batch_number
                )

                output_file_name = f"{batch_number}-data.csv"
                save_to_local(
                    bucket_name,
                    user_id,
                    selected_table,
                    output_file_name,
                    df,
                    batch_number,
                )

                try:
                    insert_data(selected_table, st.session_state.new_data, user_id)

                    st.write(
                        "Data uploaded successfully! Head to `dashboard` to see your data!"
                    )
                except Exception as e:
                    st.write(f"Error uploading data: {e}")

                st.dataframe(df)
                st.session_state.update_clicked = False
                st.session_state.button_start_ml = False

                st.write(
                    "Your next batch of experiments to run are ready! :fire: \n Remember to check your data in `dashboard` before running the next campaign. Happy experimenting!"
                )
                st.write(
                    f"Files downloaded to local directory: /{bucket_name}/{user_id}/{selected_table}/{batch_number}"
                )
                st.write(
                    "Run the proposed batch of experiments and proceed to `update` the model."
                )


if __name__ == "__main__":
    main()
