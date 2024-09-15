import streamlit as st
from utils import (
    compress_files,
    get_table_names,
    get_latest_data_for_table,
    get_latest_data_and_metadata,
    insert_data,
    py_dict_to_r_list,
    replace_value_with_nan,
    retrieve_and_download_files,
    save_to_local,
    save_metadata,
    upload_local_to_bucket,
    upload_metadata_to_bucket,
    validate_inputs,
)
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from datetime import datetime

from components.authenticate import initialize_session_state, check_authentication

initialize_session_state()

st.title("Update Experiment")


def main():
    check_authentication()

    # Reset st.session_state.button_start_ml to False when the page is loaded
    if "button_start_ml" not in st.session_state or st.session_state.button_start_ml:
        st.session_state.button_start_ml = False

    st.warning("This section is still under development.")

    user_id = st.session_state.user_id

    table_names = get_table_names(user_id)
    if not table_names:
        st.write("No tables found.")
        return

    batch_number = st.number_input("Enter batch number", min_value=2, value=2, step=1)

    # Get the latest metadata
    df_with_preds, metadata, latest_table = get_latest_data_and_metadata(
        user_id, batch_number
    )
    columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
    df = df_with_preds[columns_to_keep]

    default_table = latest_table
    selected_table = st.selectbox(
        "Select a table", table_names, index=table_names.index(default_table)
    )

    if selected_table != default_table:
        df_with_preds, metadata = get_latest_data_for_table(
            user_id, selected_table, batch_number
        )
        columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
        df = df_with_preds[columns_to_keep]

    if selected_table:
        st.write("Loading data from previous batch: ")
        # df = get_latest_row(user_id, selected_table)
        st.dataframe(df)

        with st.expander("Show metadata from previous batch", expanded=False):
            keys_to_remove = ["user_id", "bucket_name"]
            metadata_for_display = {
                k: v for k, v in metadata.items() if k not in keys_to_remove
            }
            st.write(metadata_for_display)
        print(metadata)

        bucket_name = metadata["bucket_name"]
        parameter_ranges = metadata["parameter_ranges"]
        output_column_names = metadata["output_column_names"]

        # User validates
        uploaded_file = st.file_uploader("Upload new data", type=["csv"])
        if st.button("Validate"):
            validation_errors = validate_inputs(
                df, parameter_ranges, output_column_names
            )

            # Display validation errors or metadata
            if validation_errors:
                for error in validation_errors:
                    st.write(error)

            else:
                st.write("Validation successful!")
                if uploaded_file is not None:
                    st.session_state.new_data = pd.read_csv(
                        uploaded_file, usecols=df.columns
                    )
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

                    round_to_nearest <- function(x, metadata) {
                        to_nearest = metadata$to_nearest
                        x_columns = metadata$X_columns
                        print("Metadata$to_nearest:")
                        print(to_nearest)
                        print("Metadata$X_columns:")
                        print(x_columns)

                        # Apply rounding only to metadata$X_columns 
                        if (is.data.table(x) || is.data.frame(x)) {
                            for (col_name in names(x)) {
                                if (col_name %in% x_columns) {
                                    col = x[[col_name]]
                                    if (is.numeric(col)) {
                                        nearest = as.numeric(to_nearest[[col_name]])
                                        print(paste("Column:", col_name))
                                        print(paste("Nearest value:", nearest))
                                        print("Values before rounding:")
                                        print(col)
                                        x[[col_name]] = round(col / nearest) * nearest
                                        print("Values after rounding:")
                                        print(x[[col_name]])
                                    }
                                } else {
                                    print(paste("Skipping column:", col_name))
                                }
                            }
                        } else if (is.numeric(x)) {
                            nearest = as.numeric(to_nearest)
                            print("Nearest value:")
                            print(nearest)
                            print("Values before rounding:")
                            print(x)
                            x <- round(x / nearest) * nearest
                            print("Values after rounding:")
                            print(x)
                        } else {
                            stop("x must be a data.table, data.frame, or numeric vector")
                        }
                        return(x)
                    }

                    # Function to find the latest file
                    find_latest_file <- function(files, pattern) {
                        matched_files <- grep(pattern, files, value = TRUE)
                        if (length(matched_files) == 0) {
                            stop(paste("No", pattern, "files found"))
                        }
                        timestamps <- as.numeric(gsub(".*-(\\\\d+).*", "\\\\1", matched_files))
                        latest_index <- which.max(timestamps)
                        return(matched_files[latest_index])
                    }
                    
                    # Function to load a file
                    load_file <- function(file_path, file_type) {
                        if (!file.exists(file_path)) {
                            stop(paste("File does not exist:", file_path))
                        }
                        if (file_type == "csv") {
                            return(read.csv(file_path))
                        } else if (file_type == "rds") {
                            return(readRDS(file_path))
                        } else {
                            stop(paste("Unsupported file type:", file_type))
                        }
                    }

                    # Function to load files based on pattern and type
                    load_files <- function(metadata, pattern, file_type) {
                        files = list.files(path = paste0(metadata$bucket_name, "/", metadata$user_id, "/", metadata$table_name, "/", metadata$batch_number), pattern = paste0("*.", file_type), full.names = TRUE)
                        latest_file = find_latest_file(files, pattern)
                        return(load_file(latest_file, file_type))
                    }

                    # Function to load data with predictions
                    load_predicted_data <- function(metadata) {
                        metadata$batch_number <- as.integer(metadata$batch_number)
                        return(load_files(metadata, "with-preds", "csv"))
                    }

                    # Function to load archive data
                    load_archive_data <- function(metadata) {
                        acqf = load_files(metadata, "acqf-", "rds")
                        acqopt = load_files(metadata, "acqopt-", "rds")
                        archive = load_files(metadata, "archive-", "rds")
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

                    update_and_optimize <- function(acq_function, acq_optimizer, tmp_archive, candidate_new, lie, metadata) {
                        acq_function$surrogate$update()
                        acq_function$update()
                        tmp_archive$add_evals(xdt = candidate_new,
                                              xss_trafoed = transform_xdt_to_xss(candidate_new, tmp_archive$search_space),
                                              ydt = lie)
                        candidate_new = acq_optimizer$optimize()
                        candidate_new = round_to_nearest(candidate_new, metadata)
                        return(candidate_new)
                    }

                    add_evals_to_archive <- function(archive, acq_function, acq_optimizer, data, q, metadata) {
                        # Check inputs
                        if (!is.data.table(archive$data)) {
                            stop("archive$data must be a data.table")
                        }

                        min_value <- min
                        acq_function$surrogate$update()
                        acq_function$update()
                        candidate <- acq_optimizer$optimize()
                        candidate <- round_to_nearest(candidate, metadata)
                        print(candidate)

                        tmp_archive = archive$clone(deep = TRUE)
                        acq_function$surrogate$archive = tmp_archive

                        min_values <- data.table()

                        # Apply the liar function to each column in archive$cols_y
                        for (col_name in archive$cols_y) {
                            min_values[, (col_name) := min_value(archive$data[[col_name]])]
                        }

                        candidate_new = candidate
                        for (i in seq_len(q)) {
                            
                            # Predict y or y1 y2 for the new candidate
                            prediction <- acq_function$surrogate$predict(candidate_new)

                            col_names <- c(paste0(archive$cols_y[1], "_mean"), paste0(archive$cols_y[1], "_se"))
                            if (length(archive$cols_y) > 1) {
                                col_names <- c(col_names, paste0(archive$cols_y[2], "_mean"), paste0(archive$cols_y[2], "_se"))
                            }
                            print("Column names:")
                            print(col_names)
                            # Add new columns to candidate
                            for (col_name in col_names) {
                            if (!col_name %in% names(candidate)) {
                                candidate[, (col_name) := NA]
                                }
                            }
                            print("candidate columns: ")
                            print(names(candidate))

                            # Add the predicted mean values [1] and their standard errors [2] to candidate_new
                            if (length(archive$cols_y) > 1) {
                            candidate_new[, (col_names) := .(prediction[[1]]$mean[1], prediction[[1]]$se[1],
                                                            prediction[[2]]$mean[1], prediction[[2]]$se[1])]
                            } else {
                            candidate_new[, (col_names) := .(prediction$mean[1], prediction$se[1])]
                            }
                            if (i > 1) {
                            candidate <- rbind(candidate, candidate_new, fill = TRUE)
                            }
                            if (i < q) {
                            candidate_new <- update_and_optimize(acq_function, acq_optimizer,
                                                                tmp_archive, candidate_new,
                                                                min_values, metadata)
                            }
                        }

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
                        set.seed(metadata$seed)
                        result <- load_archive_data(metadata)
                        data_with_preds <- load_predicted_data(metadata)
                        
                        # Ensure data_with_preds is a data.table
                        data_with_preds <- as.data.table(data_with_preds)

                        print("Data with preds: ")
                        print(data_with_preds)

                        print("NUM RANDOM LINES: ")
                        print(metadata$num_random_lines)

                        full_data <- as.data.table(data)
                        data <- tail(full_data, n=as.integer(metadata$num_random_lines))
                        print("Full data: ")
                        print(full_data)
                        
                        print("Tail Data: ")
                        print(data)

                        # Adjust the loop to update the last metadata$num_random_lines rows of data_with_preds
                        for (output_column_name in metadata$output_column_names) {
                            if (output_column_name %in% names(data_with_preds)) {
                                # Calculate the starting row index for the last metadata$num_random_lines rows
                                start_row <- max(1, nrow(data_with_preds) - as.integer(metadata$num_random_lines) + 1)
                                # Update the specified rows in data_with_preds
                                data_with_preds[(start_row:nrow(data_with_preds)), (output_column_name) := full_data[(start_row:nrow(data_with_preds)), ..output_column_name]]
                            }
                        }

                        print("Updated data_with_preds: ")
                        print(data_with_preds)

                        archive <- result[[1]]
                        acq_function <- result[[2]]
                        acq_optimizer <- result[[3]]

                        # Check if metadata$output_column_names is NULL or empty
                        if (is.null(metadata$output_column_names) ||
                            length(metadata$output_column_names) == 0) {
                            stop("metadata$output_column_names is NULL or empty")
                        }

                        # Check if all output_column_names exist in data
                        if (!all(metadata$output_column_names %in% names(data))) {
                            stop("Some names in metadata$output_column_names do not exist in data")
                        }

                        # Now you can safely call the add_evals method
                        archive$add_evals(xdt = data[, names(metadata$parameter_info), with=FALSE],
                                            ydt = data[, metadata$output_column_names, with=FALSE])
                        print("Model archive so far: ")
                        print(archive)
                        q = metadata$num_random_lines
                        result = add_evals_to_archive(archive, acq_function, acq_optimizer,
                                                        data, q, metadata)

                        candidate <- result[[1]]
                        archive <- result[[2]]
                        acq_function <- result[[3]]

                        print(result)

                        # x2 <- candidate[, names(metadata$parameter_info), with=FALSE]
                        # Check if all required columns exist in candidate
                        if (all(names(metadata$parameter_info) %in% names(candidate))) {
                            x2 <- candidate[, names(metadata$parameter_info), with=FALSE]
                        } else {
                            print("Error: One or more required columns do not exist in candidate.")
                            return()
                        }

                        print("New candidates: ")
                        print(x2)
                        print("New archive: ")
                        print(archive)

                        x2_dt <- as.data.table(x2)
                        full_data <- rbindlist(list(full_data, x2_dt), fill = TRUE)

                        print("Full data after adding new candidates: ")
                        print(full_data)

                        # Prepare candidate_with_preds
                        candidate_with_preds <- candidate[, -c(".already_evaluated","x_domain"), with = FALSE]

                        # Combine data_with_preds and candidate_with_preds
                        data_with_preds <- rbindlist(list(data_with_preds, candidate_with_preds), fill = TRUE)
                        print("Data with preds, new candidate: ")
                        print(data_with_preds)

                        # Combine the results into a list
                        result <- list(data_no_preds = full_data, data_with_preds = data_with_preds)

                        print("Returning data to streamlit")
                        return(result)

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

                data_no_preds = result["data_no_preds"]
                data_with_preds = result["data_with_preds"]

                metadata["batch_number"] = batch_number

                upload_metadata_to_bucket(metadata, batch_number)

                # Convert R data frame to pandas data frame
                df_no_preds = ro.conversion.get_conversion().rpy2py(data_no_preds)
                df_with_preds = ro.conversion.get_conversion().rpy2py(data_with_preds)

            num_rows_original = len(st.session_state.new_data)
            df_with_preds_subset = df_with_preds.iloc[:num_rows_original]

            # Replace -2147483648 with np.nan if -2147483648 exists in the DataFrame
            replace_value_with_nan(df_no_preds)
            replace_value_with_nan(df_with_preds)

            save_metadata(metadata, user_id, selected_table, batch_number)
            st.write("Saving metadata to local file")

            upload_local_to_bucket(bucket_name, user_id, selected_table, batch_number)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename_no_preds = f"{timestamp}_{batch_number}-data.csv"
            filename_with_preds = f"{timestamp}_{batch_number}-data-with-preds.csv"

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
            st.write(f"Files retrieved: {[file['name'] for file in downloaded_files]}")

            # Compress the files
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_zip = f"{timestamp}_{batch_number}_data.zip"
            zip_buffer = compress_files(downloaded_files)
            st.write(f"Files compressed into: {output_zip}")

            st.download_button(
                label="Download compressed data",
                data=zip_buffer.getvalue(),
                file_name=output_zip,
                mime="application/zip",
            )

            try:
                insert_data(selected_table, df_with_preds_subset, user_id, metadata)

                st.write(
                    "Data uploaded successfully! Head to `dashboard` to see your data!"
                )
            except Exception as e:
                st.write(f"Error uploading data: {e}")

            st.dataframe(df_no_preds)
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
