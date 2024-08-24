import streamlit as st
from utils import (
    compress_files,
    display_dictionary,
    get_latest_data_and_metadata,
    get_latest_data_for_table,
    get_user_inputs,
    get_table_names,
    insert_data,
    py_dict_to_r_list,
    replace_value_with_nan,
    retrieve_and_download_files, 
    save_metadata,
    save_to_local,
    upload_local_to_bucket,
    upload_metadata_to_bucket,
    validate_inputs,
)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from datetime import datetime

st.title("Propose Experiment")


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()
    # Reset st.session_state.button_start_ml to False when the page is loaded
    if "button_start_ml" not in st.session_state or st.session_state.button_start_ml:
        st.session_state.button_start_ml = False
    
    if "download_triggered" not in st.session_state:
        st.session_state.download_triggered = False

    st.warning("This section is still under development.")

    user_id = st.session_state.user_id
    table_names = get_table_names(user_id)
    if not table_names:
        st.write("No tables found.")
        return

    # Get the latest metadata
    df_with_preds, metadata, latest_table = get_latest_data_and_metadata(user_id)
    columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
    df = df_with_preds[columns_to_keep]

    # Get the table name from the latest metadata
    default_table = latest_table
    selected_table = st.selectbox(
        "Select a table", table_names, index=table_names.index(default_table)
    )
    # Add seed input
    seed = st.number_input("Enter a seed", value=42, step=1)

    if selected_table != default_table:
        df_with_preds, metadata = get_latest_data_for_table(user_id, selected_table)
        columns_to_keep = metadata["X_columns"] + metadata["output_column_names"]
        df = df_with_preds[columns_to_keep]

    if selected_table:
        (
            batch_number,
            optimization_type,
            output_column_names,
            num_parameters,
            num_random_lines,
            parameter_info,
            parameter_ranges,
            directions,
            to_nearest,
        ) = get_user_inputs(df, metadata)

        # TODO: If batch 2 already uploaded, repeating batch 1 DOES NOT overwrite batch 2 but continue as if its batch 3
        if st.button("Validate"):
            validation_errors = validate_inputs(df, parameter_ranges)

            # Display validation errors or metadata
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

        # Add Start ML button
        if "button_start_ml" not in st.session_state:
            st.session_state.button_start_ml = False

        if st.button("Start ML"):
            st.session_state.button_start_ml = not st.session_state.button_start_ml

        if st.session_state.button_start_ml:
            # Define the mlr3 R functions
            ro.r(
                """
            library(mlr3mbo)
            library(mlr3)
            library(mlr3learners)
            library(bbotk)
            library(data.table)

            round_to_nearest <- function(x, metadata) {
                to_nearest = as.numeric(metadata$to_nearest)
                if (is.data.table(x) || is.data.frame(x)) {
                    x <- lapply(x, function(col) {
                    if (is.numeric(col)) {
                        return(round(col / to_nearest) * to_nearest)
                    } else {
                        return(col)
                    }
                    })
                    x <- setDT(x) # Convert the list to a data.table
                } else if (is.numeric(x)) {
                    x <- round(x / to_nearest) * to_nearest
                }  else {
                    stop("x must be a data.table, data.frame, or numeric vector")
                }
                return(x)
            }

            save_archive <- function(archive, acq_function, acq_optimizer, metadata) {
                # Get the current timestamp
                timestamp = format(Sys.time(), "%Y%m%d%H%M%S")

                # Define the directory path
                dir_path = paste0(metadata$bucket_name, "/", metadata$user_id, "/", metadata$table_name, "/", metadata$batch_number)

                # Create the directory if it doesn't exist
                if (!dir.exists(dir_path)) {
                    dir.create(dir_path, recursive = TRUE)
                }

                # Save the objects to files
                saveRDS(archive, paste0(dir_path,  "/archive-", timestamp, ".rds"))
                saveRDS(acq_function, paste0(dir_path, "/acqf-", timestamp, ".rds"))
                saveRDS(acq_optimizer, paste0(dir_path, "/acqopt-", timestamp, ".rds"))
            }

            update_and_optimize <- function(acq_function, acq_optimizer, tmp_archive, candidate_new, lie, metadata) {
                acq_function$surrogate$update()
                acq_function$update()
                tmp_archive$add_evals(xdt = candidate_new,
                                      xss_trafoed = transform_xdt_to_xss(candidate_new,
                                                                         tmp_archive$search_space),
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
                    # print("Column names:")
                    # print(col_names)
                    # Add new columns to candidate
                    for (col_name in col_names) {
                    if (!col_name %in% names(candidate)) {
                        candidate[, (col_name) := NA]
                        }
                    }
                    # print("candidate columns: ")
                    # print(names(candidate))

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
                data <- as.data.table(data) # data.csv is queried `table`

                # Initialize a list to store parameters for search_space
                search_space_list <- list()

                # Loop through metadata$parameter_info
                for (param_name in names(metadata$parameter_info)) {
                    # print(paste("Parameter name: ", param_name))
                    param_info <- metadata$parameter_info[[param_name]]
                    param_range <- metadata$parameter_ranges[[param_name]]

                    # print(paste("Param info:", param_info))
                    # print(paste("Param range:", param_range))

                    # Check if param_info is 'object', if so, no need to convert to numeric
                    print(paste("Adding parameter to search_space with id: ", param_name))
                    if (param_info == "object") {
                        search_space_list[[param_name]] <- p_fct(levels = param_range)
                        next
                    }

                    # Convert the results to appropriate type
                    if (param_info == "integer") {
                        lower = as.integer(param_range[1])
                        upper = as.integer(param_range[2])
                    } else if (param_info == "float") {
                        lower = as.numeric(param_range[1])
                        upper = as.numeric(param_range[2])
                    }

                    # Check if lower or upper is NA
                    if (is.na(lower) | is.na(upper)) {
                        print(paste("lower or upper is NA for param_name:", param_name))
                        next
                    }

                    # Add the parameter to the search space
                    if (param_info == "float") {
                        values = seq(lower, upper, by=as.numeric(metadata$to_nearest))
                        search_space_list[[param_name]] <- p_dbl(lower = lower, upper = upper)
                    } else if (param_info == "integer") {
                        search_space_list[[param_name]] <- p_int(lower = lower, upper = upper)
                    }
                }

                # Create the ParamSet for search_space using the ps() function
                search_space <- do.call(ps, search_space_list)

                # Initialize an empty ParamSet for the codomain
                codomain_list = list()

                # Loop through metadata$output_column_names
                for (output_name in names(metadata$directions)) {
                    print(paste("Adding output to codomain with id: ", output_name))
                    direction <- toString(metadata$directions[[output_name]])

                    # Add the output to the codomain
                    codomain_list[[output_name]] <- p_dbl(tags = direction)
                }

                # Create the ParamSet for codomain using the ps() function
                codomain <- do.call(ps, codomain_list)

                archive <- ArchiveBatch$new(search_space = search_space, codomain = codomain)

                print(metadata$output_column_names)

                print(unique(data$output2))


                # Use parameter_info in the subset operation
                archive$add_evals(xdt = data[, names(metadata$parameter_info), with = FALSE],
                                  ydt = data[, metadata$output_column_names, with = FALSE])

                print("Model archive so far: ")
                print(archive)
                # Determine the number of objectives
                num_objectives <- length(metadata$output_column_names)

                if (num_objectives == 1) {
                    #surrogate <- srlrn(lrn("regr.ranger"), archive = archive)
                    surrogate <- srlrn(default_rf(), archive = archive)
                    acq_function <- acqf("ei", surrogate = surrogate)
                } else {
                    surrogate <- srlrn(list(default_rf(), default_rf()), archive = archive)
                    acq_function <- acqf("ehvi", surrogate = surrogate)
                }
                # single or multi
                acq_optimizer <- acqo(opt("random_search", batch_size = 1000),
                                        terminator = trm("evals", n_evals = 1000),
                                        acq_function = acq_function)

                q <- as.integer(metadata$num_random_lines)

                result <- add_evals_to_archive(archive, acq_function,
                                               acq_optimizer, data, q, metadata)

                candidate <- result[[1]]
                archive <- result[[2]]
                acq_function <- result[[3]]

                print(result)

                x2 <- candidate[, names(metadata$parameter_info), with = FALSE]
                print("New candidates: ")
                print(x2)
                print("New archive: ")
                print(archive)

                x2_dt <- as.data.table(x2)

                # Create data_no_preds by combining data and x2_dt
                data_no_preds <- rbindlist(list(data, x2_dt), fill = TRUE)

                # Create data_with_preds by combining data and candidate
                candidate_with_preds <- candidate[, -c(".already_evaluated","x_domain"), with = FALSE]
                data_with_preds <- rbindlist(list(data, candidate_with_preds), fill = TRUE)

                print("Data no preds: ")
                print(data_no_preds)
                print("Data with preds: ")
                print(data_with_preds)

                # Combine the results into a list
                result <- list(data_no_preds = data_no_preds, data_with_preds = data_with_preds)

                print("Returning data to streamlit")
                return(result)
                }
            """
            )

            pandas2ri.activate()

            converter = ro.default_converter + pandas2ri.converter
            with converter.context():
                r_data = ro.conversion.get_conversion().py2rpy(df)

                r_metadata = py_dict_to_r_list(metadata)

                # Call the R function
                rsum = ro.r["experiment"]
                result = rsum(r_data, r_metadata)
                result_no_preds = result["data_no_preds"]
                result_with_preds = result["data_with_preds"]

                upload_metadata_to_bucket(metadata, batch_number)

                # Convert R data frame to pandas data frame
                df_no_preds = ro.conversion.get_conversion().rpy2py(result_no_preds)
                df_with_preds = ro.conversion.get_conversion().rpy2py(result_with_preds)

            # Replace -2147483648 with np.nan if -2147483648 exists in the DataFrame
            replace_value_with_nan(df_no_preds)
            replace_value_with_nan(df_with_preds)

            save_metadata(metadata, user_id, selected_table, batch_number)

            bucket_name = metadata["bucket_name"]
            batch_number = metadata["batch_number"]

            upload_local_to_bucket(
                bucket_name, user_id, selected_table, batch_number, file_extension = ".rds"
            )

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

            downloaded_files = retrieve_and_download_files(bucket_name, user_id, selected_table, batch_number) 
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
                mime="application/zip"
            )

            # TODO: NaN appears as min largest value
            # st.write(f"Table {table_name} has been updated.")
            st.session_state.update_clicked = False
            st.session_state.button_start_ml = False

            print(df_no_preds)
            st.write(df_no_preds)

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
