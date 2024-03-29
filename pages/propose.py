import streamlit as st
from utils import (
    get_user_inputs,
    validate_inputs,
    display_dictionary,
    upload_metadata,
    save_metadata,
    py_dict_to_r_list,
    upload_local_to_bucket,
    save_to_local,
    replace_value_with_nan,
    get_table_names,
    get_latest_row,
)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

st.title("Propose Experiment")


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
    # Add seed input
    seed = st.number_input("Enter a seed", value=42, step=1)

    if selected_table:
        df = get_latest_row(user_id, selected_table)
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
        ) = get_user_inputs(df)

        # Add validate button
        if st.button("Validate"):
            validation_errors = validate_inputs(df, parameter_ranges)

            # Display validation errors or metadata
            if validation_errors:
                for error in validation_errors:
                    st.write(error)
            else:
                st.write("Validation passed.")
                st.session_state.metadata = display_dictionary(
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
                )

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
            # library(tibble)
            
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
                tmp_archive$add_evals(xdt = candidate_new, xss_trafoed = transform_xdt_to_xss(candidate_new, tmp_archive$search_space), ydt = lie)
                candidate_new = acq_optimizer$optimize()
                candidate_new = round_to_nearest(candidate_new, metadata)
                return(candidate_new)
            }

            add_evals_to_archive <- function(archive, acq_function, acq_optimizer, data, q, metadata) {
                lie <- data.table()
                liar <- min
                acq_function$surrogate$update()
                acq_function$update()
                candidate <- acq_optimizer$optimize()
                candidate <- round_to_nearest(candidate, metadata)
                print(candidate)
                tmp_archive = archive$clone(deep = TRUE)
                acq_function$surrogate$archive = tmp_archive

                # Apply the liar function to each column in archive$cols_y
                for (col_name in archive$cols_y) {
                    lie[, (col_name) := liar(archive$data[[col_name]])]
                }

                # lie[, archive$cols_y := liar(archive$data[[archive$cols_y]])]
                candidate_new = candidate

                # Check if lie is a data.table
                if (!is.data.table(lie)) {
                    stop("lie is not a data.table")
                }
                candidate_new = candidate
                for (i in seq_len(q)[-1L]) {
                    candidate_new = update_and_optimize(acq_function, acq_optimizer,
                                                        tmp_archive, candidate_new, 
                                                        lie, metadata)
                    candidate = rbind(candidate, candidate_new)
                }
                candidate_new = update_and_optimize(acq_function, acq_optimizer, 
                                                    tmp_archive, candidate_new, 
                                                    lie, metadata)
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

                # retrieve this from metadata parameter_ranges
                search_space <- ParamSet$new(params = list())
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
                        search_space$add(ParamFct$new(id = param_name, levels = param_range))
                        next
                    }

                    # Remove the parentheses and split the string at the comma
                    param_range_split = strsplit(gsub("[()]", "", param_range), ",")[[1]]

                    # Convert the results to appropriate type
                    if (param_info == "integer") {
                        lower = as.integer(param_range_split[1])
                        upper = as.integer(param_range_split[2])
                    } else if (param_info == "float") {
                        lower = as.numeric(param_range_split[1])
                        upper = as.numeric(param_range_split[2])
                    }

                    # Check if lower or upper is NA
                    if (is.na(lower) | is.na(upper)) {
                        print(paste("lower or upper is NA for param_name:", param_name))
                        next
                    }

                    # Add the parameter to the search space
                    if (param_info == "float") {
                        # Check if metadata$to_nearest is numeric
                        if (is.numeric(as.numeric(metadata$to_nearest))) {
                            values = seq(lower, upper, by=as.numeric(metadata$to_nearest))
                            search_space$add(ParamDbl$new(id = param_name, 
                                                          lower = lower, upper = upper))
                        } else {
                            print(paste("metadata$to_nearest is not numeric for 
                                         param_name:", param_name))
                        }
                    } else if (param_info == "integer") {
                        search_space$add(ParamInt$new(id = param_name, 
                                                      lower = lower, upper = upper))
                    }
                }
                # Initialize an empty ParamSet for the codomain
                codomain = ParamSet$new(params = list())

                # Loop through metadata$output_column_names
                for (i in seq_along(metadata$output_column_names)) {
                    output_name <- metadata$output_column_names[i]
                    print(paste("Adding output to codomain with id: ", output_name))
                    direction <- metadata$directions[i]
                    # Add the output to the codomain
                    codomain$add(ParamDbl$new(id = output_name, tags = direction))
                }
                # for (output_name in metadata$output_column_names) {
                #     # Add the output to the codomain
                #     codomain$add(ParamDbl$new(id = output_name, tags = metadata$direction))
                # }

                archive <- Archive$new(search_space = search_space, codomain = codomain)

                # Print metadata$output_column_names
                print(metadata$output_column_names)

                # Print unique values of output2
                print(unique(data$output2))
                
                # Convert output columns to doubles
                output_data <- data.table(sapply(data[, metadata$output_column_names,
                                                      with=FALSE], as.double))

                # Use parameter_info in the subset operation
                archive$add_evals(xdt = data[, names(metadata$parameter_info), with=FALSE],
                                  ydt = output_data)
            

                print("Model archive so far: ")
                print(archive)
                # Determine the number of objectives
                num_objectives <- length(metadata$output_column_names)

                if (num_objectives == 1) {
                    surrogate <- srlrn(lrn("regr.ranger"), archive = archive)
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
                #   print(q)
                #   print(acq_function)
                #   print(acq_optimizer)
                #   print(data)

                result <- add_evals_to_archive(archive, acq_function,
                                                acq_optimizer, data, q, metadata)

                candidate <- result[[1]]
                archive <- result[[2]]
                acq_function <- result[[3]]
                # surrogate = srlrn(lrn("regr.ranger"), archive = archive)
                # acq_function = acqf("ei", surrogate = surrogate)
                # acq_optimizer = acqo(opt("random_search", batch_size = 1000),
                #                     terminator = trm("evals", n_evals = 1000),
                #                     acq_function = acq_function)
                # q = as.integer(metadata$num_random_lines)
                # # print(q)
                # # print(acq_function)
                # # print(acq_optimizer)
                # # print(data)
                # result = add_evals_to_archive(archive, acq_function, acq_optimizer, data, q, metadata)

                # candidate = result[[1]]
                # archive = result[[2]]
                # acq_function = result[[3]]

                print(result)

                x2 <- candidate[, names(metadata$parameter_info), with = FALSE]
                print("New candidates: ")
                print(x2)
                print("New archive: ")
                print(archive)

                x2_dt <- as.data.table(x2)
                data <- rbindlist(list(data, x2_dt), fill = TRUE)
                print(data)
                print("Returning data to streamlit")
                return(data)

                }
            """
            )

            pandas2ri.activate()

            converter = ro.default_converter + pandas2ri.converter
            with converter.context():
                r_data = ro.conversion.get_conversion().py2rpy(df)
                r_metadata = py_dict_to_r_list(st.session_state.metadata)

                # Call the R function
                rsum = ro.r["experiment"]
                result = rsum(r_data, r_metadata)

                upload_metadata(st.session_state.metadata, batch_number)

                # Convert R data frame to pandas data frame
                df = ro.conversion.get_conversion().rpy2py(result)

                # Replace -2147483648 with np.nan if -2147483648 exists in the DataFrame
                replace_value_with_nan(df)

                save_metadata(
                    st.session_state.metadata, user_id, selected_table, batch_number
                )

                bucket_name = st.session_state.metadata["bucket_name"]
                batch_number = st.session_state.metadata["batch_number"]

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
                upload_local_to_bucket(
                    bucket_name,
                    user_id,
                    selected_table,
                    batch_number,
                    file_extension=".csv",
                )

                # TODO: NaN appears as min largest value
                # st.write(f"Table {table_name} has been updated.")
                st.session_state.update_clicked = False
                st.session_state.button_start_ml = False

                print(df)
                st.write(df)

                # store metadata in session_state
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
