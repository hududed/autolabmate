import streamlit as st
from utils import (
    engine,
    inspector,
    get_user_inputs,
    validate_inputs,
    display_dictionary,
    save_and_upload_results,
    save_metadata,
    py_dict_to_r_list,
    upload_local_to_bucket,
    save_to_local,
    replace_value_with_nan,
)
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from st_pages import hide_pages

st.title("Propose Experiment")


def main():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()
    # Reset st.session_state.button_start_ml to False when the page is loaded
    if "button_start_ml" not in st.session_state or st.session_state.button_start_ml:
        st.session_state.button_start_ml = False
    # choose table in supabase via streamlit dropdown
    # inspector = inspect(engine)
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
        # Load the selected table
        query = f"SELECT * FROM {table_name};"
        table = pd.read_sql_query(query, engine)

        (
            batch_number,
            optimization_type,
            output_column_names,
            num_parameters,
            num_random_lines,
            parameter_info,
            parameter_ranges,
            direction,
        ) = get_user_inputs(table)

        # Add validate button
        if st.button("Validate"):
            validation_errors = validate_inputs(table, parameter_ranges)

            # Display validation errors or metadata
            if validation_errors:
                for error in validation_errors:
                    st.write(error)
            else:
                st.write("Validation passed.")
                st.session_state.metadata = display_dictionary(
                    batch_number,
                    table_name,
                    optimization_type,
                    output_column_names,
                    num_parameters,
                    num_random_lines,
                    parameter_info,
                    parameter_ranges,
                    direction,
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
            library(tibble)

            save_archive <- function(archive, acq_function, acq_optimizer, metadata) {
                # Get the current timestamp
                timestamp = format(Sys.time(), "%Y%m%d%H%M%S")

                # Define the directory path
                dir_path = paste0(metadata$bucket_name, "/", metadata$table_name, "/", metadata$batch_number)
                
                # Create the directory if it doesn't exist
                if (!dir.exists(dir_path)) {
                    dir.create(dir_path, recursive = TRUE)
                }
                
                # Save the objects to files
                saveRDS(archive, paste0(dir_path,  "/archive-", timestamp, ".rds"))
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
                data = as.data.table(data) # data.csv is queried `table`

                # retrieve this from metadata parameter_ranges
                search_space = ParamSet$new(params = list())
                # Loop through metadata$parameter_info
                for (param_name in names(metadata$parameter_info)) {
                    print(param_name)
                    param_info = metadata$parameter_info[[param_name]]
                    param_range = metadata$parameter_ranges[[param_name]]

                    print(param_info)
                    print(param_range)

                    # Check if param_info is 'object', if so, no need to convert to numeric
                    if (param_info == 'object') {
                        search_space$add(ParamFct$new(id = param_name, levels = param_range))
                        next
                    }

                    # Remove the parentheses and split the string at the comma
                    param_range_split = strsplit(gsub("[()]", "", param_range), ",")[[1]]

                    # Convert the results to appropriate type
                    if (param_info == 'integer') {
                        lower = as.integer(param_range_split[1])
                        upper = as.integer(param_range_split[2])
                    } else if (param_info == 'float') {
                        lower = as.numeric(param_range_split[1])
                        upper = as.numeric(param_range_split[2])
                    }

                    # Check if lower or upper is NA
                    if (is.na(lower) | is.na(upper)) {
                        print(paste("lower or upper is NA for param_name:", param_name))
                        next
                    }

                    # Add the parameter to the search space
                    if (param_info == 'float') {
                        search_space$add(ParamDbl$new(id = param_name, lower = lower, upper = upper)) # TODO: Trafo since levels are inf, but id doesnt work with p_int
                    } else if (param_info == 'integer') {
                        search_space$add(ParamInt$new(id = param_name, lower = lower, upper = upper))
                    }
                }
                # Initialize an empty ParamSet for the codomain
                codomain = ParamSet$new(params = list())

                # Loop through metadata$output_column_names
                for (output_name in metadata$output_column_names) {
                    # Add the output to the codomain
                    codomain$add(ParamDbl$new(id = output_name, tags = metadata$direction))
                }

                archive = Archive$new(search_space = search_space, codomain = codomain)

                # Use parameter_info in the subset operation
                archive$add_evals(xdt = data[, names(metadata$parameter_info), with=FALSE], ydt = data[, metadata$output_column_names, with=FALSE])
            

                print("Model archive so far: ")
                print(archive)
                surrogate = srlrn(lrn("regr.ranger"), archive = archive)
                acq_function = acqf("ei", surrogate = surrogate)
                acq_optimizer = acqo(opt("random_search", batch_size = 1000),
                                    terminator = trm("evals", n_evals = 1000),
                                    acq_function = acq_function)
                q = as.integer(metadata$num_random_lines)
                # print(q)
                # print(acq_function)
                # print(acq_optimizer)
                # print(data)
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
                r_data = ro.conversion.get_conversion().py2rpy(table)
                r_metadata = py_dict_to_r_list(st.session_state.metadata)

                # Call the R function
                rsum = ro.r["experiment"]
                result = rsum(r_data, r_metadata)

                save_and_upload_results(st.session_state.metadata, batch_number)

                # Convert R data frame to pandas data frame
                df = ro.conversion.get_conversion().rpy2py(result)

                # Replace -2147483648 with np.nan if -2147483648 exists in the DataFrame
                replace_value_with_nan(df)

                save_metadata(st.session_state.metadata, table_name, batch_number)

                bucket_name = st.session_state.metadata["bucket_name"]
                batch_number = st.session_state.metadata["batch_number"]

                upload_local_to_bucket(bucket_name, table_name, batch_number)

                output_file_name = f"{batch_number}-data.csv"
                save_to_local(
                    bucket_name, table_name, output_file_name, df, batch_number
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
                    f"Files downloaded to local directory: /{bucket_name}/{table_name}/{batch_number}"
                )
                st.write(
                    "Run the proposed batch of experiments and proceed to `update` the model."
                )


if __name__ == "__main__":
    main()
