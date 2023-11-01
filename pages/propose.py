import streamlit as st
from utils import (
    engine,
    inspect,
    get_user_inputs,
    validate_inputs,
    display_dictionary,
)
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import ListVector, StrVector

st.title("Propose Experiment")


def main():
    # choose table in supabase via streamlit dropdown
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        table_name = st.selectbox("Select a table to display", table_names)

        # Load the selected table
        query = f"SELECT * FROM {table_name};"
        table = pd.read_sql_query(query, engine)
        (
            optimization_type,
            output_column_names,
            num_parameters,
            num_random_lines,
            parameter_info,
            parameter_ranges,
        ) = get_user_inputs(table, table_name)

        # Add validate button
        if st.button("Validate"):
            validation_errors = validate_inputs(table, parameter_ranges)

            # Display validation errors or metadata
            if validation_errors:
                for error in validation_errors:
                    st.write(error)
            else:
                st.write("Validation passed.")
                metadata = display_dictionary(
                    table_name,
                    optimization_type,
                    output_column_names,
                    num_parameters,
                    num_random_lines,
                    parameter_info,
                    parameter_ranges,
                )

            # Define the mlr3 R functions
            robjects.r(
                """
            library(mlr3mbo)
            library(mlr3)
            library(mlr3learners)
            library(bbotk)
            library(data.table)
            library(tibble)


            load_archive <- function() {
                archive = readRDS("archive.rds") # load from test-bucket objects e.g. test-bucket/{table_name}/archive.rds
                acq_function = readRDS("acqf.rds") # load from test-bucket objects e.g. test-bucket/{table_name}/acqf.rds
                acq_optimizer = readRDS("acqopt.rds") # load from test-bucket objects e.g. test-bucket/{table_name}/acqopt.rds
                acq_function$surrogate$archive = archive
                return(list(archive, acq_function, acq_optimizer))
            }

            save_archive <- function(archive, acq_function, acq_optimizer) {
                saveRDS(archive, "archive.rds") # save to test-bucket objects e.g. test-bucket/{table_name}/archive.rds
                saveRDS(acq_function, "acqf.rds") # save to test-bucket objects e.g. test-bucket/{table_name}/acqf.rds
                saveRDS(acq_optimizer, "acqopt.rds") # save to test-bucket objects e.g. test-bucket/{table_name}/acqopt.rds
            }

            add_evals_to_archive <- function(archive, acq_function, acq_optimizer, data, q) {
                lie = data.table()
                liar = min
                acq_function$surrogate$update()
                acq_function$update()
                candidate = acq_optimizer$optimize()
                tmp_archive = archive$clone(deep = TRUE)
                acq_function$surrogate$archive = tmp_archive
                lie[, archive$cols_y := liar(archive$data[[archive$cols_y]])]
                candidate_new = candidate

                # Check if lie is a data.table
                if (!is.data.table(lie)) {
                    stop("lie is not a data.table")
                }

                # loops through batch size q, e.g. q should equal num_random_lines
                for (i in seq_len(q)[-1L]) {
                    tmp_archive$add_evals(xdt = candidate_new, xss_trafoed = transform_xdt_to_xss(candidate_new, tmp_archive$search_space), ydt = lie)
                    acq_function$surrogate$update()
                    acq_function$update()
                    candidate_new = acq_optimizer$optimize()
                    candidate = rbind(candidate, candidate_new)
                }
                
                candidate$param2 <- format(round(candidate$param2, 2), nsmall = 2)
                save_archive(tmp_archive, acq_function, acq_optimizer)
                return(list(candidate, tmp_archive, acq_function))
            }

            experiment <- function(data, metadata, s) {
            if(s==1) {
                require(XML)
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

                    # Check if param_info is 'category', if so, no need to convert to numeric
                    if (param_info == 'category') {
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

                    print(typeof(lower))
                    print(typeof(upper))

                    # Check if lower or upper is NA
                    if (is.na(lower) | is.na(upper)) {
                        print(paste("lower or upper is NA for param_name:", param_name))
                        next
                    }

                    # Add the parameter to the search space
                    if (param_info == 'float') {
                        search_space$add(ParamDbl$new(id = param_name, lower = lower, upper = upper))
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
                # print(search_space)
                # print(codomain)

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
                print(q)
                print(acq_function)
                print(acq_optimizer)
                print(data)
                result = add_evals_to_archive(archive, acq_function, acq_optimizer, data, q)
            } else {
                result = load_archive()
                lines_to_keep <- metadata$num_random_lines
                num_lines <- countLines(data) # count number of rows in table
                data <- as.data.table(read.csv(data, header=FALSE, skip = num_lines-lines_to_keep)) 
                names(data) <- c(metadata$parameter_info, metadata$output_column_names)
                    
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


                # Now you can safely call the add_evals method
                archive$add_evals(xdt = data[, param_names, with=FALSE], ydt = data[, metadata$output_column_names, with=FALSE])
                print("Model archive so far: ")
                print(archive)
                q = metadata$num_random_lines
                result = add_evals_to_archive(archive, acq_function, acq_optimizer, data, q)
            }
            candidate = result[[1]]
            archive = result[[2]]
            acq_function = result[[3]]

            print(result)

            x2 <- candidate[, names(metadata$parameter_info)]
            print("New candidates: ")
            print(x2)

            x2_dt <- as.data.table(x2)
            data <- rbind(data, x2_dt)
            print("Returning data to streamlit")

            }
            """
            )
            # TODO: Candidate current output
            # [[1]]
            # param1 param2  x_domain    acq_ei .already_evaluated
            # 1:     32  38.42 <list[2]> 0.1048288              FALSE
            # param3 is missing, and candidate needs correction

            # print(table)
            pandas2ri.activate()
            # metadata = pd.DataFrame()

            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(table)

                def py_dict_to_r_list(py_dict):
                    r_list = {}
                    for k, v in py_dict.items():
                        if isinstance(v, dict):
                            r_list[k] = py_dict_to_r_list(v)
                        elif isinstance(v, list):
                            r_list[k] = StrVector(v)
                        else:
                            r_list[k] = StrVector([str(v)])
                    return ListVector(r_list)

                r_metadata = py_dict_to_r_list(metadata)
                print(r_metadata)
                # print(r_param_names)

                # Call the R function
                rsum = robjects.r["experiment"]
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    rsum(r_data, r_metadata, 1)


if __name__ == "__main__":
    main()
