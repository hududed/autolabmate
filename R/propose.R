library(mlr3mbo)
library(mlr3)
library(mlr3learners)
library(bbotk)
library(data.table)

helper_path <- file.path(getwd(), "R", "experiment_helpers.R")
source(helper_path)

experiment <- function(data, metadata) { # nolint: cyclocomp_linter.
  set.seed(metadata$seed)
  data <- as.data.table(data)
  search_space_list <- list()

  for (param_name in names(metadata$parameter_info)) {
    param_info <- metadata$parameter_info[[param_name]]
    param_range <- metadata$parameter_ranges[[param_name]]

    # Check if param_info is 'object', if so, no need to convert to numeric
    print(paste("Adding parameter to search_space with id: ", param_name))
    if (param_info == "object") {
      search_space_list[[param_name]] <- p_fct(levels = param_range) # nolint
      next
    }
    if (param_info == "integer") {
      lower <- as.integer(param_range[1])
      upper <- as.integer(param_range[2])
    } else if (param_info == "float") {
      lower <- as.numeric(param_range[1])
      upper <- as.numeric(param_range[2])
    }
    if (is.na(lower) || is.na(upper)) {
      print(paste("lower or upper is NA for param_name:", param_name))
      next
    }
    if (param_info == "float") {
      search_space_list[[param_name]] <- p_dbl(lower = lower, upper = upper) # nolint
    } else if (param_info == "integer") {
      search_space_list[[param_name]] <- p_int(lower = lower, upper = upper) # nolint
    }
  }

  search_space <- do.call(ps, search_space_list) # nolint
  codomain_list <- list()
  for (output_name in names(metadata$directions)) {
    print(paste("Adding output to codomain with id: ", output_name))
    direction <- toString(metadata$directions[[output_name]])

    # Add the output to the codomain
    codomain_list[[output_name]] <- p_dbl(tags = direction) # nolint
  }
  codomain <- do.call(ps, codomain_list) # nolint
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
    if (metadata$learner_choice == "Gaussian Process") {
      surrogate <- srlrn(default_gp(), archive = archive)
    } else {
      surrogate <- srlrn(default_rf(), archive = archive)
    }
    acq_function <- acqf("ei", surrogate = surrogate)
  } else {
    if (metadata$learner_choice == "Gaussian Process") {
      surrogate <- srlrn(list(default_gp(), default_gp()), archive = archive)
    } else {
      surrogate <- srlrn(list(default_rf(), default_rf()), archive = archive)
    }
    acq_function <- acqf("ehvi", surrogate = surrogate)
  }
  acq_optimizer <- acqo(opt("random_search", batch_size = 1000),
                        terminator = trm("evals", n_evals = 1000),
                        acq_function = acq_function)
  q <- as.integer(metadata$num_random_lines)
  result <- add_evals_to_archive(archive, acq_function, # nolint
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
  data_no_preds <- rbindlist(list(data, x2_dt), fill = TRUE)
  candidate_with_preds <- candidate[, -c(".already_evaluated", "x_domain"), with = FALSE] # nolint: line_length_linter.
  data_with_preds <- rbindlist(list(data, candidate_with_preds), fill = TRUE)

  print("Data no preds: ")
  print(data_no_preds)
  print("Data with preds: ")
  print(data_with_preds)

  # Combine the results into a list
  result <- list(data_no_preds = data_no_preds,
                 data_with_preds = data_with_preds)
  return(result)
}