# filepath: R/update.R
library(mlr3mbo)
library(mlr3)
library(mlr3learners)
library(bbotk)
library(data.table)
library(tibble)
library(R.utils)

helper_path <- file.path(getwd(), "R", "experiment_helpers.R")
helper_path <- file.path(getwd(), "R", "file_ops.R")
source(helper_path)

experiment <- function(data, metadata) {
  print("R: Metadata batch number:")
  print(metadata$batch_number)
  # The user passes in the new batch number (e.g., 2) via metadata.
  new_batch <- as.integer(metadata$batch_number)
  # Compute the previous batch number for loading archive files.
  old_batch <- as.character(new_batch - 1)

  print("R: New batch number:")
  print(new_batch)
  print("R: Old batch number:")
  print(old_batch)


  # Create a copy of metadata for loading:
  metadata_load <- metadata
  metadata_load$batch_number <- old_batch

  set.seed(metadata$seed)
  # Load archive and predicted data from the previous batch folder.
  result <- load_archive_data(metadata_load) # nolint
  data_with_preds <- load_predicted_data(metadata_load) # nolint
  data_with_preds <- as.data.table(data_with_preds)

  print("Data with preds: ")
  print(data_with_preds)

  print("NUM RANDOM LINES: ")
  print(metadata$num_random_lines)

  full_data <- as.data.table(data)
  data <- tail(full_data, n = as.integer(metadata$num_random_lines))
  print("Full data: ")
  print(full_data)
  print("Tail Data: ")
  print(data)
  for (output_column_name in metadata$output_column_names) {
    if (output_column_name %in% names(data_with_preds)) {
        start_row <- max(1, nrow(data_with_preds) - as.integer(metadata$num_random_lines) + 1) # nolint: line_length_linter
        data_with_preds[(start_row:nrow(data_with_preds)),
                        (output_column_name) := full_data[(start_row:nrow(data_with_preds)), ..output_column_name]] # nolint: line_length_linter
    }
  }
  print("Updated data_with_preds: ")
  print(data_with_preds)
  archive <- result[[1]]
  # Reinitialize surrogate and acquisition function based on learner_choice
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
  q <- metadata$num_random_lines
  result <- add_evals_to_archive(archive, acq_function, acq_optimizer, # nolint
                                 data, q, metadata)
  candidate <- result[[1]]
  archive <- result[[2]]
  acq_function <- result[[3]]

  print(result)

  if (all(names(metadata$parameter_info) %in% names(candidate))) {
    x2 <- candidate[, names(metadata$parameter_info), with = FALSE]
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
  candidate_with_preds <- candidate[, -c(".already_evaluated","x_domain"), with = FALSE] # nolint: line_length_linter
  data_with_preds <- rbindlist(list(data_with_preds, candidate_with_preds), fill = TRUE) # nolint: line_length_linter
  print("Data with preds, new candidate: ")
  print(data_with_preds)

  # Now update metadata so that saving uses the new batch number.
  metadata$batch_number <- as.character(new_batch)

  result <- list(data_no_preds = full_data, data_with_preds = data_with_preds)
  print("Returning data to streamlit")
  return(result)
}