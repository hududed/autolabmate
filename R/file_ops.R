# filepath: R/file_ops.R
library(mlr3mbo)
library(mlr3)
library(mlr3learners)
library(bbotk)
library(data.table)
library(tibble)
library(R.utils)

find_latest_file <- function(files, pattern) {
  print("------ Finding latest file --------")
  matched_files <- grep(pattern, files, value = TRUE)
  if (length(matched_files) == 0) {
    stop(paste("No", pattern, "files found"))
  }
  timestamps <- as.numeric(gsub(".*-(\\d+).*", "\\1", matched_files))
  latest_index <- which.max(timestamps)
  return(matched_files[latest_index])
}

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

load_files <- function(metadata, pattern, file_type) {
  print("------ Loading files --------")
  # Build the directory using the base_dir from metadata
  target_dir <- file.path(
    as.character(metadata$base_dir),
    as.character(metadata$bucket_name),
    as.character(metadata$user_id),
    as.character(metadata$table_name),
    as.character(metadata$batch_number)
  )
  if (!dir.exists(target_dir)) {
    stop(paste("Directory does not exist:", target_dir))
  }
  files <- list.files(path = target_dir,
                      pattern = paste0("*.", file_type),
                      full.names = TRUE)
  latest_file <- find_latest_file(files, pattern)
  return(load_file(latest_file, file_type))
}

load_predicted_data <- function(metadata) {
  print("------ Loading predicted data --------")
  metadata$batch_number <- as.integer(metadata$batch_number)
  return(load_files(metadata, "with-preds", "csv"))
}

load_archive_data <- function(metadata) {
  print("------ Loading archive data --------")
  acqf <- load_files(metadata, "acqf-", "rds")
  acqopt <- load_files(metadata, "acqopt-", "rds")
  archive <- load_files(metadata, "archive-", "rds")
  acqf$surrogate$archive <- archive
  return(list(archive, acqf, acqopt))
}