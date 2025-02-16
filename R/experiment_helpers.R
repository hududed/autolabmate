# filepath = "R/experiment_helpers.R"
library(mlr3mbo)
library(mlr3)
library(mlr3learners)
library(bbotk)
library(data.table)


round_to_nearest <- function(x, metadata) {
  print("######### ROUNDING TO NEAREST #########")
  to_nearest <- metadata$to_nearest
  x_columns <- metadata$X_columns

  if (is.data.table(x) || is.data.frame(x)) {
    for (col_name in names(x)) {
      if (col_name %in% x_columns) {
        col <- x[[col_name]]
        if (is.numeric(col)) {
          nearest <- as.numeric(to_nearest[[col_name]])
          print(paste("Column:", col_name))
          print(paste("Nearest value:", nearest))
          print("Values before rounding:")
          print(col)
          rounded_val <- round(col / nearest) * nearest
          if (nearest < 1) {
            decimals <- nchar(sub("0\\.", "", as.character(nearest)))
            rounded_val <- as.numeric(format(rounded_val, nsmall = decimals))
          }
          x[[col_name]] <- rounded_val
          print("Values after rounding:")
          print(x[[col_name]])
        }
      } else {
        print(paste("Skipping column:", col_name))
      }
    }
  } else if (is.numeric(x)) {
    nearest <- as.numeric(to_nearest)
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

save_archive <- function(archive, acq_function, acq_optimizer, metadata) {
  print(" ######## SAVING ARCHIVE, ACQ_FUNCTION, AND ACQ_OPTIMIZER ########")
  timestamp <- format(Sys.time(), "%Y%m%d-%H%M")
  dir_path <- file.path(metadata$base_dir,
                        metadata$bucket_name,
                        metadata$user_id,
                        metadata$table_name,
                        metadata$batch_number)
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
  }
  saveRDS(archive, paste0(dir_path,  "/archive-", timestamp, ".rds"))
  saveRDS(acq_function, paste0(dir_path, "/acqf-", timestamp, ".rds"))
  saveRDS(acq_optimizer, paste0(dir_path, "/acqopt-", timestamp, ".rds"))
  print(paste0("Archive, acq_function, and acq_optimizer saved to", dir_path))
}

update_and_optimize <- function(acq_function, acq_optimizer, tmp_archive, candidate_new, lie, metadata) { # nolint: line_length_linter.
  print("######### UPDATING AND OPTIMIZING #########")
  acq_function$surrogate$update()
  acq_function$update()
  tmp_archive$add_evals(xdt = candidate_new,
                        xss_trafoed = transform_xdt_to_xss(candidate_new, tmp_archive$search_space), # nolint: line_length_linter.
                        ydt = lie)
  candidate_new <- acq_optimizer$optimize()
  candidate_new <- round_to_nearest(candidate_new, metadata)
  return(candidate_new)
}

add_evals_to_archive <- function(archive, acq_function, acq_optimizer, data, q, metadata) { # nolint
  print("######### ADDING EVALS TO ARCHIVE #########")
  if (!is.data.table(archive$data)) {
    stop("archive$data must be a data.table")
  }

  acq_function$surrogate$update()
  acq_function$update()

  candidate <- acq_optimizer$optimize()
  candidate <- round_to_nearest(candidate, metadata)
  print("Candidate after rounding:")
  print(candidate)

  tmp_archive <- archive$clone(deep = TRUE)
  acq_function$surrogate$archive <- tmp_archive

  min_value <- min
  min_values <- data.table()
  for (col_name in archive$cols_y) {
    min_values[, (col_name) := min_value(archive$data[[col_name]])]
  }

  candidate_new <- candidate

  print("Candidate_new before update loop:")
  print(candidate_new)

  for (i in seq_len(q)) {
    prediction <- acq_function$surrogate$predict(candidate_new)
    col_names <- c(paste0(archive$cols_y[1], "_mean"),
                   paste0(archive$cols_y[1], "_se"))
    if (length(archive$cols_y) > 1) {
      col_names <- c(col_names, paste0(archive$cols_y[2], "_mean"),
                     paste0(archive$cols_y[2], "_se"))
    }
    for (col_name in col_names) {
      if (!col_name %in% names(candidate_new)) {
        candidate_new[, (col_name) := NA]
      }
    }
    if (length(archive$cols_y) > 1) {
      candidate_new[, (col_names) := .(prediction[[1]]$mean[1], prediction[[1]]$se[1], # nolint
                                       prediction[[2]]$mean[1], prediction[[2]]$se[1])] # nolint: line_length_linter.
    } else {
      candidate_new[, (col_names) := .(prediction$mean[1], prediction$se[1])] # nolint
    }

    print(paste("Iteration", i, "candidate_new before update_optimize:"))
    print(candidate_new)

    if (i > 1) {
      candidate <- rbind(candidate, candidate_new, fill = TRUE)
    } else {
      candidate <- candidate_new
    }
    if (i < q) {
      candidate_new <- update_and_optimize(acq_function, acq_optimizer,
                                           tmp_archive, candidate_new,
                                           min_values, metadata)
    }
    print("Candidate_new after update_and_optimize:")
    print(candidate_new)
  }

  for (col in names(candidate_new)) {
    if (is.double(candidate_new[[col]])) {
      candidate_new[[col]] <- format(round(candidate_new[[col]], 2),
                                     nsmall = 2)
    }
  }
  save_archive(archive, acq_function, acq_optimizer, metadata)
  return(list(candidate, archive, acq_function))
}