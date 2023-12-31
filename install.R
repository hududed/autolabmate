# TODO: This wont work since it prohibitively reinstalls the packages everytime the page is loaded

# List of packages to install
packages <- c("mlr3mbo", "mlr3", "mlr3learners", "bbotk")


install_if_not_installed <- function(package) {
  lib_path <- "./lib/R/library"
  
  # Create the directory if it doesn't exist
  if (!dir.exists(lib_path)) {
    dir.create(lib_path, recursive = TRUE)
  }
  
  # Install the package in the specified directory
  if (!require(package, lib.loc = lib_path, character.only = TRUE)) {
    install.packages(package, repos = "http://cran.rstudio.com/", lib = lib_path, dependencies=TRUE)
  }
}

# Install the packages
sapply(packages, install_if_not_installed)