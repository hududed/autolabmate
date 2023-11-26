# List of packages to install
packages <- c("mlr3mbo", "mlr3", "mlr3learners", "data.table", "tibble", "bbotk", "R.utils")

# Function to install a package if it's not already installed
install_if_not_installed <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, repos = "http://cran.rstudio.com/", lib="/usr/lib/R/library")
  }
}

# Install the packages
sapply(packages, install_if_not_installed)