# Read the list of packages and versions from the file
packages <- read.csv("/tmp/installed_packages_with_versions.txt", stringsAsFactors = FALSE)

# Install the packages with the specified versions
for (i in 1:nrow(packages)) {
  package <- packages[i, "Package"]
  version <- packages[i, "Version"]
  
  # Install the package with the specified version
  if (!require(package, character.only = TRUE)) {
    remotes::install_version(package, version = version, repos = "http://cran.rstudio.com/")
  }
}