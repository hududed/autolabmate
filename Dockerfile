# Use an official Python runtime as a parent image
FROM python:3.10.14-slim

# Prevent Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    gfortran \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libpcre2-dev \
    libjpeg-dev \
    libx11-dev \
    libxt-dev \
    x11proto-core-dev \
    libpng-dev \
    libreadline-dev \
    libcairo2-dev \
    libtiff5-dev \
    libblas-dev \
    liblapack-dev \
    cmake \
    libgit2-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    tcl-dev \
    tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set R_HOME environment variable
ENV R_HOME=/usr/local/lib/R

# Set the Streamlit port
ENV PORT=8080

# Download and install R from source
RUN curl -O https://cran.r-project.org/src/base/R-4/R-4.3.1.tar.gz \
    && tar -xf R-4.3.1.tar.gz \
    && cd R-4.3.1 \
    && ./configure --with-blas --with-lapack --with-readline=no --with-x=no --enable-R-shlib \
    && make \
    && make install

# Check R version
RUN R --version

RUN R -e "install.packages('remotes', repos='http://cran.rstudio.com/')"

# Set the GitHub Personal Access Token to avoid rate limiting remotes::install_github
ARG GITHUB_PAT
ENV GITHUB_PAT=${GITHUB_PAT}

# Copy the list of installed R packages with versions
COPY installed_packages_with_versions.txt /tmp/installed_packages_with_versions.txt

# Copy the R script to install packages
COPY install_packages.R /tmp/install_packages.R

# Run the R script to install packages
RUN Rscript /tmp/install_packages.R

# RUN R -e "devtools::install_version('mlr3mbo', version = '0.2.2', repos='http://cran.rstudio.com/')"

# # Install specific mlr3 packages with exact versions
RUN R -e "remotes::install_github('mlr-org/mlr3extralearners@*release')"
# Verify installed versions
RUN R -e "installed_packages <- installed.packages(); print(installed_packages[installed_packages[, 'Package'] %in% c('mlr3', 'mlr3benchmark', 'mlr3learners', 'mlr3mbo', 'mlr3measures', 'mlr3misc', 'mlr3tuning', 'mlr3viz', 'mlr3extralearners'), c('Package', 'Version')])"

# Install Poetry
# Ensure Python and curl are installed
RUN apt-get update && apt-get install -y python3 curl

# Install Poetry with a retry mechanism
RUN curl -sSL https://install.python-poetry.org | python3 - || \
    (sleep 5 && curl -sSL https://install.python-poetry.org | python3 -)

# Set PATH for Poetry
ENV PATH="/root/.local/bin:$PATH"

# Ensure Poetry creates the virtual environment inside the project directory
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

COPY requirements.txt .
RUN pip install --default-timeout=100 --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN rm -rf /app/.venv && poetry install --no-root

# Make port 8501 available to the world outside this container
EXPOSE 8080

# Healthcheck for the Streamlit app
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]