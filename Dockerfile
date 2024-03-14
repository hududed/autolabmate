# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Prevent Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

WORKDIR /app

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
    && rm -rf /var/lib/apt/lists/*

# Set R_HOME environment variable
ENV R_HOME /usr/local/lib/R
# Download and install R from source
RUN curl -O https://cran.r-project.org/src/base/R-4/R-4.3.1.tar.gz \
    && tar -xf R-4.3.1.tar.gz \
    && cd R-4.3.1 \
    && ./configure --with-blas --with-lapack --with-readline=no --with-x=no --enable-R-shlib \
    && make \
    && make install

# Check R version
RUN R --version
# Install R packages
RUN R -e "packages <- c('usethis', 'textshaping', 'ragg', 'pkgdown', 'devtools', 'mlr3mbo', 'mlr3', 'mlr3learners', 'bbotk', 'ranger', 'broom', 'dbplyr', 'modelr', 'reprex', 'rvest', 'stringr', 'tidyr'); not_installed <- packages[!(packages %in% installed.packages()[,'Package'])]; if (length(not_installed) > 0) { install.packages(not_installed, repos='http://cran.rstudio.com/', dependencies=TRUE); print(paste('Installed missing packages:', paste(not_installed, collapse=', '))); }" \
    && R -e "devtools::install_version('mlr3mbo', version = '0.2.2', repos='http://cran.rstudio.com/')"

COPY requirements.txt .
RUN pip install --default-timeout=100 --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]