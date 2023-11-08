# Autolabmate

Autolabmate is a tool designed for experimentalists. It leverages state-of-the-art Bayesian optimization to propose the next batch of experiments with the highest probability of improvement.

## Installation

Autolabmate uses Poetry for package management. If you don't have Poetry installed, you can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, you can install the packages needed for Autolabmate by running:

```bash
poetry install
```

## Setting up Supabase

Autolabmate uses Supabase for database management. To set it up, follow these steps:

1. Create a free account on [Supabase](https://supabase.io/).
2. Once you have created an account and logged in, create a new project.
3. After the project is created, you will be redirected to the project dashboard. Here, you can find your Supabase URL and anon key. These are needed to connect to your Supabase project.

## Configuring Environment Variables

Autolabmate uses environment variables for configuration. These are stored in a `.env` file. To set this up, follow these steps:

1. Open the `.env.default` file in the root of the project.
2. Fill in the `SUPABASE_URL`, `SUPABASE_KEY`, and `POSTGRES_KEY` variables with the values from your Supabase project.
3. Rename the `.env.default` file to `.env`.

Now, Autolabmate is configured and ready to run.


## Running Autolabmate
Autolabmate uses Streamlit for its user interface. To run Autolabmate, first activate the Poetry shell:
```
poetry shell
```

Then, start the Streamlit server:
```
streamlit run app.py
```


## Bayesian Optimization

Autolabmate uses Bayesian optimization for proposing the next batch of experiments. This optimization is performed using the `mlr3` and `mlr3mbo` packages.

- `mlr3` is a machine learning framework in R. You can find its documentation [here](https://mlr3book.mlr-org.com/).
- `mlr3mbo` is a package that extends `mlr3` with Bayesian optimization capabilities. You can find its documentation [here](https://mlr3mbo.mlr-org.com/).

These packages allow Autolabmate to efficiently and effectively propose the next batch of experiments.

