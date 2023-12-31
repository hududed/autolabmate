# autocsvgenerator

`autocsvgenerator` is a web app that generates CSV files with random data based on user-defined parameters. Users can specify the number of parameters, the type of each parameter (integer, float, or categorical), and the range of values for each parameter. The package also supports single- and multi-objective optimization, allowing users to specify the name of the objective column(s) in the CSV file.

To use `autocsvgenerator`, install the dependencies listed in the `pyproject.toml` file using Poetry and activate the Poetry environment, you can run the following command in the terminal:

```
poetry install
poetry shell # activate the environment
```

Otherwise, install `autocsvgenerator`, you can use pip:

```
pip install -r requirements.txt
```

Note that `autocsvgenerator` requires Python 3.9.7 or later.