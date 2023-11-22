import streamlit as st
import supabase
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from typing import Dict, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import rpy2.robjects as ro

from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor


load_dotenv()
# Load Supabase credentials from .env file
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase_client = supabase.create_client(supabase_url, supabase_key)
PG_PASS = os.getenv("PG_PASS")
DATABASE_URL = (
    f"postgresql://postgres:{PG_PASS}@db.zugnayzgayyoveqcmtcd.supabase.co:5432/postgres"
)
engine = create_engine(DATABASE_URL, echo=True)
inspector = inspect(engine)


# Define routes
def create_account(email, password):
    # Create new account
    response = supabase_client.auth.sign_up({"email": email, "password": password})
    return response


def login(credentials):
    # Login to existing account
    response = supabase_client.auth.sign_in_with_password(credentials)
    return response


def upload_to_bucket(bucket_name, table_name, file_name, file_content, batch_number=1):
    new_file_name = f"{table_name}/{batch_number}/{file_name}"
    print(new_file_name)

    st.write(
        f'Uploading file "{file_name}" to bucket "{bucket_name}" as "{new_file_name}"'
    )

    try:
        # Upload file to bucket
        supabase_client.storage.from_(bucket_name).upload(new_file_name, file_content)
        st.write(
            f'Uploading file "{file_name}" to bucket "{bucket_name}" as "{new_file_name}"'
        )
        st.write(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
    except Exception as e:
        if "Duplicate" in str(e):
            print(
                f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
            )
            st.write(
                f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
            )
        else:
            raise e


def save_to_local(bucket_name, table_name, file_name, df, batch_number=1):
    table_name = table_name.lower()
    new_file_name = f"{bucket_name}/{table_name}/{batch_number}/{file_name}"
    print(new_file_name)

    try:
        # Create necessary directories
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)

        # Save DataFrame to CSV file
        df.to_csv(new_file_name, index=False)

        print(f'"{new_file_name}" saved locally at "{os.path.abspath(new_file_name)}"')
    except Exception as e:
        raise e


def save_metadata(metadata, table_name, batch_number=1, bucket_name="test-bucket"):
    """
    Saves metadata to an in-memory file.

    Args:
        metadata (dict): The metadata to save.
        table_name (str): The name of the table.
        batch_number (int): The batch number. Defaults to 1.
        bucket_name (str): The name of the bucket. Defaults to 'test-bucket'.
    """
    # Convert the metadata dictionary to a JSON string
    json_metadata = json.dumps(metadata)

    # Save the JSON string to an in-memory file
    with open(f"{bucket_name}/{table_name}/{batch_number}/metadata.json", "w") as f:
        f.write(json_metadata)


def load_metadata(
    table_name, batch_number=1, bucket_name="test-bucket"
) -> Dict[str, Any]:
    """
    Loads metadata from an in-memory file.

    Args:
        table_name (str): The name of the table.
        bucket_name (str): The name of the bucket. Defaults to 'test-bucket'.

    Returns:
        dict: The loaded metadata.
    """
    # Load the JSON string from the in-memory file
    with open(f"{bucket_name}/{table_name}/{batch_number}/metadata.json", "r") as f:
        json_metadata = f.read()

    # Convert the JSON string to a dictionary
    metadata = json.loads(json_metadata)

    return metadata


def upload_local_to_bucket(
    bucket_name, table_name, batch_number=1, file_extension=".rds"
):
    # Extract file name from file path
    base_path = Path(f"{bucket_name}/{table_name}/{batch_number}")
    files = [file for file in base_path.glob(f"*{file_extension}")]

    for file in files:
        file_name = file.name
        new_file_name = f"{table_name}/{batch_number}/{file_name}"

        # Read file content
        with open(file, "rb") as f:
            file_content = f.read()

        try:
            # Upload file to bucket
            supabase_client.storage.from_(bucket_name).upload(
                new_file_name, file_content
            )
            print(new_file_name)
            st.write(
                f'Uploading file "{file_name}" to bucket "{bucket_name}" as "{new_file_name}"'
            )
            st.write(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
        except Exception as e:
            if "Duplicate" in str(e):
                print(
                    f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
                )
                st.write(
                    f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
                )
            else:
                raise e


def save_and_upload_results(metadata, batch_number=1):
    # Convert the metadata dictionary to a JSON string and encode it to bytes
    metadata_content = json.dumps(metadata).encode()

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Define the file name
    file_name = f"metadata-{timestamp}.json"

    # Upload the metadata to the bucket
    upload_to_bucket(
        metadata["bucket_name"],
        metadata["table_name"],
        file_name,
        metadata_content,
        batch_number,
    )


def create_experiments_table() -> None:
    # Create the profiles table if it doesn't exist
    create_table_stmt = """
    CREATE TABLE IF NOT EXISTS experiments (
        user_id UUID PRIMARY KEY REFERENCES auth.users,
        table_name TEXT,
        csv_dict JSONB
    );
    """
    with engine.connect() as conn:
        query = text(create_table_stmt)
        conn.execute(query)
        conn.commit()
        conn.close()


def enable_rls(table_name: str):
    table_name = table_name.lower()
    # Enable RLS
    with engine.connect() as conn:
        query = text(
            f"""
            ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;
            """
        )
        conn.execute(query)
        conn.commit()
        print(f"RLS for table {table_name} created in database")
        conn.close()


def create_policy(table_name: str):
    table_name = table_name.lower()
    # Enable RLS
    with engine.connect() as conn:
        query = text(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_policies
                    WHERE schemaname = 'public'
                    AND tablename = '{table_name}'
                    AND policyname = 'User CRUD own tables only'
                ) THEN
                    CREATE POLICY "User CRUD own tables only"
                    ON public.{table_name}
                    FOR ALL
                    TO authenticated
                    USING (
                        auth.uid() = user_id
                    )
                    WITH CHECK (
                        auth.uid() = user_id
                    );
                END IF;
            END
            $$;
            """
        )
        conn.execute(query)
        conn.commit()
        print(f"Created policy for table {table_name}.")
        conn.close()


def insert_data(table_name, file, user_id):
    table_name = table_name.lower()
    # Convert the uploaded CSV file into a DataFrame
    df = pd.read_csv(file)
    # Convert the DataFrame into a JSON string
    json_str = df.to_json(orient="records")
    # Connect to the database
    with engine.connect() as conn:
        # Prepare the INSERT INTO statement
        query = f"INSERT INTO experiments (user_id, table_name, csv_dict) VALUES (:user_id, :table_name, :csv_dict)"
        # Insert data into the table
        conn.execute(
            text(query),
            {"user_id": user_id, "table_name": table_name, "csv_dict": json_str},
        )
        conn.commit()
    st.write(f'Data "{table_name}" inserted into Experiments table')


def retrieve_bucket_files(bucket_name):
    # Retrieve bucket
    files = supabase_client.storage.from_(bucket_name).list()
    print(files)
    return files


def create_query(table_name, pair_param=None):
    if pair_param:
        query = f"SELECT {pair_param[0]}, {pair_param[1]} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    return query


def execute_query(query):
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


# TODO: check if we really need a df with just pair_param cols, it appears that y in PDP prediction is not correct
def query_table(table_name, pair_param=None):
    query = create_query(table_name, pair_param)
    df = execute_query(query)
    return df


def highlight_max(df):
    return df.style.highlight_max(subset=[df.columns[-1]])


def plot_output(df, col2):
    output = df.iloc[:, -1]
    for param in df.columns[:-1]:
        chart_data = pd.DataFrame({"x": df[param], "y": output})
        chart = (
            alt.Chart(chart_data)
            .mark_circle()
            .encode(x=alt.X("x", title=param), y=alt.Y("y", title=output.name))
        )
        col2.altair_chart(chart)


def plot_pairplot(df):
    sns.set_theme(context="talk")
    pairplot_fig = sns.pairplot(df, diag_kind="kde")
    plt.suptitle("Pairplot of the DataFrame", size=20, y=1.02)

    st.pyplot(pairplot_fig)


def plot_pdp(df):
    """
    Plot a partial dependence graph for the specified features.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    """
    # Define the model
    model = RandomForestRegressor()

    # Separate the features and the target
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1]
    y = df.iloc[:, -1]

    # Fit the model
    model.fit(X, y)

    # Compute the partial dependencies
    pdp_results = partial_dependence(model, X, features=X.columns)

    # Plot the partial dependence
    display = PartialDependenceDisplay.from_estimator(
        model, X, features=X.columns, kind="both"
    )
    # Remove the legend
    for ax in display.axes_.ravel():
        ax.get_legend().remove()

    display.figure_.suptitle("1-way numerical PDP using random forest", fontsize=16)
    st.pyplot(plt)


# add a 2-way interaction pdp plot, this function will only be called once user chooses which two parameters they want from a streamlit multi-box
def plot_interaction_pdp(
    df: pd.DataFrame, features_list: list[Tuple[str, str]], overlay: bool = None
):
    """
    Plot a 2-way interaction PDP for each pair of features in the list.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    features_list (list): The list of pairs of features to plot.
    overlay (bool): Whether to overlay the actual feature pair points.
    """
    # Define the model
    model = RandomForestRegressor()

    # Separate the features and the target
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1]
    y = df.iloc[:, -1]

    # Fit the model
    model.fit(X, y)

    for features in features_list:
        # Unpack the features tuple
        feature1, feature2 = features

        # Check if the selected features exist in the DataFrame
        if not {feature1, feature2}.issubset(df.columns):
            st.error(f"The selected features {features} must exist in the DataFrame.")
            continue

        # Check if the selected features are numeric
        if not np.issubdtype(df[feature1].dtype, np.number) or not np.issubdtype(
            df[feature2].dtype, np.number
        ):
            st.warning(f"The selected features {features} must be numeric.")
            continue

        # Compute the interaction PDP
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            [features],
            kind="average",
        )

        # Overlay the actual feature pair points if overlay is True
        if overlay:
            plt.scatter(df[feature1], df[feature2], c="r", s=30, edgecolor="k")

        # Plot the interaction PDP
        display.figure_.suptitle(
            f"2-way PDP for {features} using random forest", fontsize=16
        )
        plt.show()

        # Display the plot in Streamlit
        st.pyplot(plt)


def show_dashboard(table_name):
    df = query_table(table_name)
    df_styled = highlight_max(df)
    col1, col2 = st.columns(2)
    col1.dataframe(df_styled)
    plot_output(df, col2)
    plot_pairplot(df)
    plot_pdp(df)


def show_interaction_pdp(df, pair_param: list[Tuple[str, str]], overlay: bool = None):
    plot_interaction_pdp(df, pair_param, overlay)


def get_user_inputs(table):
    # user input batch number through number_input, default to 1
    batch_number = st.number_input("Enter batch number", min_value=1, value=1, step=1)

    # Get optimization type
    optimization_type = st.selectbox("Select optimization type", ["single", "multi"])

    direction = st.selectbox("Select optimization direction", ["minimize", "maximize"])

    # Get output column names
    if optimization_type == "single":
        output_column_names = [table.columns[-1]]
    elif optimization_type == "multi":
        output_column_names = table.columns[-2:]

    # Get number of parameters
    num_parameters = len(table.columns) - len(output_column_names)

    # Get number of random lines
    num_random_lines = st.number_input(
        "Enter the number of random lines",
        min_value=1,
        max_value=len(table),
        value=len(table),
    )

    # Get parameter info
    parameter_info = table.dtypes[: -len(output_column_names)].to_dict()
    # Define a mapping from pandas dtypes to your desired types
    dtype_mapping = {"int64": "integer", "float64": "float", "O": "object"}

    # Iterate over the items in the dictionary and replace the dtypes
    parameter_info = {
        k: dtype_mapping.get(v.name, v.name) for k, v in parameter_info.items()
    }

    # Get parameter ranges
    parameter_ranges = {}
    for column in table.columns[: -len(output_column_names)]:
        if np.issubdtype(table[column].dtype, np.number):
            min_value = st.number_input(
                f"Enter the min value for {column}", value=table[column].min()
            )
            max_value = st.number_input(
                f"Enter the max value for {column}", value=table[column].max()
            )
            parameter_ranges[column] = (min_value, max_value)
        elif np.issubdtype(table[column].dtype, object):
            categories = st.text_input(
                f"Enter the categories for {column}",
                value=", ".join(table[column].unique()),
            )
            parameter_ranges[column] = categories.split(", ")

    return (
        batch_number,
        optimization_type,
        output_column_names,
        num_parameters,
        num_random_lines,
        parameter_info,
        parameter_ranges,
        direction,
    )


def validate_inputs(table, parameter_ranges):
    validation_errors = []

    # Validate numeric parameters
    for column, range_values in parameter_ranges.items():
        if np.issubdtype(table[column].dtype, np.number):
            min_value, max_value = range_values
            if not min_value <= table[column].max() <= max_value:
                validation_errors.append(
                    f"Values for {column} are not within the specified range."
                )

    # Validate string parameters
    for column, categories in parameter_ranges.items():
        if np.issubdtype(table[column].dtype, object):
            unique_values = table[column].unique()
            if not set(unique_values).issubset(set(categories)):
                validation_errors.append(
                    f"Unique values for {column} are not within the specified categories."
                )

    return validation_errors


def display_dictionary(
    batch_number,
    table_name,
    optimization_type,
    output_column_names,
    num_parameters,
    num_random_lines,
    parameter_info,
    parameter_ranges,
    direction,
    bucket_name: str = "test-bucket",
) -> Dict[str, Any]:
    metadata = {
        "batch_number": batch_number,
        "table_name": table_name,
        "optimization_type": optimization_type,
        "batch_type": "batch",
        "output_column_names": output_column_names,
        "num_parameters": num_parameters,
        "num_random_lines": num_random_lines,
        "parameter_info": parameter_info,
        "parameter_ranges": parameter_ranges,
        "learner": "regr.ranger",
        "acquisition_function": "ei",
        "direction": direction,
        "bucket_name": bucket_name,
    }
    st.write(metadata)
    return metadata


def py_dict_to_r_list(py_dict):
    r_list = ro.ListVector({})
    for k, v in py_dict.items():
        if isinstance(v, dict):
            r_list.rx2[k] = py_dict_to_r_list(v)
        elif isinstance(v, list):
            r_list.rx2[k] = ro.StrVector([str(i) for i in v])
        else:
            r_list.rx2[k] = ro.StrVector([str(v)])
    return r_list


def py_dict_to_r_named_vector(py_dict):
    r_vector = ro.StrVector([])
    names = []
    for k, v in py_dict.items():
        if isinstance(v, dict):
            r_vector = ro.r.c(r_vector, py_dict_to_r_named_vector(v))
            names.extend([f"{k}.{sub_k}" for sub_k in v.keys()])
        elif isinstance(v, list):
            r_vector = ro.r.c(r_vector, ro.StrVector([str(i) for i in v]))
            names.extend([k] * len(v))
        else:
            r_vector = ro.r.c(r_vector, ro.StrVector([str(v)]))
            names.append(k)
    r_vector.names = ro.StrVector(names)
    return r_vector


def replace_value_with_nan(df: pd.DataFrame, value=-2147483648) -> pd.DataFrame:
    """
    Replace a specific value with np.nan in a DataFrame if the value exists.

    Parameters:
    df (pd.DataFrame): The DataFrame in which to replace the value.
    value (int): The value to replace. Defaults to -2147483648.

    Returns:
    pd.DataFrame: The DataFrame with the value replaced.
    """
    if (df.values == value).any():
        df.replace(value, np.nan, inplace=True)
    return df


def get_features(table_name):
    """
    Get a list of features i.e., all columns except the last column.

    Parameters:
    table_name (str): The name of the table.

    Returns:
    list: The list of features.
    """
    df = pd.read_sql_table(table_name, engine)

    # Return all columns except the last one
    features = df.columns.tolist()[:-1]
    print(features)
    return features


def get_dataframe(table_name):
    """
    Get a DataFrame from a table name.

    Parameters:
    table_name (str): The name of the table.

    Returns:
    pd.DataFrame: The DataFrame.
    """
    df = pd.read_sql_table(table_name, engine)
    return df
