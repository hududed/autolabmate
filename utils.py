import streamlit as st
import supabase
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect, MetaData, Table
import os
import re
import pandas as pd
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import altair as alt
from typing import List, Dict, Any, Tuple, Callable
from pathlib import Path
import json
import simplejson
from datetime import datetime
import rpy2.robjects as ro
from tenacity import retry, stop_after_attempt, wait_fixed
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import itertools
import zipfile

from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor

from abc import ABC, abstractmethod

from storage3.utils import StorageException

from components.authenticate import supabase_client, engine, refresh_jwt


SEED = 42
rng = RandomState(SEED)


def compress_files(files):
    # Create an in-memory buffer to store the zip file
    buffer = BytesIO()

    # Write the zip file to the buffer
    with zipfile.ZipFile(buffer, "w") as zip:
        for file in files:
            zip.writestr(file["name"], file["content"])

    # Seek to the beginning of the buffer
    buffer.seek(0)

    return buffer


# Define routes
def create_account(email, password):
    # Create new account
    response = supabase_client.auth.sign_up({"email": email, "password": password})
    return response


def login(credentials):
    # Login to existing account
    response = supabase_client.auth.sign_in_with_password(credentials)
    return response


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def upload_to_bucket(
    bucket_name, user_id, table_name, file_name, file_content, batch_number=1
):
    new_file_name = f"{user_id}/{table_name}/{batch_number}/{file_name}"
    print(new_file_name)

    try:
        # Upload file to bucket
        supabase_client.storage.from_(bucket_name).upload(new_file_name, file_content)
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


def save_to_local(
    bucket_name: str,
    user_id: str,
    table_name: str,
    file_name: str,
    df: pd.DataFrame,
    batch_number: int = 1,
):
    table_name = table_name.lower()
    new_file_name = f"{bucket_name}/{user_id}/{table_name}/{batch_number}/{file_name}"
    print(new_file_name)

    try:
        # Create necessary directories
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)

        # Save DataFrame to CSV file
        df.to_csv(new_file_name, index=False)

        print(f'"{new_file_name}" saved locally at "{os.path.abspath(new_file_name)}"')
    except Exception as e:
        raise e


def save_metadata(
    metadata: Dict[str, Any],
    user_id: str,
    table_name: str,
    batch_number: int = 1,
    bucket_name: str = "test-bucket",
):
    """
    Saves metadata to an in-memory file.

    Args:
        metadata (dict): The metadata to save.
        user_id (str): The user ID.
        table_name (str): The name of the table.
        batch_number (int): The batch number. Defaults to 1.
        bucket_name (str): The name of the bucket. Defaults to 'test-bucket'.
    """
    # Convert the metadata dictionary to a JSON string
    json_metadata = json.dumps(metadata)

    # Save the JSON string to an in-memory file
    with open(
        f"{bucket_name}/{user_id}/{table_name}/{batch_number}/metadata.json", "w"
    ) as f:
        f.write(json_metadata)


def load_metadata(
    user_id: str,
    table_name: str,
    batch_number: int = 1,
    bucket_name: str = "test-bucket",
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
    with open(
        f"{bucket_name}/{user_id}/{table_name}/{batch_number}/metadata.json", "r"
    ) as f:
        json_metadata = f.read()

    # Convert the JSON string to a dictionary
    metadata = json.loads(json_metadata)

    return metadata


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def upload_local_to_bucket(
    bucket_name: str,
    user_id: str,
    table_name: str,
    batch_number: int = 1,
    file_extension: str = ".rds",
):
    # Extract file name from file path
    base_path = Path(f"{bucket_name}/{user_id}/{table_name}/{batch_number}")
    files = [file for file in base_path.glob(f"*{file_extension}")]

    for file in files:
        file_name = file.name
        new_file_name = f"{user_id}/{table_name}/{batch_number}/{file_name}"

        # Read file content
        with open(file, "rb") as f:
            file_content = f.read()

        try:
            # Upload file to bucket
            supabase_client.storage.from_(bucket_name).upload(
                new_file_name, file_content
            )
            print(new_file_name)
            st.write(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
        except StorageException as e:
            if "jwt expired" in str(e):
                # Refresh the JWT and retry the upload
                new_jwt = refresh_jwt()
                if new_jwt:
                    supabase_client.storage.from_(bucket_name).upload(
                        new_file_name, file_content
                    )
                    print(new_file_name)
                    st.write(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
                else:
                    raise e
            elif "Duplicate" in str(e):
                print(
                    f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
                )
                st.write(
                    f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
                )
            else:
                raise e


def upload_metadata_to_bucket(metadata, batch_number=1):
    # Convert the metadata dictionary to a JSON string and encode it to bytes
    metadata_content = json.dumps(metadata).encode()

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Define the file name
    metadata_file_name = f"metadata-{timestamp}.json"

    # Upload the metadata to the bucket
    upload_to_bucket(
        metadata["bucket_name"],
        metadata["user_id"],
        metadata["table_name"],
        metadata_file_name,
        metadata_content,
        batch_number,
    )


def get_table_names(user_id):
    # Prepare the SELECT statement
    query = text("SELECT DISTINCT table_name FROM experiments WHERE user_id = :user_id")

    # Connect to the database
    with engine.connect() as conn:
        # Execute the SELECT statement
        result = conn.execute(query, {"user_id": user_id})
        conn.commit()
        conn.close()

    # Load the result into a DataFrame
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df["table_name"].tolist()


def get_latest_data_and_metadata(
    user_id: str, batch_number: int = None
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    # Prepare the SELECT statement
    query = text(
        "SELECT csv_dict, columns_order, metadata, table_name FROM experiments WHERE user_id = :user_id ORDER BY timestamp DESC"
    )

    # Connect to the database
    with engine.connect() as conn:
        # Execute the SELECT statement
        result = conn.execute(query, {"user_id": user_id})
        # Get the column names
        keys = result.keys()
        # Convert the result to a list of dictionaries
        rows = [dict(zip(keys, row)) for row in result]
        conn.commit()
        conn.close()

    # Filter the rows based on the batch_number
    if batch_number is not None:
        rows = [
            row
            for row in rows
            if (
                (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
                or {}
            ).get("batch_number")
            == batch_number - 1
        ]

    if rows:
        csv_dict, columns_order, metadata, table_name = (
            rows[0]["csv_dict"],
            rows[0]["columns_order"],
            rows[0]["metadata"],
            rows[0]["table_name"],
        )
    else:
        csv_dict, columns_order, metadata, table_name = (None, None, None, None)

    # Load the result into a DataFrame
    df = pd.DataFrame(csv_dict)
    # Reorder the columns according to the stored order
    df = df[columns_order]
    return df, metadata, table_name


def get_latest_data_for_table(
    user_id: str, table_name: str, batch_number: int = None
) -> pd.DataFrame:
    # Prepare the SELECT statement
    query = text(
        "SELECT csv_dict, columns_order, metadata FROM experiments WHERE user_id = :user_id AND table_name = :table_name ORDER BY timestamp DESC"
    )

    # Connect to the database
    with engine.connect() as conn:
        # Execute the SELECT statement
        result = conn.execute(query, {"user_id": user_id, "table_name": table_name})
        # Get the column names
        keys = result.keys()
        # Convert the result to a list of dictionaries
        rows = [dict(zip(keys, row)) for row in result]
        conn.commit()
        conn.close()

    # Filter the rows based on the batch_number
    if batch_number is not None:
        rows = [
            row
            for row in rows
            if (
                (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
                or {}
            ).get("batch_number")
            == batch_number - 1
        ]

    if rows:
        csv_dict, columns_order, metadata = (
            rows[0]["csv_dict"],
            rows[0]["columns_order"],
            rows[0]["metadata"],
        )
    else:
        csv_dict, columns_order, metadata = (None, None, None)
    # Load the result into a DataFrame
    df = pd.DataFrame(csv_dict)
    # Reorder the columns according to the stored order
    df = df[columns_order]
    return df, metadata


def create_experiments_table() -> None:
    # Create the profiles table if it doesn't exist
    create_table_stmt = """
    CREATE TABLE IF NOT EXISTS experiments (
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        id SERIAL PRIMARY KEY,
        user_id UUID REFERENCES auth.users,
        table_name TEXT,
        csv_dict JSONB,
        columns_order JSON
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


def drop_column_from_table(table_name, column_name):
    # TODO: Archive. not needed after RLS is enabled
    metadata = MetaData()
    db_table = Table(table_name, metadata, autoload_with=engine)

    # Check if the column exists
    if column_name in db_table.c:
        # Drop the column
        with engine.connect() as connection:
            db_table._columns.remove(db_table.c[column_name])
        st.write(f"Dropped column {column_name} from table {table_name}")
    else:
        st.write(f'Column "{column_name}" does not exist in table "{table_name}"')


def sanitize_column_names(table):
    # Sanitize column names
    table.columns = [re.sub("[^0-9a-zA-Z_]", "", col) for col in table.columns]
    table.columns = ["col_" + col if col[0].isdigit() else col for col in table.columns]
    return table


def insert_data(
    table_name: str, df: pd.DataFrame, user_id: str, metadata: Dict[str, Any] = None
) -> None:
    table_name = table_name.lower()
    df = df.where(pd.notnull(df), None)
    # Convert the DataFrame into a dictionary and then into a JSON string
    json_str = simplejson.dumps(df.to_dict(orient="records"), ignore_nan=True)
    # Store the order of the columns
    columns_order = json.dumps(list(df.columns))
    # Convert the metadata into a JSON string
    metadata_str = json.dumps(metadata)
    # Connect to the database
    with engine.connect() as conn:
        # Prepare the INSERT INTO statement
        query = "INSERT INTO experiments (user_id, table_name, csv_dict, columns_order, metadata) VALUES (:user_id, :table_name, :csv_dict, :columns_order, :metadata)"
        # Insert data into the table
        conn.execute(
            text(query),
            {
                "user_id": user_id,
                "table_name": table_name,
                "csv_dict": json_str,
                "columns_order": columns_order,
                "metadata": metadata_str,
            },
        )
        conn.commit()
    st.write(f'Data "{table_name}" inserted into Experiments table at {datetime.now()}')


def retrieve_bucket_files(bucket_name):
    # Retrieve bucket
    files = supabase_client.storage.from_(bucket_name).list()
    print(files)
    return files


def retrieve_and_download_files(
    bucket_name, user_id, table_name, batch_number=1, local_dir="./"
):
    """
    Retrieve all files from the specified bucket and download them locally.

    Args:
        bucket_name (str): The name of the bucket.
        user_id (str): The user ID.
        table_name (str): The table name.
        batch_number (int): The batch number.
        local_dir (str): The local directory to save the downloaded files.

    Returns:
        list: A list of local file paths.
    """
    files = supabase_client.storage.from_(bucket_name).list(
        f"{user_id}/{table_name}/{batch_number}"
    )
    if not files:
        raise Exception("No files to download")

    # print("Files listed in bucket: ", files)
    downloaded_files = []
    for file in files:
        file_name = file["name"]
        response = supabase_client.storage.from_(bucket_name).download(
            f"{user_id}/{table_name}/{batch_number}/{file_name}"
        )

        # Store the file name and content in a dictionary
        downloaded_files.append({"name": file_name, "content": response})

    return downloaded_files


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


def highlight_max(df: pd.DataFrame, direction: str):
    """
    Highlight the maximum or minimum value in the last column of a DataFrame based on the specified direction.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    direction (str): Either "min" or "max".

    Returns:
    df: The DataFrame with the maximum or minimum value in the last column highlighted.
    """
    if direction == "max":
        return df.style.highlight_max(subset=[df.columns[-1]])
    else:
        return df.style.highlight_min(subset=[df.columns[-1]])


def highlight_max_multi(df: pd.DataFrame, directions: Dict[str, str]):
    """
    Highlight the maximum or minimum value in each column of a DataFrame based on the specified directions.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    directions (dict): A dictionary mapping column names to either "minimize" or "maximize".

    Returns:
    df: The DataFrame with the maximum or minimum values in each column highlighted.
    """

    def highlight(s):
        is_max = directions[s.name] == "maximize"
        return (
            ["background-color: yellow" if v == s.max() else "" for v in s]
            if is_max
            else ["background-color: yellow" if v == s.min() else "" for v in s]
        )

    # Apply the highlight function only to the last two columns of the DataFrame
    df_styler = df.style.apply(highlight, subset=df.columns[-2:])

    return df_styler


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


def plot_output_with_confidence(
    df: pd.DataFrame, output_columns: list[str], metadata: Dict[str, Any]
):
    """
    Plots raw output values as dots, mean values as crosses, and confidence intervals as filled areas
    for each output column in a DataFrame. Adds hover functionality to reveal input metadata values.

    Parameters:
    - df: DataFrame containing the data.
    - output_columns: List of output column names to plot.
    - metadata: Dictionary containing metadata for hover information.
    """
    X_columns = metadata[
        "X_columns"
    ]  # Assuming metadata contains a key 'X_columns' for hover data

    for output in output_columns:
        fig = go.Figure()

        # Variables to ensure legend is only shown once for each type
        show_legend_for_raw = True
        show_legend_for_mean = True
        show_legend_for_confidence = True

        # Define mean and standard error column names
        mean_col = f"{output}_mean"
        se_col = f"{output}_se"

        for i, row in df.iterrows():
            hover_text = "<br>".join([f"{col}: {row[col]}" for col in X_columns])
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[row[output]],
                    mode="markers+lines",
                    marker=dict(size=10),
                    text=hover_text,
                    hoverinfo="text",
                    name="Raw Output",
                    showlegend=show_legend_for_raw,
                )
            )
            show_legend_for_raw = (
                False  # Only show legend for the first raw output trace
            )

        if mean_col in df.columns and se_col in df.columns:
            df_not_null = df.dropna(subset=[mean_col, se_col])

            # Line plot for mean values as crosses
            fig.add_trace(
                go.Scatter(
                    x=df_not_null.index,
                    y=df_not_null[mean_col],
                    mode="markers",
                    marker_symbol="x",
                    marker_size=12,
                    name="Mean Predicted Output",
                    hoverinfo="skip",
                    showlegend=show_legend_for_mean,
                )
            )
            show_legend_for_mean = False  # Only show legend for the first mean trace

            # Fill between for confidence interval
            fig.add_trace(
                go.Scatter(
                    x=df_not_null.index,
                    y=df_not_null[mean_col] - df_not_null[se_col],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_not_null.index,
                    y=df_not_null[mean_col] + df_not_null[se_col],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    fillcolor="rgba(128, 128, 128, 0.2)",
                    name="Confidence Interval",
                    hoverinfo="skip",
                    showlegend=show_legend_for_confidence,
                )
            )
            show_legend_for_confidence = (
                False  # Only show legend for the first confidence interval trace
            )

        fig.update_layout(
            title=f"{output} as a function of iteration",
            xaxis_title="Iteration",
            yaxis_title=output,
        )

        st.plotly_chart(fig)


def report_output_with_confidence(df: pd.DataFrame, output: str):
    plt.figure(figsize=(10, 6))
    # Scatter plot for raw output values as dots
    plt.scatter(df.index, df[output], label=f"Raw {output}", marker="o")
    plt.plot(df.index, df[output], label=f"Raw {output} (line)", alpha=0.5)

    # Check for mean and se columns and plot if they exist
    mean_col = f"{output}_mean"
    se_col = f"{output}_se"
    if mean_col in df.columns and se_col in df.columns:
        df_not_null = df.dropna(subset=[mean_col, se_col])
        plt.plot(
            df_not_null.index,
            df_not_null[mean_col],
            "x",
            label=f"Mean {output}",
            color="red",
        )
        lower_bound = df_not_null[mean_col] - df_not_null[se_col]
        upper_bound = df_not_null[mean_col] + df_not_null[se_col]
        plt.fill_between(
            df_not_null.index,
            lower_bound,
            upper_bound,
            color="gray",
            alpha=0.2,
            label=f"Confidence Interval ({output})",
        )

    plt.title(f"{output} as a function of iteration")
    plt.xlabel("Iteration")
    plt.ylabel(output)
    plt.legend()
    return plt.gcf()


class DashboardReportBase(ABC):
    def __init__(
        self, df: pd.DataFrame, df_with_preds: pd.DataFrame, metadata: Dict[str, Any]
    ):
        self.df = df
        self.df_with_preds = df_with_preds
        self.metadata = metadata
        self.buf = BytesIO()

    def save_fig_to_pdf(self, fig, pdf: PdfPages):
        pdf.savefig(fig)
        plt.close(fig)

    def generate_pairplot(self, pdf: PdfPages, pairplot_func: Callable):
        fig = pairplot_func()
        self.save_fig_to_pdf(fig, pdf)
        st.info("Pairplot created")

    def generate_output_vs_iteration_plot(
        self, output_name, pdf, output_plot_func: Callable
    ):
        fig = output_plot_func(output_name)
        self.save_fig_to_pdf(fig, pdf)
        st.info("Output vs Iteration plot created")

    def generate_single_pdp(
        self, model, output_name, pdf, pdp_func: Callable, n_outputs
    ):
        fig = pdp_func(model, output_name, n_outputs=n_outputs)
        self.save_fig_to_pdf(fig, pdf)
        st.info("Single PDP plot created")

    def generate_two_way_pdp(
        self,
        feature_pairs,
        model,
        output_name,
        pdf,
        two_way_pdp_func: Callable,
        n_outputs,
    ):
        for pair in feature_pairs:
            figures = two_way_pdp_func(
                [pair],
                model=model,
                output_name=output_name,
                n_outputs=n_outputs,
                overlay=True,
                for_report=True,
            )
            for fig in figures:
                self.save_fig_to_pdf(fig, pdf)
            st.info("2-way PDP plot created")

    @abstractmethod
    def create_report(self, **plot_funcs):
        pass


class DashboardReportMulti(DashboardReportBase):
    def __init__(
        self,
        df: pd.DataFrame,
        df_with_preds: pd.DataFrame,
        models: Tuple[RandomForestRegressor, RandomForestRegressor],
        output_columns: List[str],
        metadata: Dict[str, Any],
    ):
        super().__init__(df, df_with_preds, metadata)
        self.models = models
        self.output_columns = output_columns

    def create_report(self, **plot_funcs):
        with PdfPages(self.buf) as pdf:
            # Pairplot
            self.generate_pairplot(pdf, plot_funcs["pairplot_func"])

            # Generate all unique pairs of features
            feature_pairs = list(itertools.combinations(self.metadata["X_columns"], 2))

            # Output vs Iteration Plot and PDPs
            for model, output_name in zip(self.models, self.output_columns):
                self.generate_output_vs_iteration_plot(
                    output_name, pdf, plot_funcs["output_plot_func"]
                )
                self.generate_single_pdp(
                    model, output_name, pdf, plot_funcs["pdp_func"], n_outputs=2
                )
                self.generate_two_way_pdp(
                    feature_pairs,
                    model,
                    output_name,
                    pdf,
                    plot_funcs["two_way_pdp_func"],
                    n_outputs=2,
                )

        self.buf.seek(0)
        return self.buf


class DashboardReportSingle(DashboardReportBase):
    def __init__(
        self,
        df: pd.DataFrame,
        df_with_preds: pd.DataFrame,
        model: RandomForestRegressor,
        output_columns: List[str],
        metadata: Dict[str, Any],
    ):
        super().__init__(df, df_with_preds, metadata)
        self.model = model
        self.output_columns = output_columns

    def create_report(self, **plot_funcs):
        with PdfPages(self.buf) as pdf:
            # Pairplot
            self.generate_pairplot(pdf, plot_funcs["pairplot_func"])

            # Generate all unique pairs of features
            feature_pairs = list(itertools.combinations(self.metadata["X_columns"], 2))

            # Output vs Iteration Plot and PDPs
            output_name = self.output_columns[0]
            self.generate_output_vs_iteration_plot(
                output_name, pdf, plot_funcs["output_plot_func"]
            )
            self.generate_single_pdp(
                self.model, output_name, pdf, plot_funcs["pdp_func"], n_outputs=1
            )
            self.generate_two_way_pdp(
                feature_pairs,
                self.model,
                output_name,
                pdf,
                plot_funcs["two_way_pdp_func"],
                n_outputs=1,
            )

        self.buf.seek(0)
        return self.buf


def plot_pairplot(df):
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    variables = df_numeric.columns

    # Create a subplot with as many rows and columns as there are variables in the DataFrame
    fig = make_subplots(rows=len(variables), cols=len(variables))

    # Calculate x-axis limits based on KDE plots
    x_limits = []
    x_values, y_values = [], []
    for variable in variables:
        x = np.linspace(
            -df_numeric[variable].max(),  # Start from a negative value
            df_numeric[variable].max() * 5,
            20000,
        )
        x_values.append(x)
        y = gaussian_kde(df_numeric[variable])(x)
        y_values.append(y)
        indices = np.where(y > 1e-6)[0]  # Get the indices where y > 1e-12
        if len(indices) > 0:
            lower_limit = x[indices[0]]  # Get the first x value where y > 1e-12
            upper_limit = x[indices[-1]]  # Get the last x value where y > 1e-12
        else:
            lower_limit = df_numeric[variable].min()
            upper_limit = df_numeric[variable].max()
        x_limits.append((lower_limit, upper_limit))

    # Add scatter plots for each pair of variables and KDE plots on the diagonal
    for i in range(len(variables)):
        for j in range(len(variables)):
            lower_limit, upper_limit = x_limits[j]

            if i == j:
                # Add a line plot for the KDE
                fig.add_trace(
                    go.Scatter(
                        x=x_values[j], y=y_values[j], mode="lines", showlegend=False
                    ),
                    row=i + 1,
                    col=j + 1,
                )
            else:
                # Add a scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=df_numeric[variables[j]],
                        y=df_numeric[variables[i]],
                        mode="markers",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=j + 1,
                )
                # Compute the x-axis range for scatter plots
                lower_limit, upper_limit = x_limits[j]

            # Set the x-axis title only for the bottom row
            title_text = variables[j] if i == len(variables) - 1 else ""
            # Update the x-axis range
            fig.update_xaxes(
                title_text=title_text,
                row=i + 1,
                col=j + 1,
                range=[lower_limit, upper_limit],
            )

    # Set y-axis titles
    for i, variable in enumerate(variables, start=1):
        fig.update_yaxes(title_text=variable, row=i, col=1)

    fig.update_layout(title="Pairplot of the DataFrame", title_x=0.4)

    st.plotly_chart(fig)


def report_pairplot(df: pd.DataFrame) -> Figure:
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    variables = df_numeric.columns
    n_vars = len(variables)

    # Initialize the figure
    fig, axs = plt.subplots(n_vars, n_vars, figsize=(n_vars * 2.5, n_vars * 2.5))

    # Loop through rows and columns for plotting
    for i, var_i in enumerate(variables):
        for j, var_j in enumerate(variables):
            ax = axs[i, j]

            if i == j:  # Diagonal: Plot KDE
                sns.kdeplot(df_numeric[var_i], ax=ax, fill=True)
                ax.set_ylabel("")  # Remove y-axis label for KDE plots
            else:  # Off-diagonal: Plot scatter
                ax.scatter(df_numeric[var_j], df_numeric[var_i], alpha=0.5)

            # Set labels
            if i == n_vars - 1:  # Bottom row
                ax.set_xlabel(var_j)
            else:  # Hide x-axis labels for other rows
                ax.set_xlabel("")
                ax.set_xticks([])

            if j == 0:  # First column
                ax.set_ylabel(var_i)
            else:  # Hide y-axis labels for other columns
                ax.set_ylabel("")
                ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_pdp(
    df: pd.DataFrame, model: RandomForestRegressor, output_name: str, n_outputs: int = 1
):
    """
    This function plots a partial dependence graph for each numerical feature in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    model: The trained model for which the partial dependence is calculated.
    """
    # Separate the features and the target
    X = df.select_dtypes(include=[np.number]).iloc[:, :-n_outputs]

    # Create a subplot for each feature
    fig = make_subplots(rows=1, cols=len(X.columns))

    # Add a scatter plot for each feature
    for i, feature in enumerate(X.columns):
        # Compute the partial dependencies
        pdp_results = partial_dependence(model, X, features=[i], kind="both")
        pdp_values = pdp_results["average"][0]
        ice_values = pdp_results["individual"][0]

        # Add PDP line
        fig.add_trace(
            go.Scatter(
                x=pdp_results["values"][0],
                y=pdp_values,
                mode="lines",
                name="Average",
                line=dict(color="orange", width=4, dash="dash"),
                legendgroup="Average",
                hoverinfo="skip",
                showlegend=i == 0,
            ),
            row=1,
            col=i + 1,
        )

        # Add ICE lines
        for j, ice_line in enumerate(ice_values):
            fig.add_trace(
                go.Scatter(
                    x=pdp_results["values"][0],
                    y=ice_line,
                    mode="lines",
                    name="ICE"
                    if i == 0 and j == 0
                    else None,  # to prevent duplicate legend entries
                    line=dict(color="gray", width=0.5),
                    legendgroup="ICE",
                    hoverinfo="skip",  # to prevent cluttering hover information
                    showlegend=i == 0 and j == 0,
                ),
                row=1,
                col=i + 1,
            )

        # Set x-axis title
        fig.update_xaxes(title_text=feature, row=1, col=i + 1)

    # Update the layout
    fig.update_layout(
        height=200 * len(X.columns),
        title_text=f"1-way numerical PDP and ICE for {output_name} using Random Forest",
        legend_title_text="Line Type",
    )

    # Display the plot
    st.plotly_chart(fig)


def report_pdp(
    df: pd.DataFrame, model: RandomForestRegressor, output_name: str, n_outputs: int = 1
):
    # Select numeric features, excluding target columns
    X = df.select_dtypes(include=[np.number]).iloc[:, :-n_outputs]
    features = X.columns

    # Initialize figure
    fig, axs = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axs = [axs]  # Ensure axs is iterable for a single subplot

    # Plot PDP and ICE for each feature
    for i, feature in enumerate(features):
        ax = axs[i]
        pdp_results = partial_dependence(model, X, features=[feature], kind="both")
        pdp_values = pdp_results["average"][0]
        ice_values = pdp_results["individual"][0]
        feature_values = pdp_results["values"][0]

        # PDP line
        ax.plot(
            feature_values,
            pdp_values,
            label="PDP",
            color="orange",
            linewidth=2,
            linestyle="--",
        )

        # ICE lines
        for ice_line in ice_values:
            ax.plot(feature_values, ice_line, color="grey", linewidth=0.5, alpha=0.3)

        ax.set_title(feature)
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Partial Dependence")

    # Adjust layout and add legend
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"PDP and ICE for {output_name}", fontsize=16, y=1.05)
    return fig


def plot_interaction_pdp(
    df: pd.DataFrame,
    features_list: list[Tuple[str, str]],
    model: RandomForestRegressor,
    output_name: str = "",  # TODO: dynamic for both single and multi
    n_outputs: int = 1,
    overlay: bool = None,
    for_report: bool = False,
):
    """
    Plot a 2-way interaction PDP for each pair of features in the list.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    features_list (list): The list of pairs of features to plot.
    overlay (bool): Whether to overlay the actual feature pair points.
    """
    # Separate the features and the target
    X = df.select_dtypes(include=[np.number]).iloc[:, :-n_outputs]

    figs = []

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

        fig, ax = plt.subplots(figsize=(8, 6))
        # Compute the interaction PDP
        display = PartialDependenceDisplay.from_estimator(
            model, X, [features], kind="average", random_state=rng, ax=ax
        )

        # Overlay the actual feature pair points if overlay is True
        if overlay:
            ax.scatter(df[feature1], df[feature2], c="r", s=30, edgecolor="k", zorder=5)

        ax.set_title(
            f"2-way PDP between {features} for {output_name} using RandomForest",
            fontsize=16,
        )

        figs.append(fig)

        # Display the plot in Streamlit
        if not for_report:
            st.pyplot(fig)
    if for_report:
        return figs


def feature_importance(df: pd.DataFrame, model: RandomForestRegressor):
    """
    Display feature importances.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    model: The trained model.
    """
    # Separate the features and the target
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1]

    # Get feature importances
    importances = model.feature_importances_

    # Create a Plotly figure
    fig = go.Figure()

    # Add a bar for each feature
    for i, importance in enumerate(importances):
        fig.add_trace(go.Bar(x=[importance], y=[X.columns[i]], orientation="h"))

    # Update the layout with a title
    fig.update_layout(title_text="Feature importances")
    # Display the plot in Streamlit
    st.plotly_chart(fig)


def feature_importance_multi(
    df: pd.DataFrame,
    models: Tuple[RandomForestRegressor, RandomForestRegressor],
    output_names: list[str],
):
    """
    Display feature importances for multiple models.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    models: The trained models.
    output_names (List[str]): The names of the outputs.
    """
    # Separate the features and the target
    X = df.select_dtypes(include=[np.number]).iloc[:, :-2]

    # For each model
    for model, output_name in zip(models, output_names):
        # Get feature importances
        importances = model.feature_importances_

        # Create a Plotly figure
        fig = go.Figure()

        # Add a bar for each feature
        for i, importance in enumerate(importances):
            fig.add_trace(go.Bar(x=[importance], y=[X.columns[i]], orientation="h"))

        # Update the layout with a title
        fig.update_layout(title_text=f"Feature importances for {output_name}")

        # Display the plot in Streamlit
        st.plotly_chart(fig)


def show_dashboard(
    df: pd.DataFrame,
    model: RandomForestRegressor,
    y_directions: Dict[str, str],
    y_columns: list[str],
):
    y_direction = next(iter(y_directions.values()))
    y_column = y_columns[0]
    df_styled = highlight_max(df, y_direction)
    st.dataframe(df_styled)
    plot_pairplot(df)
    plot_pdp(df, model, y_column)


def show_dashboard_multi(
    df: pd.DataFrame,
    models: Tuple[RandomForestRegressor, RandomForestRegressor],
    y_directions: Dict[str, str],
    y_columns: list[str],
):
    """
    Display a dashboard for multiple models.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    models: The trained models.
    y_columns (list): The names of the output columns.
    y_directions (dict): A dictionary mapping column names to either "min" or "max".
    """
    # Highlight the max or min value in each output column
    df_styled = highlight_max_multi(df, y_directions)

    st.dataframe(df_styled)

    # Plot a pairplot for the output columns
    plot_pairplot(df)

    # For each model
    for model, output_name in zip(models, y_columns):
        # Plot a Partial Dependence Plot
        plot_pdp(df, model, output_name, len(y_directions))


def show_interaction_pdp(
    df: pd.DataFrame,
    pair_param: list[Tuple[str, str]],
    model: RandomForestRegressor,
    overlay: bool = None,
):
    plot_interaction_pdp(df, pair_param, model, overlay)


def show_interaction_pdp_multi(
    df: pd.DataFrame,
    pair_param: list[Tuple[str, str]],
    models: Tuple[RandomForestRegressor, RandomForestRegressor],
    output_names: list[str],
    overlay: bool = None,
):
    n_outputs = len(output_names)
    for model, output_name in zip(models, output_names):
        plot_interaction_pdp(df, pair_param, model, output_name, n_outputs, overlay)


def train_model(df: pd.DataFrame, rng: int = rng):
    """
    Train a RandomForestRegressor model.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    rng (int): The random seed.

    Returns:
    model: The trained model.
    """
    df = df.dropna()
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1]
    y = df.iloc[:, -1]
    model = RandomForestRegressor(random_state=rng)
    model.fit(X, y)
    return model


def train_model_multi(df: pd.DataFrame, rng: int = rng):
    """
    Train two RandomForestRegressor models.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    rng (int): The random seed.

    Returns:
    models: The trained models.
    """
    df = df.dropna()
    X = df.select_dtypes(include=[np.number]).iloc[:, :-2]
    y1 = df.iloc[:, -2]
    y2 = df.iloc[:, -1]

    model1 = RandomForestRegressor(random_state=rng)
    model1.fit(X, y1)

    model2 = RandomForestRegressor(random_state=rng)
    model2.fit(X, y2)

    return model1, model2


def get_user_inputs(df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple:
    # user input batch number through number_input, default to 1
    batch_number = st.number_input("Enter batch number", min_value=1, value=1, step=1)

    # print(metadata)

    # Get optimization type with default from metadata
    optimization_type = metadata["optimization_type"]

    # Get output column names and directions from metadata
    output_column_names = metadata["output_column_names"]
    directions = metadata["directions"]

    # print(df)

    # Get number of parameters
    num_parameters = len(df.columns) - len(output_column_names)

    # Get number of random lines
    num_random_lines = st.number_input(
        "Enter the number of random lines",
        min_value=1,
        max_value=len(df),
        value=len(df),
    )

    parameter_info = df.dtypes[: -len(output_column_names)].to_dict()
    # Define a mapping from pandas dtypes to your desired types
    dtype_mapping = {"int64": "integer", "float64": "float", "O": "object"}

    # Iterate over the items in the dictionary and replace the dtypes
    parameter_info = {
        k: dtype_mapping.get(v.name, v.name) for k, v in parameter_info.items()
    }

    # Get parameter ranges
    parameter_ranges = {}
    to_nearest_values = {}

    for column in df.columns[: -len(output_column_names)]:
        if np.issubdtype(df[column].dtype, np.number):
            min_value = st.number_input(
                f"Enter the min value for {column}", value=df[column].min()
            )
            max_value = st.number_input(
                f"Enter the max value for {column}", value=df[column].max()
            )
            parameter_ranges[column] = (min_value, max_value)

            # Get to_nearest_value for numeric parameters
            if np.issubdtype(df[column].dtype, np.integer):
                to_nearest_value = st.number_input(
                    f"Enter the value to round {column} to",
                    min_value=1,
                    value=1,
                    step=1,
                )
                to_nearest_values[column] = to_nearest_value
            elif np.issubdtype(df[column].dtype, np.floating):
                to_nearest_value = st.number_input(
                    f"Enter the value to round {column} to",
                    min_value=0.01,
                    value=0.01,
                    step=0.01,
                )
                to_nearest_values[column] = to_nearest_value

        elif np.issubdtype(df[column].dtype, object):
            categories = st.text_input(
                f"Enter the categories for {column}",
                value=", ".join(df[column].unique()),
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
        directions,
        to_nearest_values,
    )


def validate_inputs(
    df: pd.DataFrame,
    parameter_ranges: Dict[str, Any],
    output_column_names: list[str] = None,
):
    validation_errors = []

    if output_column_names is not None:
        df = df[list(parameter_ranges.keys()) + output_column_names]

    # Validate numeric parameters
    for column, range_values in parameter_ranges.items():
        if np.issubdtype(df[column].dtype, np.number):
            min_value, max_value = range_values
            if not min_value <= df[column].max() <= max_value:
                validation_errors.append(
                    f"Values for {column} are not within the specified range."
                )

    # Validate string parameters
    for column, categories in parameter_ranges.items():
        if np.issubdtype(df[column].dtype, object):
            unique_values = df[column].unique()
            if not set(unique_values).issubset(set(categories)):
                validation_errors.append(
                    f"Unique values for {column} are not within the specified categories."
                )

    return validation_errors


def display_dictionary(
    seed: int,
    batch_number: int,
    table_name: str,
    optimization_type: str,
    output_column_names: list[str],
    num_parameters: int,
    num_random_lines: int,
    parameter_info: Dict[str, Any],
    parameter_ranges: tuple,
    directions: Dict[str, Any],
    user_id: str,
    to_nearest: float,
    metadata: Dict[str, Any],
    bucket_name: str = "test-bucket",
) -> Dict[str, Any]:
    metadata.update(
        {
            "seed": seed,
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
            "directions": directions,
            "bucket_name": bucket_name,
            "user_id": user_id,
            "to_nearest": to_nearest,
        }
    )
    display_metadata = {
        k: v for k, v in metadata.items() if k != "user_id" and k != "bucket_name"
    }
    with st.expander("Show metadata", expanded=False):
        st.write(display_metadata)
    return metadata


def py_dict_to_r_list(py_dict: Dict[str, Any]):
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


def get_features(df: pd.DataFrame) -> list[str]:
    """
    Get a list of features i.e., all columns except the last column.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    list: The list of features.
    """
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
