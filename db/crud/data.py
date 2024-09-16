from sqlalchemy import text
from db.database import engine
import pandas as pd
from typing import Dict, Any, Tuple
import json
import simplejson
import streamlit as st
from datetime import datetime


def get_latest_data_metadata_table_by_user_id(
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


def get_latest_data_metadata_by_user_id_table(
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
