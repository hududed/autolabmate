import json
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from numpy.random import RandomState
from storage3.utils import StorageException
from tenacity import retry, stop_after_attempt, wait_fixed

from db.database import supabase_client
from dependencies.authentication import refresh_jwt
from utils.io import generate_timestamps

SEED = 42
rng = RandomState(SEED)


def compress_files(files: List[Dict[str, Any]]) -> BytesIO:
    # Create an in-memory buffer to store the zip file
    buffer = BytesIO()

    # Write the zip file to the buffer
    with zipfile.ZipFile(buffer, "w") as zip:
        for file in files:
            zip.writestr(file["name"], file["content"])

    # Seek to the beginning of the buffer
    buffer.seek(0)

    return buffer


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
    with st.spinner("Preparing files for download..."):
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
                print(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
            except StorageException as e:
                if "jwt expired" in str(e):
                    # Refresh the JWT and retry the upload
                    new_jwt = refresh_jwt()
                    if new_jwt:
                        supabase_client.storage.from_(bucket_name).upload(
                            new_file_name, file_content
                        )
                        print(new_file_name)
                        print(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
                    else:
                        raise e
                elif "Duplicate" in str(e):
                    print(
                        f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
                    )
                else:
                    raise e
    st.success("Files ready for download!")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def upload_to_bucket(
    bucket_name, user_id, table_name, file_name, file_content, batch_number=1
):
    new_file_name = f"{user_id}/{table_name}/{batch_number}/{file_name}"
    print(new_file_name)

    try:
        # Upload file to bucket
        supabase_client.storage.from_(bucket_name).upload(new_file_name, file_content)
        print(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')
    except Exception as e:
        if "Duplicate" in str(e):
            print(
                f'File "{new_file_name}" already exists in bucket "{bucket_name}", skipping upload'
            )
        else:
            raise e


def upload_metadata_to_bucket(metadata: Dict[str, Any], batch_number: int = 1):
    # Convert the metadata dictionary to a JSON string and encode it to bytes
    metadata_content = json.dumps(metadata).encode()

    # Get the current timestamp
    filename_timestamp, _ = generate_timestamps()

    # Define the file name
    metadata_file_name = f"metadata-{filename_timestamp}.json"

    # Upload the metadata to the bucket
    upload_to_bucket(
        metadata["bucket_name"],
        metadata["user_id"],
        metadata["table_name"],
        metadata_file_name,
        metadata_content,
        batch_number,
    )
