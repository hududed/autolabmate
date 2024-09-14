import csv
import pandas as pd
import streamlit as st
from typing import List, Union


def write_csv(
    data_header: List[str], param_values: List[List[Union[int, float, str]]]
) -> None:
    with open("data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_header)
        for values in param_values:
            writer.writerow(values)
    df2 = pd.read_csv("data.csv")
    df2.to_csv("data.csv", index=False)


def download_csv() -> None:
    st.download_button(
        label="Download CSV",
        data=open("data.csv", "rb").read(),
        file_name="data.csv",
        mime="text/csv",
    )
