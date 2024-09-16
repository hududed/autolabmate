import streamlit as st
import pandas as pd
import numpy as np
from numpy.random import RandomState
import plotly.graph_objects as go
from typing import Tuple

from sklearn.ensemble import RandomForestRegressor


SEED = 42
rng = RandomState(SEED)


def feature_importance(df: pd.DataFrame, model: RandomForestRegressor) -> None:
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
) -> None:
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


def train_model(df: pd.DataFrame, rng: int = rng) -> RandomForestRegressor:
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


def train_model_multi(df: pd.DataFrame, rng: int = rng) -> Tuple[RandomForestRegressor]:
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
