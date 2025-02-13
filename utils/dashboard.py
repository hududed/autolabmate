from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure
from numpy.random import RandomState
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

SEED = 42
rng = RandomState(SEED)


def highlight_max(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Highlight the maximum or minimum value in the last column of a DataFrame based on the specified direction.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    direction (str): Either "min" or "max".

    Returns:
    df: The DataFrame with the maximum or minimum value in the last column highlighted.
    """
    if direction == "maximize":
        return df.style.highlight_max(subset=[df.columns[-1]])
    else:
        return df.style.highlight_min(subset=[df.columns[-1]])


def highlight_max_multi(df: pd.DataFrame, directions: Dict[str, str]) -> pd.DataFrame:
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


def plot_output_with_confidence(
    df: pd.DataFrame, output_columns: list[str], metadata: Dict[str, Any]
) -> None:
    """
    Plots raw output values, mean predictions as crosses, and confidence intervals as filled areas.
    Works for both single- and multi-objective cases.

    Parameters:
    - df: DataFrame containing the data.
    - output_columns: List of output column names.
    - metadata: Dictionary containing metadata; must include 'X_columns' for hover info.
    """
    X_columns = metadata["X_columns"]

    if len(output_columns) == 1:
        # Single objective: plot one figure
        output = output_columns[0]
        fig = go.Figure()
        show_legend_for_raw = True
        show_legend_for_mean = True
        show_legend_for_confidence = True
        mean_col = f"{output}_mean"
        se_col = f"{output}_se"

        # Plot raw output values
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
            show_legend_for_raw = False

        # Plot mean predictions and confidence intervals if available
        if mean_col in df.columns and se_col in df.columns:
            df_not_null = df.dropna(subset=[mean_col, se_col])
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
            show_legend_for_mean = False
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
            show_legend_for_confidence = False

        fig.update_layout(
            title=f"{output} as a function of iteration",
            xaxis_title="Iteration",
            yaxis_title=output,
        )
        st.plotly_chart(fig)
    else:
        # Multi-objective: loop over each output column
        for output in output_columns:
            fig = go.Figure()
            show_legend_for_raw = True
            show_legend_for_mean = True
            show_legend_for_confidence = True
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
                show_legend_for_raw = False

            if mean_col in df.columns and se_col in df.columns:
                df_not_null = df.dropna(subset=[mean_col, se_col])
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
                show_legend_for_mean = False
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
                show_legend_for_confidence = False

            fig.update_layout(
                title=f"{output} as a function of iteration",
                xaxis_title="Iteration",
                yaxis_title=output,
            )
            st.plotly_chart(fig)


def plot_pairplot(df: pd.DataFrame) -> None:
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


def plot_pdp(
    df: pd.DataFrame, model: RandomForestRegressor, output_name: str, n_outputs: int = 1
) -> None:
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
        # Use 'grid_values' instead of 'values'
        grid_values = pdp_results["grid_values"][0]

        # Add PDP line
        fig.add_trace(
            go.Scatter(
                x=grid_values,
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
                    x=grid_values,
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


def plot_interaction_pdp(
    df: pd.DataFrame,
    features_list: list[Tuple[str, str]],
    model: RandomForestRegressor,
    output_name: str = "",  # TODO: dynamic for both single and multi
    n_outputs: int = 1,
    overlay: bool = None,
    for_report: bool = False,
) -> None | List[Figure]:
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
        _ = PartialDependenceDisplay.from_estimator(
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


def show_dashboard(
    df: pd.DataFrame,
    model: RandomForestRegressor,
    y_directions: Dict[str, str],
    y_columns: list[str],
) -> None:
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
) -> None:
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
) -> None:
    plot_interaction_pdp(df, pair_param, model, overlay)


def show_interaction_pdp_multi(
    df: pd.DataFrame,
    pair_param: list[Tuple[str, str]],
    models: Tuple[RandomForestRegressor, RandomForestRegressor],
    output_names: list[str],
    overlay: bool = None,
) -> None:
    n_outputs = len(output_names)
    for model, output_name in zip(models, output_names):
        plot_interaction_pdp(df, pair_param, model, output_name, n_outputs, overlay)
