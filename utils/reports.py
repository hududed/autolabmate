import itertools
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from numpy.random import RandomState
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

SEED = 42
rng = RandomState(SEED)


def report_output_with_confidence(df: pd.DataFrame, output: str) -> Figure:
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


def report_pdp(
    df: pd.DataFrame, model: RandomForestRegressor, output_name: str, n_outputs: int = 1
) -> Figure:
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
        grid_values = pdp_results["grid_values"][0]

        # PDP line
        ax.plot(
            grid_values,
            pdp_values,
            label="PDP",
            color="orange",
            linewidth=2,
            linestyle="--",
        )

        # ICE lines
        for ice_line in ice_values:
            ax.plot(grid_values, ice_line, color="grey", linewidth=0.5, alpha=0.3)

        ax.set_title(feature)
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Partial Dependence")

    # Adjust layout and add legend
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"PDP and ICE for {output_name}", fontsize=16, y=1.05)
    return fig


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
    def create_report(self, **plot_funcs) -> BytesIO:
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

    def create_report(self, **plot_funcs) -> BytesIO:
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

    def create_report(self, **plot_funcs) -> BytesIO:
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
