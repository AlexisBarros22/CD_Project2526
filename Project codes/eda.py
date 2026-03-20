import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class EDA:
    """
    Exploratory Data Analysis class for a cleaned flight-delay dataset.

    This class provides summary information and a set of plots to inspect
    numeric variables, delay patterns, and engineered features.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    verbose : bool, default=True
        Whether to print summary output.
    """

    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        """
        Initialize the EDA object.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe.
        verbose : bool, default=True
            Whether to print extra output.
        """
        self.data = data.copy()
        self.verbose = verbose

        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "#f8f9fa"
        plt.rcParams["axes.edgecolor"] = "#333333"

    def _sanitize_filename(self, name: str) -> str:
        """
        Convert a plot name into a safe filename.
        """
        safe = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in name)
        return safe.strip().replace(" ", "_").lower()

    def _ensure_plot_folder(self, plot_type: str) -> Path:
        """
        Create and return a folder for a specific plot type inside Output files.
        """
        base_dir = Path("..") / "Output files"
        plot_dir = base_dir / plot_type
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    def _export_current_plot(self, plot_type: str, plot_name: str, dpi: int = 300):
        """
        Export the currently active matplotlib figure as a PNG file.
        """
        folder = self._ensure_plot_folder(plot_type)
        filename = f"{self._sanitize_filename(plot_name)}.png"
        filepath = folder / filename

        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")

        if self.verbose:
            print(f"Saved plot to: {filepath}")

    def summary(self):
        """
        Print a dataset summary.

        Returns
        -------
        EDA
            The current object.
        """
        if not self.verbose:
            return self

        print("Exploratory Data Analysis (EDA) Report")
        print("-" * 50)
        print(f"Shape: {self.data.shape}")
        print("\nColumns:")
        print(self.data.columns.tolist())
        print("\nData types:")
        print(self.data.dtypes)
        print("\nMissing values:")
        print(self.data.isnull().sum())
        print("\nSummary statistics:")
        print(self.data.describe(include="all"))
        return self

    def plot_target_distribution(self, bins: int = 80, clip_range=(-60, 180), export: bool = False):
        """
        Plot the distribution of arrival delay.

        Parameters
        ----------
        bins : int, default=80
            Number of bins.
        clip_range : tuple, default=(-60, 180)
            Range used to clip values for plotting.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        if "ARR_DELAY" not in self.data.columns:
            print("ARR_DELAY column not found.")
            return self

        arr_delay_clipped = self.data["ARR_DELAY"].clip(*clip_range)

        plt.figure(figsize=(10, 5))
        sns.histplot(
            arr_delay_clipped,
            bins=bins,
            kde=True,
            color="dodgerblue",
            edgecolor="white",
            alpha=0.85
        )
        plt.title("Arrival Delay Distribution (Clipped)")
        plt.xlabel("Arrival Delay (minutes)")
        plt.ylabel("Count")
        plt.tight_layout()

        if export:
            self._export_current_plot("distributions", "arrival_delay_distribution_clipped")

        plt.show()
        return self

    def plot_numeric_distributions(self, columns=None, bins: int = 40, export: bool = False):
        """
        Plot histograms for selected numeric columns.

        Parameters
        ----------
        columns : list or None, default=None
            Columns to plot. If None, a default set is used.
        bins : int, default=40
            Number of bins.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        if columns is None:
            columns = [
                "CRS_ELAPSED_TIME",
                "DISTANCE",
                "AVG_SPEED",
                "ARR_DELAY"
            ]

        existing_cols = [col for col in columns if col in self.data.columns]

        if not existing_cols:
            print("No valid numeric columns found.")
            return self

        n = len(existing_cols)
        ncols = 2
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)

        palette = sns.color_palette("Set2", n_colors=n)

        for i, col in enumerate(existing_cols):
            plot_data = self.data[col].copy()

            if col == "ARR_DELAY":
                plot_data = plot_data.clip(-60, 180)

            sns.histplot(
                plot_data,
                bins=bins,
                kde=True,
                ax=axes[i],
                color=palette[i],
                edgecolor="white",
                alpha=0.9
            )
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if export:
            self._export_current_plot("distributions", "numeric_distributions")

        plt.show()
        return self

    def plot_boxplots(self, columns=None, clip_dict=None, export: bool = False):
        """
        Plot boxplots for selected continuous columns.

        Parameters
        ----------
        columns : list or None, default=None
            Columns to plot.
        clip_dict : dict or None, default=None
            Optional clipping ranges for columns.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        if columns is None:
            columns = ["CRS_ELAPSED_TIME", "DISTANCE", "ARR_DELAY", "AVG_SPEED"]

        if clip_dict is None:
            clip_dict = {
                "CRS_ELAPSED_TIME": (0, 400),
                "DISTANCE": (0, 3000),
                "ARR_DELAY": (-60, 300),
                "AVG_SPEED": (1, 6)
            }

        existing_cols = [col for col in columns if col in self.data.columns]

        if not existing_cols:
            print("No valid continuous columns found for boxplots.")
            return self

        n = len(existing_cols)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n))

        if n == 1:
            axes = [axes]

        palette = sns.color_palette("Spectral", n_colors=n)

        for i, col in enumerate(existing_cols):
            plot_data = self.data[col].copy()

            if col in clip_dict:
                plot_data = plot_data.clip(*clip_dict[col])

            sns.boxplot(
                x=plot_data,
                ax=axes[i],
                color=palette[i],
                showfliers=False,
                width=0.5
            )
            axes[i].set_title(f"Boxplot of {col} (Clipped)")
            axes[i].set_xlabel(col)

        plt.tight_layout()

        if export:
            self._export_current_plot("boxplots", "continuous_boxplots")

        plt.show()
        return self

    def plot_cyclical_time_features(self, export: bool = False):
        """
        Plot encoded departure and arrival time features.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        feature_pairs = [
            ("CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "Scheduled Departure Time"),
            ("CRS_ARR_TIME_sin", "CRS_ARR_TIME_cos", "Scheduled Arrival Time")
        ]

        available_pairs = [
            (sin_col, cos_col, title)
            for sin_col, cos_col, title in feature_pairs
            if sin_col in self.data.columns and cos_col in self.data.columns
        ]

        if not available_pairs:
            print("No encoded time features found.")
            return self

        n = len(available_pairs)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

        if n == 1:
            axes = [axes]

        for ax, (sin_col, cos_col, title) in zip(axes, available_pairs):
            plot_df = self.data[[sin_col, cos_col]].dropna()

            ax.scatter(
                plot_df[cos_col],
                plot_df[sin_col],
                alpha=0.25,
                s=15
            )
            ax.set_title(title)
            ax.set_xlabel("Cosine")
            ax.set_ylabel("Sine")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal")

            circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", alpha=0.5)
            ax.add_patch(circle)

        plt.tight_layout()

        if export:
            self._export_current_plot("cyclical_features", "cyclical_time_features")

        plt.show()
        return self

    def plot_correlation_heatmap(self, figsize=(12, 8), export: bool = False):
        """
        Plot a correlation heatmap for selected numeric features.

        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        numeric_cols = [
            "DISTANCE",
            "CRS_ELAPSED_TIME",
            "ARR_DELAY",
            "AVG_SPEED",
            "PEAK_MORNING",
            "PEAK_EVENING",
            "CRS_DEP_TIME_sin",
            "CRS_DEP_TIME_cos",
            "CRS_ARR_TIME_sin",
            "CRS_ARR_TIME_cos"
        ]

        existing_cols = [col for col in numeric_cols if col in self.data.columns]

        if not existing_cols:
            print("No numeric columns available for correlation heatmap.")
            return self

        corr_matrix = self.data[existing_cols].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            square=True
        )
        plt.title("Correlation Heatmap")
        plt.tight_layout()

        if export:
            self._export_current_plot("correlations", "correlation_heatmap")

        plt.show()
        return self

    def plot_delay_by_day_of_week(self, export: bool = False):
        """
        Plot mean arrival delay by day of week.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"FL_DAY_OF_WEEK", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("FL_DAY_OF_WEEK and/or ARR_DELAY not found.")
            return self

        delay_by_day = (
            self.data.groupby("FL_DAY_OF_WEEK", observed=False)["ARR_DELAY"]
            .mean()
            .reset_index()
            .sort_values("FL_DAY_OF_WEEK")
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=delay_by_day,
            x="FL_DAY_OF_WEEK",
            y="ARR_DELAY",
            hue="FL_DAY_OF_WEEK",
            palette="viridis",
            legend=False
        )
        plt.title("Mean Arrival Delay by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Mean Delay (minutes)")
        plt.tight_layout()

        if export:
            self._export_current_plot("barplots", "mean_arrival_delay_by_day_of_week")

        plt.show()
        return self

    def plot_delay_rate_by_day_of_week(self, export: bool = False):
        """
        Plot the proportion of delayed flights by day of week.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"FL_DAY_OF_WEEK", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("FL_DAY_OF_WEEK and/or ARR_DELAY not found.")
            return self

        df_plot = self.data.copy()
        df_plot["DELAYED"] = (df_plot["ARR_DELAY"] > 0).astype(int)

        delay_rate = (
            df_plot.groupby("FL_DAY_OF_WEEK", observed=False)["DELAYED"]
            .mean()
            .mul(100)
            .reset_index()
            .sort_values("FL_DAY_OF_WEEK")
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=delay_rate,
            x="FL_DAY_OF_WEEK",
            y="DELAYED",
            hue="FL_DAY_OF_WEEK",
            palette="magma",
            legend=False
        )
        plt.title("Proportion of Delayed Flights by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Delay Rate (%)")
        plt.tight_layout()

        if export:
            self._export_current_plot("barplots", "delay_rate_by_day_of_week")

        plt.show()
        return self

    def plot_delay_by_month(self, export: bool = False):
        """
        Plot mean arrival delay by month.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"FL_MONTH", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("FL_MONTH and/or ARR_DELAY not found.")
            return self

        delay_by_month = (
            self.data.groupby("FL_MONTH", observed=False)["ARR_DELAY"]
            .mean()
            .reset_index()
            .sort_values("FL_MONTH")
        )

        plt.figure(figsize=(12, 5))
        sns.lineplot(
            data=delay_by_month,
            x="FL_MONTH",
            y="ARR_DELAY",
            marker="o",
            linewidth=2.5,
            color="teal"
        )
        plt.title("Mean Arrival Delay by Month")
        plt.xlabel("Month")
        plt.ylabel("Mean Delay (minutes)")
        plt.tight_layout()

        if export:
            self._export_current_plot("lineplots", "mean_arrival_delay_by_month")

        plt.show()
        return self

    def plot_delay_vs_distance(self, export: bool = False):
        """
        Plot arrival delay against distance.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"DISTANCE", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("DISTANCE and/or ARR_DELAY not found.")
            return self

        plot_data = self.data[["DISTANCE", "ARR_DELAY"]].copy()
        plot_data["ARR_DELAY"] = plot_data["ARR_DELAY"].clip(-60, 180)

        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=plot_data,
            x="DISTANCE",
            y="ARR_DELAY",
            scatter_kws={"alpha": 0.35, "s": 20},
            line_kws={"color": "red", "linewidth": 2}
        )
        plt.title("Arrival Delay vs Distance")
        plt.xlabel("Distance (miles)")
        plt.ylabel("Arrival Delay (minutes, clipped)")
        plt.tight_layout()

        if export:
            self._export_current_plot("scatterplots", "arrival_delay_vs_distance")

        plt.show()
        return self

    def plot_delay_vs_elapsed_time(self, export: bool = False):
        """
        Plot arrival delay against scheduled elapsed time.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"CRS_ELAPSED_TIME", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("CRS_ELAPSED_TIME and/or ARR_DELAY not found.")
            return self

        plot_data = self.data[["CRS_ELAPSED_TIME", "ARR_DELAY"]].copy()
        plot_data["ARR_DELAY"] = plot_data["ARR_DELAY"].clip(-60, 180)

        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=plot_data,
            x="CRS_ELAPSED_TIME",
            y="ARR_DELAY",
            scatter_kws={"alpha": 0.35, "s": 20},
            line_kws={"color": "red", "linewidth": 2}
        )
        plt.title("Arrival Delay vs Scheduled Elapsed Time")
        plt.xlabel("Scheduled Elapsed Time (minutes)")
        plt.ylabel("Arrival Delay (minutes, clipped)")
        plt.tight_layout()

        if export:
            self._export_current_plot("scatterplots", "arrival_delay_vs_elapsed_time")

        plt.show()
        return self

    def plot_delay_heatmap_month_day(self, export: bool = False):
        """
        Plot mean arrival delay by month and day of week.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"FL_MONTH", "FL_DAY_OF_WEEK", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("Required columns not found.")
            return self

        pivot = self.data.pivot_table(
            values="ARR_DELAY",
            index="FL_DAY_OF_WEEK",
            columns="FL_MONTH",
            aggfunc="mean"
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
        plt.title("Mean Arrival Delay by Month and Day of Week")
        plt.xlabel("Month")
        plt.ylabel("Day of Week")
        plt.tight_layout()

        if export:
            self._export_current_plot("heatmaps", "mean_arrival_delay_month_day")

        plt.show()
        return self

    def plot_departure_time_month_heatmap(self, bins: int = 24, export: bool = False):
        """
        Plot mean delay by departure time bin and month.

        Parameters
        ----------
        bins : int, default=24
            Number of bins.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "FL_MONTH", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("Required columns not found.")
            return self

        df_plot = self.data[
            ["CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "FL_MONTH", "ARR_DELAY"]
        ].dropna().copy()

        angles = np.arctan2(df_plot["CRS_DEP_TIME_sin"], df_plot["CRS_DEP_TIME_cos"])
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        df_plot["time_bin"] = pd.cut(
            angles,
            bins=np.linspace(0, 2 * np.pi, bins + 1),
            labels=[f"{i:02d}:00" for i in range(bins)],
            include_lowest=True
        )

        pivot = df_plot.pivot_table(
            values="ARR_DELAY",
            index="time_bin",
            columns="FL_MONTH",
            aggfunc="mean"
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, cmap="coolwarm")
        plt.title("Mean Delay by Departure Time and Month")
        plt.xlabel("Month")
        plt.ylabel("Departure Time Bin")
        plt.tight_layout()

        if export:
            self._export_current_plot("heatmaps", "mean_delay_departure_time_month")

        plt.show()
        return self

    def plot_delay_by_departure_time_circle(self, bins: int = 24, export: bool = False):
        """
        Plot mean arrival delay over departure time bins on a polar chart.

        Parameters
        ----------
        bins : int, default=24
            Number of bins.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("Required encoded departure time columns not found.")
            return self

        df_plot = self.data[
            ["CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos", "ARR_DELAY"]
        ].dropna().copy()

        angles = np.arctan2(df_plot["CRS_DEP_TIME_sin"], df_plot["CRS_DEP_TIME_cos"])
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        df_plot["time_bin"] = pd.cut(
            angles,
            bins=np.linspace(0, 2 * np.pi, bins + 1),
            labels=False,
            include_lowest=True
        )

        summary = (
            df_plot.groupby("time_bin", observed=False)["ARR_DELAY"]
            .mean()
            .reset_index()
        )

        theta = np.linspace(0, 2 * np.pi, bins, endpoint=False)

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(theta, summary["ARR_DELAY"], marker="o")
        ax.fill(theta, summary["ARR_DELAY"], alpha=0.25)
        ax.set_title("Mean Arrival Delay by Scheduled Departure Time")

        if export:
            self._export_current_plot("polar_plots", "mean_arrival_delay_by_departure_time")

        plt.show()
        return self

    def plot_route_delay_rate(self, top_n: int = 15, min_flights: int = 3000, export: bool = False):
        """
        Plot delay rate for busy routes.

        Parameters
        ----------
        top_n : int, default=15
            Number of routes to show.
        min_flights : int, default=3000
            Minimum number of flights required.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"ROUTE", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("Required columns not found.")
            return self

        df_plot = self.data.copy()
        df_plot["DELAYED"] = (df_plot["ARR_DELAY"] > 0).astype(int)

        stats = (
            df_plot.groupby("ROUTE", observed=False)
            .agg(
                delay_rate=("DELAYED", "mean"),
                flights=("DELAYED", "size")
            )
            .query("flights >= @min_flights")
            .sort_values("flights", ascending=False)
            .head(top_n)
            .reset_index()
        )

        stats["delay_rate"] *= 100

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=stats,
            x="delay_rate",
            y="ROUTE",
            hue="delay_rate",
            palette="viridis",
            legend=False
        )
        plt.title("Delay Rate (%) for Top Busy Routes")
        plt.xlabel("Delay Rate (%)")
        plt.ylabel("Route")
        plt.tight_layout()

        if export:
            self._export_current_plot("barplots", "route_delay_rate_busy_routes")

        plt.show()
        return self

    def plot_delay_by_season_violin(self, export: bool = False):
        """
        Plot arrival delay distribution by season.

        Parameters
        ----------
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"SEASON", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("Required columns not found.")
            return self

        df_plot = self.data[["SEASON", "ARR_DELAY"]].copy()
        df_plot["ARR_DELAY"] = df_plot["ARR_DELAY"].clip(-60, 180)

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_plot, x="SEASON", y="ARR_DELAY", inner="quartile")
        plt.title("Arrival Delay Distribution by Season")
        plt.xlabel("Season")
        plt.ylabel("Arrival Delay (minutes, clipped)")
        plt.tight_layout()

        if export:
            self._export_current_plot("violin_plots", "arrival_delay_by_season")

        plt.show()
        return self

    def plot_origin_city_volume_vs_delay(self, min_flights: int = 2000, export: bool = False):
        """
        Plot flight volume against average delay for origin cities.

        Parameters
        ----------
        min_flights : int, default=2000
            Minimum number of flights required.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"ORIGIN_CITY", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("Required columns not found.")
            return self

        stats = (
            self.data.groupby("ORIGIN_CITY", observed=False)
            .agg(
                avg_delay=("ARR_DELAY", "mean"),
                flights=("ARR_DELAY", "size")
            )
            .query("flights >= @min_flights")
            .reset_index()
        )

        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=stats, x="flights", y="avg_delay", alpha=0.7)
        plt.title("Origin City: Flight Volume vs Average Delay")
        plt.xlabel("Number of Flights")
        plt.ylabel("Average Arrival Delay")
        plt.tight_layout()

        if export:
            self._export_current_plot("scatterplots", "origin_city_volume_vs_average_delay")

        plt.show()
        return self

    def plot_top_airlines_by_average_delay(self, top_n: int = 15, min_flights: int = 5000, export: bool = False):
        """
        Plot airlines with the highest average delay.

        Parameters
        ----------
        top_n : int, default=15
            Number of airlines to show.
        min_flights : int, default=5000
            Minimum number of flights required.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"DOT_CODE", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("DOT_CODE and/or ARR_DELAY not found.")
            return self

        airline_stats = (
            self.data.groupby("DOT_CODE", observed=False)
            .agg(avg_delay=("ARR_DELAY", "mean"), flights=("ARR_DELAY", "size"))
            .query("flights >= @min_flights")
            .sort_values("avg_delay", ascending=False)
            .head(top_n)
            .reset_index()
        )

        airline_stats["DOT_CODE"] = airline_stats["DOT_CODE"].astype(str)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=airline_stats,
            x="avg_delay",
            y="DOT_CODE",
            hue="avg_delay",
            palette="flare",
            legend=False
        )
        plt.title(f"Top {top_n} Airlines by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("DOT_CODE")
        plt.tight_layout()

        if export:
            self._export_current_plot("barplots", "top_airlines_by_average_delay")

        plt.show()
        return self

    def plot_top_origin_cities_by_average_delay(self, top_n: int = 15, min_flights: int = 2000, export: bool = False):
        """
        Plot origin cities with the highest average delay.

        Parameters
        ----------
        top_n : int, default=15
            Number of cities to show.
        min_flights : int, default=2000
            Minimum number of flights required.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"ORIGIN_CITY", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("ORIGIN_CITY and/or ARR_DELAY not found.")
            return self

        city_stats = (
            self.data.groupby("ORIGIN_CITY", observed=False)
            .agg(avg_delay=("ARR_DELAY", "mean"), flights=("ARR_DELAY", "size"))
            .query("flights >= @min_flights")
            .sort_values("avg_delay", ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=city_stats,
            x="avg_delay",
            y="ORIGIN_CITY",
            hue="avg_delay",
            palette="magma",
            legend=False
        )
        plt.title(f"Top {top_n} Origin Cities by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Origin City")
        plt.tight_layout()

        if export:
            self._export_current_plot("barplots", "top_origin_cities_by_average_delay")

        plt.show()
        return self

    def plot_top_dest_cities_by_average_delay(self, top_n: int = 15, min_flights: int = 2000, export: bool = False):
        """
        Plot destination cities with the highest average delay.

        Parameters
        ----------
        top_n : int, default=15
            Number of cities to show.
        min_flights : int, default=2000
            Minimum number of flights required.
        export : bool, default=False
            Whether to export the plot as PNG.

        Returns
        -------
        EDA
            The current object.
        """
        required_cols = {"DEST_CITY", "ARR_DELAY"}
        if not required_cols.issubset(self.data.columns):
            print("DEST_CITY and/or ARR_DELAY not found.")
            return self

        city_stats = (
            self.data.groupby("DEST_CITY", observed=False)
            .agg(avg_delay=("ARR_DELAY", "mean"), flights=("ARR_DELAY", "size"))
            .query("flights >= @min_flights")
            .sort_values("avg_delay", ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=city_stats,
            x="avg_delay",
            y="DEST_CITY",
            hue="avg_delay",
            palette="cubehelix",
            legend=False
        )
        plt.title(f"Top {top_n} Destination Cities by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Destination City")
        plt.tight_layout()

        if export:
            self._export_current_plot("barplots", "top_destination_cities_by_average_delay")

        plt.show()
        return self

    def plot_all_core(self):
        """
        Run a core set of EDA plots for display only.

        Returns
        -------
        EDA
            The current object.
        """
        self.summary()
        self.plot_target_distribution()
        self.plot_numeric_distributions()
        self.plot_boxplots()
        self.plot_cyclical_time_features()
        self.plot_correlation_heatmap()
        self.plot_delay_by_day_of_week()
        self.plot_delay_rate_by_day_of_week()
        self.plot_delay_by_month()
        self.plot_delay_vs_distance()
        self.plot_delay_vs_elapsed_time()
        self.plot_delay_heatmap_month_day()
        self.plot_departure_time_month_heatmap()
        self.plot_delay_by_departure_time_circle()
        self.plot_route_delay_rate()
        self.plot_delay_by_season_violin()
        self.plot_origin_city_volume_vs_delay()
        self.plot_top_airlines_by_average_delay()
        self.plot_top_origin_cities_by_average_delay()
        self.plot_top_dest_cities_by_average_delay()

        return self

    def export_all_core(self):
        """
        Run and export the core EDA plots into PNG files inside Output files.

        Returns
        -------
        EDA
            The current object.
        """
        self.plot_target_distribution(export=True)
        self.plot_numeric_distributions(export=True)
        self.plot_boxplots(export=True)
        self.plot_cyclical_time_features(export=True)
        self.plot_correlation_heatmap(export=True)
        self.plot_delay_by_day_of_week(export=True)
        self.plot_delay_rate_by_day_of_week(export=True)
        self.plot_delay_by_month(export=True)
        self.plot_delay_vs_distance(export=True)
        self.plot_delay_vs_elapsed_time(export=True)
        self.plot_delay_heatmap_month_day(export=True)
        self.plot_departure_time_month_heatmap(export=True)
        self.plot_delay_by_departure_time_circle(export=True)
        self.plot_route_delay_rate(export=True)
        self.plot_delay_by_season_violin(export=True)
        self.plot_origin_city_volume_vs_delay(export=True)
        self.plot_top_airlines_by_average_delay(export=True)
        self.plot_top_origin_cities_by_average_delay(export=True)
        self.plot_top_dest_cities_by_average_delay(export=True)

        return self