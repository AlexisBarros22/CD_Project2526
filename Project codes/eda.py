import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class EDA:
    """
    Exploratory Data Analysis class for the cleaned dataset.

    This class is meant to be used BEFORE train/test split and BEFORE
    categorical encoding/scaling for modeling.
    """

    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        self.data = data.copy()
        self.verbose = verbose

        # Global plot style
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "#f8f9fa"
        plt.rcParams["axes.edgecolor"] = "#333333"

    def summary(self):
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

    def plot_target_distribution(self, bins: int = 80, clip_range=(-60, 180)):
        """
        Plot clipped ARR_DELAY distribution for better readability.
        """
        if 'ARR_DELAY' not in self.data.columns:
            print("ARR_DELAY column not found.")
            return self

        arr_delay_clipped = self.data['ARR_DELAY'].clip(*clip_range)

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
        plt.show()

        return self

    def plot_numeric_distributions(self, columns=None, bins: int = 40):
        """
        Plot histograms for selected numeric columns in a grid layout.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=['number']).columns.tolist()

        if not columns:
            print("No numeric columns found.")
            return self

        n = len(columns)
        ncols = 2
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)

        palette = sns.color_palette("Set2", n_colors=n)

        for i, col in enumerate(columns):
            plot_data = self.data[col]

            if col == 'ARR_DELAY':
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
        plt.show()

        return self

    def plot_boxplots(self, columns=None, clip_dict=None):
        """
        Plot all boxplots in one figure with clipping for better readability.
        """
        if columns is None:
            columns = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'ARR_DELAY', 'SEASON', 'FL_DAY_OF_WEEK', 'FL_MONTH', 'AVG_SPEED',
                       'PEAK_MORNING', 'PEAK_EVENING']

        if clip_dict is None:
            clip_dict = {
                'CRS_DEP_TIME': (0, 1440),
                'CRS_ARR_TIME': (0, 1440),
                'CRS_ELAPSED_TIME': (0, 400),
                'DISTANCE': (0, 3000),
                'ARR_DELAY': (-60, 300),
                'SEASON': (1, 4),
                'FL_DAY_OF_WEEK': (1, 7),
                'FL_MONTH': (1, 12),
                'AVG_SPEED': (1, 6),
                'PEAK_MORNING': (0, 1),
                'PEAK_EVENING': (0, 1)
            }

        existing_cols = [col for col in columns if col in self.data.columns]

        if not existing_cols:
            print("No valid columns found for boxplots.")
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
        plt.show()

        return self

    def plot_correlation_heatmap(self, figsize=(10, 8)):
        """
        Plot correlation heatmap using numeric columns only.
        """
        numeric_data = self.data.select_dtypes(include=['number'])

        if numeric_data.empty:
            print("No numeric columns available for correlation heatmap.")
            return self

        corr_matrix = numeric_data.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            square=True
        )
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        return self

    def plot_delay_by_day_of_week(self):
        """
        Plot median ARR_DELAY by day of week instead of raw boxplot.
        """
        required_cols = {'FL_DAY_OF_WEEK', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("FL_DAY_OF_WEEK and/or ARR_DELAY not found.")
            return self

        delay_by_day = (
            self.data.groupby("FL_DAY_OF_WEEK")["ARR_DELAY"]
            .median()
            .reset_index()
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
        plt.title("Median Arrival Delay by Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Median Delay (minutes)")
        plt.tight_layout()
        plt.show()

        return self

    def plot_delay_rate_by_day_of_week(self):
        """
        Plot percentage of delayed flights by day of week.
        """
        required_cols = {'FL_DAY_OF_WEEK', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("FL_DAY_OF_WEEK and/or ARR_DELAY not found.")
            return self

        df_plot = self.data.copy()
        df_plot["DELAYED"] = (df_plot["ARR_DELAY"] > 0).astype(int)

        delay_rate = (
            df_plot.groupby("FL_DAY_OF_WEEK")["DELAYED"]
            .mean()
            .reset_index()
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
        plt.ylabel("Delay Rate")
        plt.tight_layout()
        plt.show()

        return self

    def plot_delay_by_month(self):
        """
        Plot median ARR_DELAY by month.
        """
        required_cols = {'FL_MONTH', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("FL_MONTH and/or ARR_DELAY not found.")
            return self

        delay_by_month = (
            self.data.groupby("FL_MONTH")["ARR_DELAY"]
            .median()
            .reset_index()
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
        plt.title("Median Arrival Delay by Month")
        plt.xlabel("Month")
        plt.ylabel("Median Delay (minutes)")
        plt.tight_layout()
        plt.show()

        return self

    def plot_delay_vs_distance(self):
        """
        Hexbin plot of DISTANCE vs ARR_DELAY for cleaner density visualization.
        """
        required_cols = {'DISTANCE', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("DISTANCE and/or ARR_DELAY not found.")
            return self

        plt.figure(figsize=(10, 6))
        plt.hexbin(
            self.data['DISTANCE'],
            self.data['ARR_DELAY'].clip(-60, 180),
            gridsize=50,
            cmap='viridis',
            mincnt=1
        )
        plt.colorbar(label='Number of Flights')
        plt.title("Arrival Delay vs Distance")
        plt.xlabel("Distance (miles)")
        plt.ylabel("Arrival Delay (minutes, clipped)")
        plt.tight_layout()
        plt.show()

        return self

    def plot_delay_vs_elapsed_time(self):
        """
        Hexbin plot of CRS_ELAPSED_TIME vs ARR_DELAY.
        """
        required_cols = {'CRS_ELAPSED_TIME', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("CRS_ELAPSED_TIME and/or ARR_DELAY not found.")
            return self

        plt.figure(figsize=(10, 6))
        plt.hexbin(
            self.data['CRS_ELAPSED_TIME'],
            self.data['ARR_DELAY'].clip(-60, 180),
            gridsize=45,
            cmap='plasma',
            mincnt=1
        )
        plt.colorbar(label='Number of Flights')
        plt.title("Arrival Delay vs Scheduled Elapsed Time")
        plt.xlabel("Scheduled Elapsed Time (minutes)")
        plt.ylabel("Arrival Delay (minutes, clipped)")
        plt.tight_layout()
        plt.show()

        return self

    def plot_top_origin_cities_by_average_delay(self, top_n: int = 15, min_flights: int = 2000):
        """
        Barplot of top origin cities by average arrival delay.
        Filters tiny sample sizes for more reliable results.
        """
        required_cols = {'ORIGIN_CITY', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("ORIGIN_CITY and/or ARR_DELAY not found.")
            return self

        city_stats = (
            self.data.groupby('ORIGIN_CITY')
            .agg(avg_delay=('ARR_DELAY', 'mean'), flights=('ARR_DELAY', 'size'))
            .query('flights >= @min_flights')
            .sort_values('avg_delay', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=city_stats,
            x='avg_delay',
            y='ORIGIN_CITY',
            hue='avg_delay',
            palette='magma',
            legend=False
        )
        plt.title(f"Top {top_n} Origin Cities by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Origin City")
        plt.tight_layout()
        plt.show()

        return self

    def plot_top_dest_cities_by_average_delay(self, top_n: int = 15, min_flights: int = 2000):
        """
        Barplot of top destination cities by average arrival delay.
        """
        required_cols = {'DEST_CITY', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("DEST_CITY and/or ARR_DELAY not found.")
            return self

        city_stats = (
            self.data.groupby('DEST_CITY')
            .agg(avg_delay=('ARR_DELAY', 'mean'), flights=('ARR_DELAY', 'size'))
            .query('flights >= @min_flights')
            .sort_values('avg_delay', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=city_stats,
            x='avg_delay',
            y='DEST_CITY',
            hue='avg_delay',
            palette='cubehelix',
            legend=False
        )
        plt.title(f"Top {top_n} Destination Cities by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Destination City")
        plt.tight_layout()
        plt.show()

        return self

    def plot_top_airlines_by_average_delay(self, top_n: int = 15, min_flights: int = 5000):
        """
        Barplot of DOT_CODE values by average arrival delay.
        """
        required_cols = {'DOT_CODE', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("DOT_CODE and/or ARR_DELAY not found.")
            return self

        airline_stats = (
            self.data.groupby('DOT_CODE')
            .agg(avg_delay=('ARR_DELAY', 'mean'), flights=('ARR_DELAY', 'size'))
            .query('flights >= @min_flights')
            .sort_values('avg_delay', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=airline_stats,
            x='avg_delay',
            y=airline_stats['DOT_CODE'].astype(str),
            hue='avg_delay',
            palette='flare',
            legend=False
        )
        plt.title(f"Top {top_n} Airlines (DOT_CODE) by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("DOT_CODE")
        plt.tight_layout()
        plt.show()

        return self

    def plot_top_routes_by_average_delay(self, top_n: int = 15, min_flights: int = 2000):
        """
        Barplot of top routes by average arrival delay.
        """
        required_cols = {'ROUTE', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("ROUTE and/or ARR_DELAY not found.")
            return self

        route_stats = (
            self.data.groupby('ROUTE')
            .agg(avg_delay=('ARR_DELAY', 'mean'), flights=('ARR_DELAY', 'size'))
            .query('flights >= @min_flights')
            .sort_values('avg_delay', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=route_stats,
            x='avg_delay',
            y='ROUTE',
            hue='avg_delay',
            palette='viridis',
            legend=False
        )
        plt.title(f"Top {top_n} Routes by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Route")
        plt.tight_layout()
        plt.show()

        return self

    def plot_top_origin_sta_by_average_delay(self, top_n: int = 15, min_flights: int = 2000):
        """
        Barplot of top origin states by average arrival delay.
        """
        required_cols = {'ORIGIN_STATE', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("ORIGIN_STATE and/or ARR_DELAY not found.")
            return self

        state_stats = (
            self.data.groupby('ORIGIN_STATE')
            .agg(avg_delay=('ARR_DELAY', 'mean'), flights=('ARR_DELAY', 'size'))
            .query('flights >= @min_flights')
            .sort_values('avg_delay', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=state_stats,
            x='avg_delay',
            y='ORIGIN_STATE',
            hue='avg_delay',
            palette='magma',
            legend=False
        )
        plt.title(f"Top {top_n} Origin States by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Origin State")
        plt.tight_layout()
        plt.show()

        return self


    def plot_top_dest_state_by_average_delay(self, top_n: int = 15, min_flights: int = 2000):
        """
        Barplot of top destination states by average arrival delay.
        """
        required_cols = {'DEST_STATE', 'ARR_DELAY'}
        if not required_cols.issubset(self.data.columns):
            print("DEST_STATE and/or ARR_DELAY not found.")
            return self

        state_stats = (
            self.data.groupby('DEST_STATE')
            .agg(avg_delay=('ARR_DELAY', 'mean'), flights=('ARR_DELAY', 'size'))
            .query('flights >= @min_flights')
            .sort_values('avg_delay', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=state_stats,
            x='avg_delay',
            y='DEST_STATE',
            hue='avg_delay',
            palette='cubehelix',
            legend=False
        )
        plt.title(f"Top {top_n} Destination States by Average Delay")
        plt.xlabel("Average Arrival Delay (minutes)")
        plt.ylabel("Destination State")
        plt.tight_layout()
        plt.show()

        return self

    def plot_all_core(self):
        """
        Run a sensible core set of EDA steps for the cleaned flight dataset.
        """
        self.summary()
        self.plot_target_distribution()
        self.plot_numeric_distributions(columns=[
            'CRS_DEP_TIME',
            'CRS_ARR_TIME',
            'CRS_ELAPSED_TIME',
            'DISTANCE',
            'SEASON',
            'FL_DAY_OF_WEEK',
            'FL_MONTH',
            'AVG_SPEED',
            'PEAK_MORNING',
            'PEAK_EVENING'
        ])
        self.plot_boxplots(columns=[
            'CRS_DEP_TIME',
            'CRS_ARR_TIME',
            'CRS_ELAPSED_TIME',
            'DISTANCE',
            'SEASON',
            'FL_DAY_OF_WEEK',
            'FL_MONTH',
            'ARR_DELAY'
            'AVG_SPEED',
            'PEAK_MORNING',
            'PEAK_EVENING'
        ])
        self.plot_correlation_heatmap()
        self.plot_delay_by_day_of_week()
        self.plot_delay_rate_by_day_of_week()
        self.plot_delay_by_month()
        self.plot_delay_vs_distance()
        self.plot_delay_vs_elapsed_time()
        self.plot_top_origin_cities_by_average_delay()
        self.plot_top_dest_cities_by_average_delay()
        self.plot_top_airlines_by_average_delay()
        self.plot_top_routes_by_average_delay()
        self.plot_top_origin_sta_by_average_delay()
        self.plot_top_dest_state_by_average_delay()

        return self