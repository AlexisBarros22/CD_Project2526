import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap


class DimReduction:
    """
    Dimensionality reduction utility for visualizing flight data with PCA and UMAP.

    This class supports:
    - PCA for linear dimensionality reduction
    - UMAP for non-linear dimensionality reduction
    - plot coloring using:
        - delay-based categories
        - generic categorical labels
        - continuous numeric labels

    Notes
    -----
    - Assumes preprocessing such as scaling and missing-value handling
      has already been completed before the data is passed in.
    - Labels are used only for coloring plots, not for fitting PCA or UMAP.
    - PCA is best used with continuous / cyclical features.
    - UMAP can be used with a broader mixed feature set for visualization.
    - PCA is interpretable through explained variance and principal components,
      while UMAP is mainly for visualization and exploration. :contentReference[oaicite:2]{index=2}

    Parameters
    ----------
    data : pd.DataFrame or array-like
        Input feature matrix.
    labels : array-like
        Labels aligned row-by-row with the input data. Used only for visualization.
    verbose : bool, default=True
        Whether to print progress and summary messages.

    Attributes
    ----------
    data : pd.DataFrame
        Feature matrix stored as a DataFrame.
    labels : np.ndarray
        Labels used for plot coloring.
    verbose : bool
        Whether to print progress messages.
    pca_result : np.ndarray or None
        PCA-transformed coordinates.
    pca_model : sklearn.decomposition.PCA or None
        Fitted PCA model.
    pca_labels : np.ndarray or None
        Labels aligned with PCA results.
    umap_result : np.ndarray or None
        UMAP-transformed coordinates.
    umap_labels : np.ndarray or None
        Labels aligned with UMAP results.
    """

    def __init__(self, data, labels, verbose=True):
        """
        Initialize the dimensionality reduction helper.

        Parameters
        ----------
        data : pd.DataFrame or array-like
            Input feature matrix.
        labels : array-like
            Labels used only for plot coloring.
        verbose : bool, default=True
            Whether to print progress information.
        """
        self.data = pd.DataFrame(data).copy()
        self.labels = np.array(labels)
        self.verbose = verbose

        self.pca_result = None
        self.pca_model = None
        self.pca_labels = None

        self.umap_result = None
        self.umap_labels = None

        sns.set_theme(
            style="whitegrid",
            rc={
                "axes.facecolor": "#f8f9fa",
                "figure.facecolor": "white"
            }
        )

    def _get_feature_columns(self, feature_cols=None):
        """
        Return the valid feature columns available in the dataset.

        If no feature list is provided, a default subset is used.

        Parameters
        ----------
        feature_cols : list or None, default=None
            Requested columns. If None, a default subset is used.

        Returns
        -------
        list
            List of columns present in the dataset.
        """
        if feature_cols is None:
            feature_cols = ["AVG_SPEED", "DISTANCE", "CRS_ELAPSED_TIME"]
        return [col for col in feature_cols if col in self.data.columns]

    def _categorize_delay_labels(self, labels):
        """
        Convert numeric delay values into grouped delay categories.

        Delay bins
        ----------
        - On-time / Early : delay <= 0
        - Minor delay     : 0 < delay <= 15
        - Moderate delay  : 15 < delay <= 60
        - Long delay      : delay > 60

        Parameters
        ----------
        labels : array-like
            Numeric delay values.

        Returns
        -------
        np.ndarray
            Array of categorical delay labels.
        """
        labels = np.asarray(labels)
        return np.where(
            labels <= 0, "On-time / Early",
            np.where(
                labels <= 15, "Minor delay",
                np.where(labels <= 60, "Moderate delay", "Long delay")
            )
        )

    def _generic_categorical_labels(self, labels):
        """
        Convert labels to strings for generic categorical plotting.

        Useful for labels such as:
        - month
        - season
        - weekday
        - peak flags
        - encoded categories

        Parameters
        ----------
        labels : array-like
            Input labels.

        Returns
        -------
        np.ndarray
            Labels converted to strings.
        """
        return pd.Series(labels).astype(str).values

    def run_pca(self, feature_cols=None):
        """
        Fit PCA on the selected feature columns.

        PCA is a linear dimensionality reduction method that finds orthogonal
        components explaining the maximum variance in the data. It is most
        appropriate when features are already standardized and when the goal
        is dimensionality reduction, structure inspection, or visualization
        of global variance patterns. :contentReference[oaicite:3]{index=3}

        Parameters
        ----------
        feature_cols : list or None, default=None
            Columns to use for PCA. If None, a default set is used.

        Returns
        -------
        DimReduction
            The current object.
        """
        if self.verbose:
            print("\n--- Running PCA (Linear) ---")

        cols_present = self._get_feature_columns(feature_cols)

        if len(cols_present) < 2:
            print("Not enough feature columns found to run PCA.")
            return self

        numeric_data = self.data[cols_present]

        self.pca_model = PCA()
        self.pca_result = self.pca_model.fit_transform(numeric_data)
        self.pca_labels = self.labels.copy()

        if self.verbose:
            var_explained = self.pca_model.explained_variance_ratio_
            print(f"Features used: {cols_present}")
            print(f"PC1 explains {var_explained[0] * 100:.2f}% of the variance.")
            if len(var_explained) > 1:
                print(f"PC2 explains {var_explained[1] * 100:.2f}% of the variance.")

        return self

    def plot_pca(self, max_samples=100000, label_mode="delay_categorical"):
        """
        Plot the first two PCA components.

        Label modes
        -----------
        - 'delay_categorical'
            Bin numeric delay labels into delay groups.
        - 'generic_categorical'
            Use raw labels as categories.
        - 'continuous'
            Use raw numeric labels as a continuous color gradient.

        Parameters
        ----------
        max_samples : int, default=100000
            Maximum number of points to display. If the PCA result is larger,
            a random sample is used for readability.
        label_mode : str, default='delay_categorical'
            Coloring mode for the plot.

        Returns
        -------
        DimReduction
            The current object.
        """
        if self.pca_result is None:
            print("Please run_pca() first.")
            return self

        n_rows = self.pca_result.shape[0]

        if n_rows > max_samples:
            if self.verbose:
                print(f"Plotting a random sample of {max_samples:,} points for visibility...")
            np.random.seed(42)
            idx = np.random.choice(n_rows, max_samples, replace=False)
        else:
            idx = np.arange(n_rows)

        plot_df = pd.DataFrame({
            "PC1": self.pca_result[idx, 0],
            "PC2": self.pca_result[idx, 1]
        })

        plt.figure(figsize=(9, 7))

        if label_mode == "delay_categorical":
            plot_df["Label"] = self._categorize_delay_labels(self.pca_labels[idx])

            hue_order = ["On-time / Early", "Minor delay", "Moderate delay", "Long delay"]
            palette = {
                "On-time / Early": "#4575b4",
                "Minor delay": "#91bfdb",
                "Moderate delay": "#fdae61",
                "Long delay": "#d73027"
            }

            sns.scatterplot(
                data=plot_df,
                x="PC1",
                y="PC2",
                hue="Label",
                hue_order=hue_order,
                palette=palette,
                alpha=0.5,
                s=15,
                edgecolor=None
            )
            plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=4)

        elif label_mode == "generic_categorical":
            plot_df["Label"] = self._generic_categorical_labels(self.pca_labels[idx])

            sns.scatterplot(
                data=plot_df,
                x="PC1",
                y="PC2",
                hue="Label",
                alpha=0.5,
                s=15,
                edgecolor=None
            )
            plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")

        elif label_mode == "continuous":
            plot_df["Label"] = self.pca_labels[idx]

            sns.scatterplot(
                data=plot_df,
                x="PC1",
                y="PC2",
                hue="Label",
                palette="coolwarm",
                alpha=0.5,
                s=15,
                edgecolor=None
            )
            plt.legend(title="Value", bbox_to_anchor=(1.02, 1), loc="upper left")

        else:
            print("label_mode must be 'delay_categorical', 'generic_categorical', or 'continuous'.")
            return self

        plt.title(f"PCA: Flight Data Patterns (n={len(idx):,})", fontweight="bold", fontsize=14)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.tight_layout()
        plt.show()

        return self

    def run_umap(self, feature_cols=None, n_neighbors=15, min_dist=0.1, max_samples=100000):
        """
        Fit UMAP on the selected feature columns.

        UMAP is a nonlinear dimensionality reduction method intended mainly for
        visualization and exploration. It tends to preserve local neighborhoods
        while also retaining more global structure than t-SNE, and it usually
        scales better to larger datasets. :contentReference[oaicite:4]{index=4}

        Parameters
        ----------
        feature_cols : list or None, default=None
            Columns to use for UMAP. If None, a default set is used.
        n_neighbors : int, default=15
            Number of nearest neighbors used in the neighborhood graph.
            Smaller values emphasize local structure more strongly.
        min_dist : float, default=0.1
            Minimum distance between embedded points. Smaller values allow
            tighter clusters in the 2D layout.
        max_samples : int, default=100000
            Maximum number of rows to use for fitting UMAP. If the dataset is
            larger, a random sample is taken for speed and readability.

        Returns
        -------
        DimReduction
            The current object.
        """
        if self.verbose:
            print("\n--- Running UMAP (Non-linear) ---")

        cols_present = self._get_feature_columns(feature_cols)

        if len(cols_present) < 2:
            print("Not enough feature columns found to run UMAP.")
            return self

        if len(self.data) > max_samples:
            if self.verbose:
                print(f"Dataset too large. Downsampling to {max_samples:,} rows...")
            np.random.seed(42)
            idx = np.random.choice(len(self.data), max_samples, replace=False)
            numeric_data = self.data.iloc[idx][cols_present]
            self.umap_labels = self.labels[idx]
        else:
            numeric_data = self.data[cols_present]
            self.umap_labels = self.labels.copy()

        if self.verbose:
            print(f"Features used: {cols_present}")
            print("Calculating UMAP...")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            init="random",
            random_state=42
        )
        self.umap_result = reducer.fit_transform(numeric_data)

        if self.verbose:
            print("UMAP complete.")

        return self

    def plot_umap(self, label_mode="delay_categorical"):
        """
        Plot the first two UMAP dimensions.

        Label modes
        -----------
        - 'delay_categorical'
            Bin numeric delay labels into delay groups.
        - 'generic_categorical'
            Use raw labels as categories.
        - 'continuous'
            Use raw numeric labels as a continuous color gradient.

        Parameters
        ----------
        label_mode : str, default='delay_categorical'
            Coloring mode for the plot.

        Returns
        -------
        DimReduction
            The current object.
        """
        if self.umap_result is None:
            print("Please run_umap() first.")
            return self

        plot_df = pd.DataFrame({
            "UMAP1": self.umap_result[:, 0],
            "UMAP2": self.umap_result[:, 1]
        })

        plt.figure(figsize=(9, 7))

        if label_mode == "delay_categorical":
            plot_df["Label"] = self._categorize_delay_labels(self.umap_labels)

            hue_order = ["On-time / Early", "Minor delay", "Moderate delay", "Long delay"]
            palette = {
                "On-time / Early": "#4575b4",
                "Minor delay": "#91bfdb",
                "Moderate delay": "#fdae61",
                "Long delay": "#d73027"
            }

            sns.scatterplot(
                data=plot_df,
                x="UMAP1",
                y="UMAP2",
                hue="Label",
                hue_order=hue_order,
                palette=palette,
                alpha=0.5,
                s=15,
                edgecolor=None
            )
            plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=4)

        elif label_mode == "generic_categorical":
            plot_df["Label"] = self._generic_categorical_labels(self.umap_labels)

            sns.scatterplot(
                data=plot_df,
                x="UMAP1",
                y="UMAP2",
                hue="Label",
                alpha=0.5,
                s=15,
                edgecolor=None
            )
            plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")

        elif label_mode == "continuous":
            plot_df["Label"] = self.umap_labels

            sns.scatterplot(
                data=plot_df,
                x="UMAP1",
                y="UMAP2",
                hue="Label",
                palette="coolwarm",
                alpha=0.5,
                s=15,
                edgecolor=None
            )
            plt.legend(title="Value", bbox_to_anchor=(1.02, 1), loc="upper left")

        else:
            print("label_mode must be 'delay_categorical', 'generic_categorical', or 'continuous'.")
            return self

        plt.title("UMAP: Flight Data Patterns", fontweight="bold", fontsize=14)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()
        plt.show()

        return self