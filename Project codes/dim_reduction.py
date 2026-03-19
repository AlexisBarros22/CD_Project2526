import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

class DimReduction:
    def __init__(self, data, labels, verbose=True):
        self.data = pd.DataFrame(data)
        self.labels = np.array(labels)
        self.verbose = verbose

        self.pca_result = None
        self.pca_model = None
        self.umap_result = None
        self.umap_labels = None

        sns.set_theme(style="whitegrid", rc={
            "axes.facecolor": "#f8f9fa",
            "figure.facecolor": "white"
        })

    def run_pca(self):
        if self.verbose:
            print("\n--- Running PCA (Linear) ---")

        target_cols = ["AVG_SPEED", "DISTANCE", "CRS_ELAPSED_TIME", "ARR_DELAY"]
        cols_present = [col for col in target_cols if col in self.data.columns]

        if len(cols_present) < 2:
            print("Not enough target columns found to run PCA.")
            return self

        numeric_data = self.data[cols_present]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        self.pca_model = PCA()
        self.pca_result = self.pca_model.fit_transform(scaled_data)

        if self.verbose:
            var_explained = self.pca_model.explained_variance_ratio_
            print(f"PC1 explains {var_explained[0] * 100:.2f}% of the variance.")
            if len(var_explained) > 1:
                print(f"PC2 explains {var_explained[1] * 100:.2f}% of the variance.")

        return self

    def plot_pca(self, max_samples=100000):
        if self.pca_result is None:
            print("Please run_pca() first.")
            return self

        n_rows = self.pca_result.shape[0]

        if n_rows > max_samples:
            if self.verbose:
                print(f"Plotting a random sample of {max_samples} points for visibility...")
            np.random.seed(42)
            idx = np.random.choice(n_rows, max_samples, replace=False)
        else:
            idx = np.arange(n_rows)

        labels_subset = self.labels[idx]
        status_vec = np.where(labels_subset < 15, "On-time",
                              np.where(labels_subset <= 30, "Short delay", "Long delay"))

        plot_df = pd.DataFrame({
            'PC1': self.pca_result[idx, 0],
            'PC2': self.pca_result[idx, 1],
            'Status': status_vec
        })

        hue_order = ["On-time", "Short delay", "Long delay"]
        palette = {"On-time": "#4575b4", "Short delay": "#fdae61", "Long delay": "#d73027"}

        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            data=plot_df, x='PC1', y='PC2', hue='Status',
            hue_order=hue_order, palette=palette,
            alpha=0.5, s=15, edgecolor=None  # s is the dot size
        )

        plt.title(f"PCA: Flight Data Patterns ({len(idx) // 1000}k Sample)", fontweight='bold', fontsize=14)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="", loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        plt.tight_layout()
        plt.show()

        return self

    def run_umap(self, n_neighbors=15, min_dist=0.1, max_samples=100000):
        if self.verbose:
            print("\n--- Running UMAP (Non-linear) ---")

        target_cols = ["AVG_SPEED", "DISTANCE", "CRS_ELAPSED_TIME", "ARR_DELAY"]
        cols_present = [col for col in target_cols if col in self.data.columns]

        if len(cols_present) < 2:
            print("Not enough target columns found to run UMAP.")
            return self

        if len(self.data) > max_samples:
            if self.verbose:
                print(f"Dataset too large! Downsampling to a random {max_samples} flights...")
            np.random.seed(42)
            idx = np.random.choice(len(self.data), max_samples, replace=False)

            numeric_data = self.data.iloc[idx][cols_present]
            self.umap_labels = self.labels[idx]
        else:
            numeric_data = self.data[cols_present]
            self.umap_labels = self.labels

        if self.verbose:
            print("Calculating UMAP (this will take 1-3 minutes)...")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            init='random',
            random_state=42
        )
        self.umap_result = reducer.fit_transform(numeric_data)

        if self.verbose:
            print("UMAP complete!")

        return self

    def plot_umap(self):
        if self.umap_result is None:
            print("Please run_umap() first.")
            return self

        status_vec = np.where(self.umap_labels < 15, "On-time",
                              np.where(self.umap_labels <= 30, "Short delay", "Long delay"))

        plot_df = pd.DataFrame({
            'UMAP1': self.umap_result[:, 0],
            'UMAP2': self.umap_result[:, 1],
            'Status': status_vec
        })

        hue_order = ["On-time", "Short delay", "Long delay"]
        palette = {"On-time": "#4575b4", "Short delay": "#fdae61", "Long delay": "#d73027"}

        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            data=plot_df, x='UMAP1', y='UMAP2', hue='Status',
            hue_order=hue_order, palette=palette,
            alpha=0.5, s=15, edgecolor=None
        )

        plt.title("UMAP: Flight Data Patterns", fontweight='bold', fontsize=14)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(title="", loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        plt.tight_layout()
        plt.show()

        return self