import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class EDA:
    """
    A class responsible for exploratory data analysis (EDA).

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        perform_eda(): Performs exploratory data analysis.
        plot_distributions(): Plots distributions of the features.
        plot_correlation_heatmap(): Plots a correlation heatmap between features and labels.
        plot_feature_importance(): Computes and visualizes feature importance using permutation importance.
    """

    def __init__(self, splitter):
        """
        Initializes the EDA class with a DataLoader object.
        """
        self.splitter = splitter

    def perform_eda(self):
        """
        Performs exploratory data analysis.
        """
        print("Exploratory Data Analysis (EDA) Report:")
        print("--------------------------------------")

        # Summary statistics
        print("\nSummary Statistics for train data:")
        print(self.splitter.data_train.describe())
        print("\nSummary Statistics for test data:")
        print(self.splitter.data_test.describe())

        # Distribution analysis
        print("\nDistribution Analysis:")
        self.plot_distributions()

        # Correlation analysis
        print("\nCorrelation Analysis:")
        self.plot_correlation_heatmap()

        # Feature Importance analysis
        print("\nFeature Importance Analysis:")
        self.plot_feature_importance()

    def plot_distributions(self):
        """
        Plots distributions of the features.
        """
        num_cols = len(self.splitter.data_train.columns)
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
        for i, feature in enumerate(self.splitter.data_train.columns):
            sns.histplot(data=self.splitter.data_train, x=feature, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plots a correlation heatmap between features and labels.
        """
        # Concatenate features and labels horizontally for correlation calculation
        data_with_labels = pd.concat([self.splitter.data_train, self.splitter.labels_train], axis=1)

        # Compute the correlation matrix
        corr_matrix = data_with_labels.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap between Features and Labels")
        plt.show()

    def plot_feature_importance(self, n_estimators=5, n_repeats=2):
        """
        Computes and visualizes feature importance using permutation importance.
        """
        # Fit a random forest classifier to compute feature importance
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(self.splitter.data_train, self.splitter.labels_train)

        # Compute permutation importance
        result = permutation_importance(clf, self.splitter.data_train, self.splitter.labels_train,
                                        n_repeats=n_repeats)
        sorted_idx = result.importances_mean.argsort()

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), self.splitter.data_train.columns[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')
        plt.show()