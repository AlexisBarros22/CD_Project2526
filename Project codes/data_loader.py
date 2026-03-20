import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


class DataLoader:
    """
    Data loading utility for the flight delay dataset from Kaggle.

    This class handles:
    - Downloading the dataset using kagglehub
    - Loading a specific CSV file into a pandas DataFrame
    - Displaying a preview and summary of the dataset

    Notes
    -----
    - Uses a sample dataset (`flights_sample_3m.csv`) for faster loading.
    - Requires KaggleHub to be properly configured with API credentials.
    """

    def __init__(self):
        """
        Initialize the DataLoader and automatically load the dataset.

        Attributes
        ----------
        data : pd.DataFrame or None
            Loaded dataset. None if loading fails.
        """
        self.data = self._load_data()
        self.show_data()

    def _load_data(self):
        """
        Download and load the flight dataset from Kaggle.

        Returns
        -------
        pd.DataFrame or None
            Loaded dataset as a DataFrame, or None if an error occurs.
        """
        try:
            # Download dataset from Kaggle
            path = kagglehub.dataset_download(
                "patrickzel/flight-delay-and-cancellation-dataset-2019-2023"
            )

            # Select sample file (faster than full dataset)
            file_path = f"{path}/flights_sample_3m.csv"

            # Load CSV into DataFrame
            df = pd.read_csv(file_path, encoding='latin1')

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def show_data(self):
        """
        Display a preview and summary of the dataset.

        Outputs
        -------
        - First 5 rows of the dataset
        - Summary statistics of numeric columns

        Returns
        -------
        None
        """
        if self.data is not None:
            print("Data Preview:")
            print(self.data.head())

            print("\nData Summary:")
            print(self.data.describe())

        else:
            print("No data loaded")

    def get_data(self):
        """
        Retrieve the loaded dataset.

        Returns
        -------
        pd.DataFrame or None
            The loaded dataset.
        """
        return self.data