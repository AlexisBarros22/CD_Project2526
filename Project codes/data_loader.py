import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

class DataLoader:
    def __init__(self):
        self.data = self._load_data()
        self.show_data()

    def _load_data(self):
        try:
            path = kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")
            file_path = f"{path}/flights_sample_3m.csv"
            df = pd.read_csv(file_path, encoding='latin1')
            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def show_data(self):
        if self.data is not None:
            print("Data Preview:")
            print(self.data.head())
            print("Data Summary:")
            print(self.data.describe())
        else:
            print("No data loaded")

