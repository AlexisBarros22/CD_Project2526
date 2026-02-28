import pandas as pd

class DataPreprocess:
    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        self.data = data.copy()
        self.verbose = verbose

    def drop_columns(self):
        columns_to_drop = [
            'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_SECURITY',
            'DELAY_DUE_NAS', 'DELAY_DUE_LATE_AIRCRAFT', 'ARR_TIME', 'DEP_TIME',
            'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME',
            'AIR_TIME', 'CANCELLATION_CODE','AIRLINE','AIRLINE_CODE','AIRLINE_DOT','FL_NUMBER','ORIGIN','DEST'
        ]
        self.data.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        if self.verbose:
            print(self.data.head())

        return self  # enables chaining

    def report_missing_values(self):
        if not self.verbose:
            return self

        print(f"Total flights: {len(self.data)}")
        print(f"\nNA values per column:")
        print(self.data.isnull().sum())

        na_rows = self.data.isnull().any(axis=1).sum()
        print(f"\nTotal rows with at least one NA value: {na_rows}")
        print(f"Percentage of rows with NA: {(na_rows / len(self.data) * 100):.2f}%")

        # Cancelled / diverted summaries (guard in case columns missing)
        if 'CANCELLED' in self.data.columns:
            print(f"\nCancelled flights: {self.data['CANCELLED'].sum()}")
        if 'DIVERTED' in self.data.columns:
            print(f"Diverted flights: {self.data['DIVERTED'].sum()}")

        if 'CANCELLED' in self.data.columns:
            cancelled = self.data[self.data['CANCELLED'] == 1]
            if len(cancelled) > 0:
                print(f"\nNA values in CANCELLED flights:")
                print(cancelled.isnull().sum())

        if 'DIVERTED' in self.data.columns:
            diverted = self.data[self.data['DIVERTED'] == 1]
            if len(diverted) > 0:
                print(f"\nNA values in DIVERTED flights:")
                print(diverted.isnull().sum())

        return self

    def filter_cancelled_diverted(self):
        if self.verbose:
            print("\n" + "=" * 60)
            print("NOW filtering out cancelled/diverted flights...")
            print("=" * 60)

        # Only filter if both columns exist
        if {'CANCELLED', 'DIVERTED'}.issubset(self.data.columns):
            self.data = self.data[(self.data['CANCELLED'] == 0) & (self.data['DIVERTED'] == 0)]

            if self.verbose:
                print(f"\nTotal flights after filtering: {len(self.data)}")

            self.data.drop(columns=['CANCELLED', 'DIVERTED'], inplace=True, errors="ignore")

        return self

    def clean_na(self):
        if self.verbose:
            print("\n" + "=" * 60)
            print("Number of NA values before dropping:")
            print(self.data.isnull().sum())
            print("=" * 60)
        self.data.dropna(inplace=True)
        if self.verbose:
            print(f"\nTotal flights after dropping NA: {len(self.data)}")

        return self

    def timestamp_to_datetime(self):
        if 'FL_DATE' in self.data.columns:
            self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'], errors='coerce')

        return self

    def timestamp_to_date(self):
        if 'FL_DATE' in self.data.columns:
            # Ensure FL_DATE is datetime type
            if not pd.api.types.is_datetime64_any_dtype(self.data['FL_DATE']):
                self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'], errors='coerce')

            # Extract year, month, and day
            self.data['FL_YEAR'] = self.data['FL_DATE'].dt.year
            self.data['FL_MONTH'] = self.data['FL_DATE'].dt.month
            self.data['FL_DAY'] = self.data['FL_DATE'].dt.day

            if self.verbose:
                print("\nDate components extracted:")
                print(self.data[['FL_DATE', 'FL_YEAR', 'FL_MONTH', 'FL_DAY']].head())

        return self




    def get_data(self) -> pd.DataFrame:
        return self.data