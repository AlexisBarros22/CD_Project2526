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
            'AIR_TIME', 'CANCELLATION_CODE', 'AIRLINE', 'AIRLINE_CODE',
            'AIRLINE_DOT', 'FL_NUMBER', 'ORIGIN', 'DEST'
        ]
        self.data.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        if self.verbose:
            print(self.data.head())

        return self

    def report_missing_values(self):
        if not self.verbose:
            return self

        print(f"Total flights: {len(self.data)}")
        print("\nNA values per column:")
        print(self.data.isnull().sum())

        na_rows = self.data.isnull().any(axis=1).sum()
        print(f"\nTotal rows with at least one NA value: {na_rows}")
        print(f"Percentage of rows with NA: {(na_rows / len(self.data) * 100):.2f}%")

        if 'CANCELLED' in self.data.columns:
            print(f"\nCancelled flights: {self.data['CANCELLED'].sum()}")
        if 'DIVERTED' in self.data.columns:
            print(f"Diverted flights: {self.data['DIVERTED'].sum()}")

        return self

    def filter_cancelled_diverted(self):
        if {'CANCELLED', 'DIVERTED'}.issubset(self.data.columns):
            self.data = self.data[
                (self.data['CANCELLED'] == 0) & (self.data['DIVERTED'] == 0)
            ]
            self.data.drop(columns=['CANCELLED', 'DIVERTED'], inplace=True, errors="ignore")

        if self.verbose:
            print(f"\nTotal flights after filtering: {len(self.data)}")

        return self

    def clean_na(self):
        if self.verbose:
            print("\nNA values before dropping:")
            print(self.data.isnull().sum())

        self.data.dropna(inplace=True)

        if self.verbose:
            print(f"\nTotal flights after dropping NA: {len(self.data)}")

        return self

    def add_date_features(self):
        if 'FL_DATE' in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data['FL_DATE']):
                self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'], errors='coerce')

            # Extract Year, Month, and Day of Week
            self.data['FL_YEAR'] = self.data['FL_DATE'].dt.year
            self.data['FL_MONTH'] = self.data['FL_DATE'].dt.month
            self.data['FL_DAY_OF_WEEK'] = self.data['FL_DATE'].dt.isocalendar().day.astype(int)

            if self.verbose:
                print("\nDate features extracted:")
                # Updated the print list to include FL_YEAR
                print(self.data[['FL_DATE', 'FL_YEAR', 'FL_MONTH', 'FL_DAY_OF_WEEK']].head())

            self.data.drop(columns=['FL_DATE'], inplace=True, errors='ignore')

        return self

    def convert_scheduled_times(self):
        time_cols = ['CRS_DEP_TIME', 'CRS_ARR_TIME']

        for col in time_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

                hours = self.data[col] // 100
                minutes = self.data[col] % 100
                self.data[col] = hours * 60 + minutes

                if self.verbose:
                    print(f"\nConverted {col} to minutes since midnight:")
                    print(self.data[col].head())

        return self

    def convert_to_season(self):
        if 'FL_MONTH' in self.data.columns:
            if 'SEASON' not in self.data.columns:
                self.data['SEASON'] = self.data['FL_MONTH'].apply(
                    lambda x: (
                         1 if x in [12, 1, 2] else
                         2 if x in [3, 4, 5] else
                         3 if x in [6, 7, 8] else
                         4 if x in [9, 10, 11] else 'Unknown'
                    )
            )
            if self.verbose:
                print("\nSeason feature created:")
                print(self.data[['FL_MONTH', 'SEASON']].head())

        return self

    def is_weekend(self):
        if 'FL_DAY_OF_WEEK' in self.data.columns:
            if 'IS_WEEKEND' not in self.data.columns:
                self.data['IS_WEEKEND'] = self.data['FL_DAY_OF_WEEK'].apply(
                    lambda x: 1 if x in [6, 7] else 0
                )
            if self.verbose:
                print("\nWeekend feature created:")
                print(self.data[['FL_DAY_OF_WEEK', 'IS_WEEKEND']].head())

        return self

    def route(self):
        if {'ORIGIN_CITY', 'DEST_CITY'}.issubset(self.data.columns):
            if 'ROUTE' not in self.data.columns:
                self.data['ROUTE'] = self.data['ORIGIN_CITY'] + "_" + self.data['DEST_CITY']

            if self.verbose:
                print("\nRoute feature created:")
                print(self.data[['ORIGIN_CITY', 'DEST_CITY', 'ROUTE']].head())

        return self

    def avg_speed(self):
        if {'DISTANCE', 'CRS_ELAPSED_TIME'}.issubset(self.data.columns):
            if 'AVG_SPEED' not in self.data.columns:
                self.data['AVG_SPEED'] = self.data['DISTANCE'] / self.data['CRS_ELAPSED_TIME']

            if self.verbose:
                print("\nAverage speed feature created:")
                print(self.data[['DISTANCE', 'CRS_ELAPSED_TIME', 'AVG_SPEED']].head())

        return self

    def dep_hour(self):
        if 'CRS_DEP_TIME' in self.data.columns:
            if 'DEP_HOUR' not in self.data.columns:
                self.data['DEP_HOUR'] = self.data['CRS_DEP_TIME'] // 60

            if self.verbose:
                print("\nDeparture hour feature created:")
                print(self.data[['CRS_DEP_TIME', 'DEP_HOUR']].head())

        return self

    def arr_hour(self):
        if 'CRS_ARR_TIME' in self.data.columns:
            if 'ARR_HOUR' not in self.data.columns:
                self.data['ARR_HOUR'] = self.data['CRS_ARR_TIME'] // 60

            if self.verbose:
                print("\nArrival hour feature created:")
                print(self.data[['CRS_ARR_TIME', 'ARR_HOUR']].head())

        return self

    def peak_morning(self):
        if 'DEP_HOUR' in self.data.columns:
            if 'PEAK_MORNING' not in self.data.columns:
                self.data['PEAK_MORNING'] = self.data['DEP_HOUR'].apply(
                    lambda x: 1 if 7 <= x <= 10 else 0
                )

            if self.verbose:
                print("\nMorning peak feature created:")
                print(self.data[['DEP_HOUR', 'PEAK_MORNING']].head())

        return self

    def peak_evening(self):
        if 'DEP_HOUR' in self.data.columns:
            if 'PEAK_EVENING' not in self.data.columns:
                self.data['PEAK_EVENING'] = self.data['DEP_HOUR'].apply(
                    lambda x: 1 if 16 <= x <= 19 else 0
                )

            if self.verbose:
                print("\nEvening peak feature created:")
                print(self.data[['DEP_HOUR', 'PEAK_EVENING']].head())

        return self

    def origin_state(self):
        if 'ORIGIN_CITY' in self.data.columns:
            if 'ORIGIN_STATE' not in self.data.columns:
                self.data['ORIGIN_STATE'] = (
                    self.data['ORIGIN_CITY']
                    .astype(str)
                    .str.split(',')
                    .str[-1]
                    .str.strip()
                )

            if self.verbose:
                print("\nOrigin state feature created:")
                print(self.data[['ORIGIN_CITY', 'ORIGIN_STATE']].head())

        return self

    def dest_state(self):
        if 'DEST_CITY' in self.data.columns:
            if 'DEST_STATE' not in self.data.columns:
                self.data['DEST_STATE'] = (
                    self.data['DEST_CITY']
                    .astype(str)
                    .str.split(',')
                    .str[-1]
                    .str.strip()
                )

            if self.verbose:
                print("\nDestination state feature created:")
                print(self.data[['DEST_CITY', 'DEST_STATE']].head())

        return self



    def export_to_csv(self, path: str):
        self.data.to_csv(path, index=False)
        if self.verbose:
            print(f"\nData exported to: {path}")
        return self

    def get_data(self) -> pd.DataFrame:
        return self.data