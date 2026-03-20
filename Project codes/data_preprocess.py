import pandas as pd
import numpy as np


class DataPreprocess:
    """
    Preprocessing and feature-engineering pipeline for a flight-delay dataset.

    This class is designed to prepare raw flight records for exploratory data
    analysis and predictive modeling. It performs column removal, missing-value
    inspection and handling, filtering of cancelled/diverted flights, and
    creation of several derived temporal and route-based features.

    Workflow
    --------
    Typical usage follows this order:
    1. Drop leakage or irrelevant columns.
    2. Inspect missing values.
    3. Remove cancelled/diverted flights.
    4. Drop remaining missing values.
    5. Create date-based and time-based engineered features.
    6. Export or retrieve the processed dataframe.

    Notes
    -----
    - The target is assumed to be `ARR_DELAY`.
    - Scheduled times are transformed using cyclical encoding.
    - Negative arrival delays are optionally clipped to zero.
    - Most methods are chainable and return `self`.

    Parameters
    ----------
    data : pd.DataFrame
        Input raw or semi-cleaned flight dataframe.
    verbose : bool, default=True
        Whether to print progress information and preview outputs.

    Attributes
    ----------
    data : pd.DataFrame
        Internal working copy of the dataset.
    verbose : bool
        Whether to print progress information.
    """

    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        """
        Initialize the preprocessing object.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe.
        verbose : bool, default=True
            Whether to print intermediate outputs.
        """
        self.data = data.copy()
        self.verbose = verbose

    def drop_columns(self):
        """
        Drop columns that are irrelevant, redundant, or may introduce leakage.

        These columns include realised delay breakdowns, post-flight timestamps,
        airline identifiers, and other fields not intended for modeling.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Report missing-value statistics and cancellation/diversion counts.

        Prints:
        - total number of rows
        - NA count per column
        - number and percentage of rows with at least one NA
        - counts of cancelled and diverted flights if available

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Remove cancelled and diverted flights.

        Rows are retained only when:
        - `CANCELLED == 0`
        - `DIVERTED == 0`

        After filtering, the `CANCELLED` and `DIVERTED` columns are dropped.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if {'CANCELLED', 'DIVERTED'}.issubset(self.data.columns):
            self.data = self.data[
                (self.data['CANCELLED'] == 0) & (self.data['DIVERTED'] == 0)
            ]
            self.data.drop(columns=['CANCELLED', 'DIVERTED'], inplace=True, errors="ignore")

        if self.verbose:
            print(f"\nTotal flights after filtering: {len(self.data)}")

        return self

    def clean_na(self):
        """
        Drop all rows containing missing values.

        This is a complete-case approach and should typically be used after
        filtering and before modeling.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if self.verbose:
            print("\nNA values before dropping:")
            print(self.data.isnull().sum())

        self.data.dropna(inplace=True)

        if self.verbose:
            print(f"\nTotal flights after dropping NA: {len(self.data)}")

        return self

    def add_date_features(self):
        """
        Extract year, month, and day-of-week features from `FL_DATE`.

        Generated features
        ------------------
        - `FL_YEAR`
        - `FL_MONTH`
        - `FL_DAY_OF_WEEK`

        The original `FL_DATE` column is dropped afterwards.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if 'FL_DATE' in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data['FL_DATE']):
                self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'], errors='coerce')

            self.data['FL_YEAR'] = self.data['FL_DATE'].dt.year
            self.data['FL_MONTH'] = self.data['FL_DATE'].dt.month
            self.data['FL_DAY_OF_WEEK'] = self.data['FL_DATE'].dt.isocalendar().day.astype(int)

            if self.verbose:
                print("\nDate features extracted:")
                print(self.data[['FL_DATE', 'FL_YEAR', 'FL_MONTH', 'FL_DAY_OF_WEEK']].head())

            self.data.drop(columns=['FL_DATE'], inplace=True, errors='ignore')

        return self

    def convert_scheduled_times_cyclical(self):
        """
        Convert scheduled departure and arrival times into cyclical features.

        For each available scheduled time column:
        - `CRS_DEP_TIME`
        - `CRS_ARR_TIME`

        the method creates:
        - `<column>_sin`
        - `<column>_cos`

        based on minutes since midnight, then drops the original raw time column.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        time_cols = ['CRS_DEP_TIME', 'CRS_ARR_TIME']

        for col in time_cols:
            if col in self.data.columns:
                time_numeric = pd.to_numeric(self.data[col], errors='coerce')

                hours = time_numeric // 100
                minutes = time_numeric % 100
                total_minutes = hours * 60 + minutes

                minutes_in_day = 24 * 60

                self.data[f'{col}_sin'] = np.sin(2 * np.pi * total_minutes / minutes_in_day)
                self.data[f'{col}_cos'] = np.cos(2 * np.pi * total_minutes / minutes_in_day)

                self.data.drop(columns=[col], inplace=True)

                if self.verbose:
                    print(f"\nConverted {col} to cyclical features:")
                    print(self.data[[f'{col}_sin', f'{col}_cos']].head())

        return self

    def convert_to_season(self):
        """
        Create a season feature from `FL_MONTH`.

        Season encoding
        ---------------
        - 1: Winter  (Dec, Jan, Feb)
        - 2: Spring  (Mar, Apr, May)
        - 3: Summer  (Jun, Jul, Aug)
        - 4: Autumn  (Sep, Oct, Nov)

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Create a binary weekend indicator from `FL_DAY_OF_WEEK`.

        Encoded as:
        - 1 for Saturday or Sunday
        - 0 otherwise

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Create a route feature by concatenating origin and destination city.

        Generated feature
        -----------------
        - `ROUTE` = `ORIGIN_CITY` + "_" + `DEST_CITY`

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if {'ORIGIN_CITY', 'DEST_CITY'}.issubset(self.data.columns):
            if 'ROUTE' not in self.data.columns:
                self.data['ROUTE'] = self.data['ORIGIN_CITY'] + "_" + self.data['DEST_CITY']

            if self.verbose:
                print("\nRoute feature created:")
                print(self.data[['ORIGIN_CITY', 'DEST_CITY', 'ROUTE']].head())

        return self

    def avg_speed(self):
        """
        Create average scheduled speed as distance divided by scheduled elapsed time.

        Generated feature
        -----------------
        - `AVG_SPEED` = `DISTANCE` / `CRS_ELAPSED_TIME`

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if {'DISTANCE', 'CRS_ELAPSED_TIME'}.issubset(self.data.columns):
            if 'AVG_SPEED' not in self.data.columns:
                self.data['AVG_SPEED'] = self.data['DISTANCE'] / self.data['CRS_ELAPSED_TIME']

            if self.verbose:
                print("\nAverage speed feature created:")
                print(self.data[['DISTANCE', 'CRS_ELAPSED_TIME', 'AVG_SPEED']].head())

        return self

    def dep_hour(self):
        """
        Create departure hour from scheduled departure time.

        Generated feature
        -----------------
        - `DEP_HOUR` = `CRS_DEP_TIME` // 60

        Note
        ----
        This method assumes `CRS_DEP_TIME` is expressed in minutes since midnight.
        If cyclical conversion has already dropped `CRS_DEP_TIME`, this method will
        do nothing.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if 'CRS_DEP_TIME' in self.data.columns:
            if 'DEP_HOUR' not in self.data.columns:
                self.data['DEP_HOUR'] = self.data['CRS_DEP_TIME'] // 60

            if self.verbose:
                print("\nDeparture hour feature created:")
                print(self.data[['CRS_DEP_TIME', 'DEP_HOUR']].head())

        return self

    def arr_hour(self):
        """
        Create arrival hour from scheduled arrival time.

        Generated feature
        -----------------
        - `ARR_HOUR` = `CRS_ARR_TIME` // 60

        Note
        ----
        This method assumes `CRS_ARR_TIME` is expressed in minutes since midnight.
        If cyclical conversion has already dropped `CRS_ARR_TIME`, this method will
        do nothing.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        if 'CRS_ARR_TIME' in self.data.columns:
            if 'ARR_HOUR' not in self.data.columns:
                self.data['ARR_HOUR'] = self.data['CRS_ARR_TIME'] // 60

            if self.verbose:
                print("\nArrival hour feature created:")
                print(self.data[['CRS_ARR_TIME', 'ARR_HOUR']].head())

        return self

    def peak_morning(self):
        """
        Create a morning-peak indicator from `DEP_HOUR`.

        Encoded as:
        - 1 for departure hours between 7 and 10 inclusive
        - 0 otherwise

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Create an evening-peak indicator from `DEP_HOUR`.

        Encoded as:
        - 1 for departure hours between 16 and 19 inclusive
        - 0 otherwise

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Extract origin state from `ORIGIN_CITY`.

        Assumes city values follow a format such as:
        'City Name, ST'

        Generated feature
        -----------------
        - `ORIGIN_STATE`

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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
        """
        Extract destination state from `DEST_CITY`.

        Assumes city values follow a format such as:
        'City Name, ST'

        Generated feature
        -----------------
        - `DEST_STATE`

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
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

    def fix_negative_delays(self):
        """
        Replace negative arrival delays with zero.

        This converts early arrivals and negative delay values into a
        non-negative target representation.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        columns = ['ARR_DELAY']
        for column in columns:
            if column in self.data.columns:
                negative_count = (self.data[column] < 0).sum()
                if negative_count > 0:
                    if self.verbose:
                        print(f"\nFound {negative_count} negative values in '{column}'. Setting them to 0.")
                    self.data.loc[self.data[column] < 0, column] = 0

        return self

    def export_to_csv(self, path: str):
        """
        Export the processed dataframe to a CSV file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        DataPreprocess
            The current preprocessing object.
        """
        self.data.to_csv(path, index=False)
        if self.verbose:
            print(f"\nData exported to: {path}")
        return self

    def get_data(self) -> pd.DataFrame:
        """
        Return the processed dataframe.

        Returns
        -------
        pd.DataFrame
            The processed dataset.
        """
        return self.data