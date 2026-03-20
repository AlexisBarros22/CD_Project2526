import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class DataSplit:
    """
    Split a cleaned flight-delay dataset into training and testing sets,
    preserve copies for EDA, encode categorical variables, and scale
    selected numeric features.

    This class is intended for preprocessing after cleaning/feature engineering
    and before model training.

    Workflow
    --------
    1. Split the dataset into train and test partitions.
    2. Keep unencoded copies of the split data for EDA.
    3. Encode categorical variables using train-set-derived mappings only.
    4. Scale selected numeric columns using a StandardScaler fitted on the
       training set only.

    Notes
    -----
    - The target column is assumed to be `ARR_DELAY`.
    - State columns are encoded using one shared mapping across both
      `ORIGIN_STATE` and `DEST_STATE`.
    - Route values are encoded symmetrically, so routes like `A_B` and `B_A`
      map to the same canonical representation.
    - Remaining categorical columns are ordinal-encoded with unseen test values
      mapped to `-1`.
    - Only columns listed in `normalize_columns` are scaled, and only if they
      are present in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing predictors and target.
    test_size : float, default=0.2
        Proportion of the dataset to reserve for testing.
    random_state : int, default=48
        Random seed used for reproducible train/test splitting.
    verbose : bool, default=True
        Whether to print status information during preprocessing.

    Attributes
    ----------
    data_train_eda : pd.DataFrame or None
        Training predictors before encoding/scaling, useful for EDA.
    data_test_eda : pd.DataFrame or None
        Testing predictors before encoding/scaling, useful for EDA.
    data_train : pd.DataFrame or None
        Training predictors after encoding/scaling.
    data_test : pd.DataFrame or None
        Testing predictors after encoding/scaling.
    labels_train : pd.Series or None
        Training target values (`ARR_DELAY`).
    labels_test : pd.Series or None
        Testing target values (`ARR_DELAY`).
    scaler : StandardScaler
        Scaler fitted on selected numeric training columns.
    encoder : OrdinalEncoder or None
        Encoder fitted on remaining categorical columns.
    state_mapping : dict, optional
        Shared mapping used for state columns.
    route_mapping : dict, optional
        Mapping used for canonicalized route values.
    categorical_cols : list
        Categorical columns expected for encoding.
    normalize_columns : list
        Numeric columns expected for scaling.
    """

    def __init__(self, data: pd.DataFrame, test_size=0.2, random_state=48, verbose=True):
        """
        Initialize the DataSplit pipeline and run splitting, encoding, and scaling.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe containing features and target.
        test_size : float, default=0.2
            Fraction of rows assigned to the test set.
        random_state : int, default=48
            Random seed for reproducibility.
        verbose : bool, default=True
            Whether to print status messages.
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

        self.data_train_eda = None
        self.data_test_eda = None
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        self.scaler = StandardScaler()
        self.encoder = None

        self.categorical_cols = [
            'ORIGIN_CITY',
            'DEST_CITY',
            'ORIGIN_STATE',
            'DEST_STATE',
            'ROUTE'
        ]

        self.normalize_columns = [
            'CRS_DEP_TIME',
            'CRS_ARR_TIME',
            'DISTANCE',
            'CRS_ELAPSED_TIME',
            'AVG_SPEED',
            'DEP_HOUR',
            'ARR_HOUR'
        ]

        self._split_data()
        self._encode_categorical()
        self._scale_numeric()

    def _split_data(self):
        """
        Split the dataset into training and testing sets.

        The target column `ARR_DELAY` is separated from the predictors.
        Copies of the raw train/test predictor sets are stored for EDA before
        encoding and scaling are applied.

        Returns
        -------
        None
        """
        X = self.data.drop(columns=['ARR_DELAY'])
        y = self.data['ARR_DELAY']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.data_train_eda = X_train.copy()
        self.data_test_eda = X_test.copy()

        self.data_train = X_train.copy()
        self.data_test = X_test.copy()
        self.labels_train = y_train
        self.labels_test = y_test

    def _encode_categorical(self):
        """
        Encode categorical variables using mappings learned from the training set.

        Encoding strategy
        -----------------
        1. Shared mapping for `ORIGIN_STATE` and `DEST_STATE`
           - Both columns use the same dictionary so the same state always gets
             the same encoded value regardless of column.
        2. Symmetric encoding for `ROUTE`
           - Routes are canonicalized so `A_B` and `B_A` become the same route.
        3. Ordinal encoding for remaining categorical columns
           - Currently applies to columns such as `ORIGIN_CITY` and `DEST_CITY`.
           - Unknown categories in the test set are encoded as `-1`.

        Returns
        -------
        None
        """
        # ---------- 1) Shared mapping for state columns ----------
        state_cols = [col for col in ['ORIGIN_STATE', 'DEST_STATE'] if col in self.data_train.columns]

        if state_cols:
            all_states = pd.concat(
                [self.data_train[col].astype(str) for col in state_cols],
                axis=0
            ).unique()

            all_states = sorted(all_states)
            self.state_mapping = {state: i for i, state in enumerate(all_states)}

            for col in state_cols:
                self.data_train[col] = (
                    self.data_train[col]
                    .astype(str)
                    .map(self.state_mapping)
                    .fillna(-1)
                    .astype(int)
                )
                self.data_test[col] = (
                    self.data_test[col]
                    .astype(str)
                    .map(self.state_mapping)
                    .fillna(-1)
                    .astype(int)
                )

        # ---------- 2) Symmetric encoding for route ----------
        if 'ROUTE' in self.data_train.columns:
            def canonical_route(route):
                """
                Convert a route to a canonical order-independent representation.

                Examples
                --------
                'JFK_LAX' -> 'JFK_LAX'
                'LAX_JFK' -> 'JFK_LAX'

                Parameters
                ----------
                route : any
                    Raw route value.

                Returns
                -------
                str
                    Canonicalized route string.
                """
                parts = str(route).split('_')
                if len(parts) != 2:
                    return str(route)
                return '_'.join(sorted(parts))

            train_routes = self.data_train['ROUTE'].astype(str).apply(canonical_route)
            test_routes = self.data_test['ROUTE'].astype(str).apply(canonical_route)

            unique_routes = sorted(train_routes.unique())
            self.route_mapping = {route: i for i, route in enumerate(unique_routes)}

            self.data_train['ROUTE'] = train_routes.map(self.route_mapping).fillna(-1).astype(int)
            self.data_test['ROUTE'] = test_routes.map(self.route_mapping).fillna(-1).astype(int)

        # ---------- 3) Ordinal encoding for remaining categorical columns ----------
        remaining_cols = [
            col for col in self.categorical_cols
            if col in self.data_train.columns and col not in ['ORIGIN_STATE', 'DEST_STATE', 'ROUTE']
        ]

        if remaining_cols:
            self.encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )

            self.data_train[remaining_cols] = self.encoder.fit_transform(
                self.data_train[remaining_cols].astype(str)
            )
            self.data_test[remaining_cols] = self.encoder.transform(
                self.data_test[remaining_cols].astype(str)
            )

    def _scale_numeric(self):
        """
        Scale selected numeric columns using a StandardScaler.

        The scaler is fitted only on the training set and then applied to both
        training and testing data to avoid data leakage.

        Only columns listed in `normalize_columns` that are actually present
        in the dataset are scaled.

        Returns
        -------
        None
        """
        cols_present = [col for col in self.normalize_columns if col in self.data_train.columns]

        if not cols_present:
            return

        self.data_train[cols_present] = self.scaler.fit_transform(self.data_train[cols_present])
        self.data_test[cols_present] = self.scaler.transform(self.data_test[cols_present])

        if self.verbose:
            print(
                f"Data split successful: {len(self.data_train)} training samples, "
                f"{len(self.data_test)} testing samples.\n"
                f"Scaled columns: {cols_present}"
            )

    def export_encoding_mappings(self, path: str):
        """
        Export all learned encoding mappings to a CSV file.

        Exported mapping groups
        -----------------------
        1. Shared state mapping
        2. Route mapping
        3. OrdinalEncoder mappings for remaining categorical columns

        Parameters
        ----------
        path : str
            Output file path for the CSV export.

        Returns
        -------
        DataSplit
            The current DataSplit object.
        """
        rows = []

        # 1) Shared state mapping
        if hasattr(self, 'state_mapping') and self.state_mapping:
            for original_value, encoded_code in self.state_mapping.items():
                rows.append({
                    'Column': 'STATE_SHARED',
                    'Original_Value': original_value,
                    'Encoded_Code': encoded_code
                })

        # 2) Route mapping
        if hasattr(self, 'route_mapping') and self.route_mapping:
            for original_value, encoded_code in self.route_mapping.items():
                rows.append({
                    'Column': 'ROUTE',
                    'Original_Value': original_value,
                    'Encoded_Code': encoded_code
                })

        # 3) Remaining OrdinalEncoder mappings
        remaining_cols = [
            col for col in self.categorical_cols
            if col in self.data_train.columns and col not in ['ORIGIN_STATE', 'DEST_STATE', 'ROUTE']
        ]

        if hasattr(self, 'encoder') and self.encoder is not None:
            for i, col in enumerate(remaining_cols):
                categories = self.encoder.categories_[i]
                for code, original_value in enumerate(categories):
                    rows.append({
                        'Column': col,
                        'Original_Value': original_value,
                        'Encoded_Code': code
                    })

        if not rows:
            if self.verbose:
                print("No encoding mappings to export.")
            return self

        mappings_df = pd.DataFrame(rows)
        mappings_df.to_csv(path, index=False)

        if self.verbose:
            print(f"Encoding mappings exported to: {path}")
            print(mappings_df.head(10))

        return self