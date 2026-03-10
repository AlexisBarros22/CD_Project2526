import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class DataSplit:
    def __init__(self, data: pd.DataFrame, test_size=0.2, random_state=48, verbose=True):
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

        self.categorical_cols = ['ORIGIN_CITY', 'DEST_CITY', 'ORIGIN_STATE', 'DEST_STATE', 'ROUTE']
        self.normalize_columns = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME', 'AVG_SPEED', 'DEP_HOUR', 'ARR_HOUR']

        self._split_data()
        self._encode_categorical()
        self._scale_numeric()

    def _split_data(self):
        X = self.data.drop(columns=['ARR_DELAY'])
        y = self.data['ARR_DELAY']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.data_train_eda = X_train.copy()
        self.data_test_eda = X_test.copy()

        self.data_train = X_train.copy()
        self.data_test = X_test.copy()
        self.labels_train = y_train
        self.labels_test = y_test

    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    def _encode_categorical(self):
        # ---------- 1) Shared mapping for states ----------
        state_cols = [col for col in ['ORIGIN_STATE', 'DEST_STATE'] if col in self.data_train.columns]

        if state_cols:
            all_states = pd.concat(
                [self.data_train[col].astype(str) for col in state_cols],
                axis=0
            ).unique()

            all_states = sorted(all_states)
            self.state_mapping = {state: i for i, state in enumerate(all_states)}

            for col in state_cols:
                self.data_train[col] = self.data_train[col].astype(str).map(self.state_mapping).fillna(-1).astype(int)
                self.data_test[col] = self.data_test[col].astype(str).map(self.state_mapping).fillna(-1).astype(int)

        # ---------- 2) Symmetric encoding for route ----------
        if 'ROUTE' in self.data_train.columns:
            def canonical_route(route):
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

        # ---------- 3) Separate encoding for remaining categorical columns ----------
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