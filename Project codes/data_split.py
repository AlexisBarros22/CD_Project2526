import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataSplit:
    def __init__(self, data: pd.DataFrame, test_size=0.2, random_state=48):
        self.data = data
        self.test_size = test_size
        self.random_state = random_state

        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        # scaler + columns used for scaling
        self.scaler = StandardScaler()
        self.numeric_cols = None

        self._load_data()

    def _load_data(self):
        X = self.data.drop(columns=['ARR_DELAY'])
        y = self.data['ARR_DELAY']

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # pick numeric columns only
            self.numeric_cols = X_train.select_dtypes(include="number").columns

            # fit scaler on TRAIN numeric cols only (no leakage)
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            X_train_scaled[self.numeric_cols] = self.scaler.fit_transform(X_train[self.numeric_cols])
            X_test_scaled[self.numeric_cols] = self.scaler.transform(X_test[self.numeric_cols])

            self.data_train = X_train_scaled
            self.labels_train = y_train
            self.data_test = X_test_scaled
            self.labels_test = y_test

            print(
                f"Data split successful: {len(X_train)} training samples, {len(X_test)} testing samples. "
                f"Scaled {len(self.numeric_cols)} numeric columns."
            )

        except Exception as e:
            print(f"Error during data split: {e}")

