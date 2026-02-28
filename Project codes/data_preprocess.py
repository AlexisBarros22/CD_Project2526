import pandas as pd
class DataPreprocess:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # Implement your preprocessing steps here
        # For example, you can handle missing values, encode categorical variables, etc.
        # This is just a placeholder implementation
        self.data.fillna(0, inplace=True)  # Example: Fill missing values with 0
        return self.data