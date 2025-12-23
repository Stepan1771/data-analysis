import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()

    def encode_and_scale(self, df: pd.DataFrame):
        cat_cols = ['Manufacturer', 'Vehicle type']
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        features = df_encoded.drop(['Model', 'Latest Launch'], axis=1)

        scaled = self.scaler.fit_transform(features)
        X = pd.DataFrame(scaled, columns=features.columns)

        return df_encoded, X
