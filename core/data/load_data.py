import pandas as pd
import numpy as np


class DataLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_clean(self):
        df = pd.read_csv(self.file_path)

        df.map(lambda x: x.strip() if isinstance(x, str) else x)

        df.replace(".", np.nan, inplace=True)

        numeric_cols = [
            'Sales in thousands', '4-year resale value', 'Price in thousands',
            'Engine size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
            'Curb weight', 'Fuel capacity', 'Fuel efficiency'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        df_clean = df.dropna()

        print("Размер датасета после очистки:", df_clean.shape)
        return df_clean