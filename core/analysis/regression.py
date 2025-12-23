import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

class RegressionAnalyzer:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, df: pd.DataFrame):
        y = df['Price in thousands']
        X = df.drop(['Price in thousands', 'Model', 'Latest Launch'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print("R²:", r2_score(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        return X.columns, self.model.coef_
