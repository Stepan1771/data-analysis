from sklearn.decomposition import PCA
import pandas as pd

class PCAAnalyzer:
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, X: pd.DataFrame):
        return self.pca.fit_transform(X)

    def explained_variance(self):
        return self.pca.explained_variance_ratio_
