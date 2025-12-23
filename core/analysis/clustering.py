from sklearn.cluster import KMeans

class ClusterAnalyzer:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def inertia(self, X, max_k=9):
        values = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            values.append(kmeans.inertia_)
        return values
