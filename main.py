import pandas as pd

from core.data.load_data import DataLoader
from core.data.encode_scale import Preprocessor
from core.analysis.pca_analysis import PCAAnalyzer
from core.analysis.clustering import ClusterAnalyzer
from core.analysis.regression import RegressionAnalyzer
from core.visualization.plots import (
    plot_pca_variance,
    plot_clusters,
)


def main():
    # 1. Данные
    loader = DataLoader("data_files/Car_sales.csv")
    df = loader.load_and_clean()

    # 2. Предобработка
    prep = Preprocessor()
    df_encoded, X = prep.encode_and_scale(df)

    # 3. PCA
    pca = PCAAnalyzer()
    X_pca = pca.fit_transform(X)
    plot_pca_variance(pca.explained_variance())

    df_encoded['Factor1'] = X_pca[:, 0]
    df_encoded['Factor2'] = X_pca[:, 1]

    # 4. Кластеры
    cluster = ClusterAnalyzer(n_clusters=3)
    df_encoded['Cluster'] = cluster.fit_predict(X_pca)
    plot_clusters(df_encoded)

    # 5. Регрессия
    reg = RegressionAnalyzer()
    features, coefs = reg.train(df_encoded)

    # Создаём DataFrame с признаками и коэффициентами
    coef_df = pd.DataFrame({
        'Признак': features,
        'Коэффициент': coefs
    })
    coef_df = coef_df.sort_values(by='Коэффициент', ascending=False)
    df_manufacturers = coef_df[coef_df['Признак'].str.startswith('Manufacturer')].copy()
    df_manufacturers['Производитель'] = df_manufacturers['Признак'].str.replace('Manufacturer_', '', regex=False)
    df_manufacturers = df_manufacturers[['Производитель', 'Коэффициент']].sort_values(by='Коэффициент', ascending=False)
    print("=== Топ-10 производителей по влиянию на цену автомобиля ===")
    print(df_manufacturers.head(10).to_string(index=False))
    return None


if __name__ == "__main__":
    main()
