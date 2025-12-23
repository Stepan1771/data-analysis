import matplotlib.pyplot as plt
import numpy as np

def plot_pca_variance(variance):
    plt.plot(np.cumsum(variance))
    plt.xlabel("Количество факторов")
    plt.ylabel("Накопленная объяснённая дисперсия")
    plt.grid()
    plt.show()


def plot_clusters(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df['Factor1'],
        df['Factor2'],
        c=df['Cluster'],
        cmap='viridis'
    )
    plt.xlabel("Фактор 1")
    plt.ylabel("Фактор 2")
    plt.colorbar(label="Кластер")
    plt.show()
