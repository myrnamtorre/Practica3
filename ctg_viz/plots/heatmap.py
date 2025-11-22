import pandas as pd
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Dibuja un heatmap de correlación.

    Args:
        df (pd.DataFrame): DataFrame numérico.
    """
    corr = df.corr()
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Mapa de calor de correlación")
    plt.show()
