import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(df: pd.DataFrame, column: str) -> None:
    """Dibuja un histograma.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        column (str): Columna a graficar.
    """
    plt.hist(df[column].dropna())
    plt.title(f"Histograma de {column}")
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    plt.show()
