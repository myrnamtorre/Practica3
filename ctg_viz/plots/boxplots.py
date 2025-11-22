import pandas as pd
import matplotlib.pyplot as plt

def plot_box(df: pd.DataFrame, column: str) -> None:
    """Dibuja un boxplot.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        column (str): Columna numérica a graficar.
    """
    plt.boxplot(df[column].dropna())
    plt.title(f"Boxplot de {column}")
    plt.ylabel(column)
    plt.show()
