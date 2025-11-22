import pandas as pd
import matplotlib.pyplot as plt

def plot_density(df: pd.DataFrame, column: str) -> None:
    """Grafica la densidad de un dato numérico.

    Args:
        df (pd.DataFrame): DataFrame.
        column (str): Columna numérica.
    """
    df[column].dropna().plot(kind="density")
    plt.title(f"Densidad de {column}")
    plt.xlabel(column)
    plt.show()
