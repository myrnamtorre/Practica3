import pandas as pd
import matplotlib.pyplot as plt

def plot_bar(df: pd.DataFrame, column: str) -> None:
    """Dibuja un gráfico de barras para valores categóricos.

    Args:
        df (pd.DataFrame): DataFrame.
        column (str): Columna categórica.
    """
    counts = df[column].value_counts()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"Barras de {column}")
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    plt.show()
