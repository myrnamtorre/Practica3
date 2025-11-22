import pandas as pd
import numpy as np

def categorize_quantile(df: pd.DataFrame, column: str, q: int = 4, k: int | None = None) -> pd.Series:
    """
    Categoriza una columna numérica usando cuantiles.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        column (str): Columna a categorizar.
        q (int): Número de cuantiles.
        k (int | None): Alias alternativo usado por los tests. Si se usa, reemplaza a q.

    Returns:
        pd.Series: Categorías asignadas.
    """

    # Si el usuario proporciona k, usarlo
    if k is not None:
        q = k

    return pd.qcut(df[column], q=q, labels=False, duplicates="drop")



def categorize_bins(df: pd.DataFrame, column: str, bins: int = 4) -> pd.Series:
    """
    Categoriza una columna numérica dividiéndola en bins.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        column (str): Columna a categorizar.
        bins (int): Número de bins. Default = 4.

    Returns:
        pd.Series: Categorías asignadas.
    """
    return pd.cut(df[column], bins=bins, labels=False)
