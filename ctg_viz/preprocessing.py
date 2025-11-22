import pandas as pd

def clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convierte columnas seleccionadas a valores numéricos.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        columns (list[str]): Columnas que deben convertirse.

    Returns:
        pd.DataFrame: Copia del DataFrame con columnas convertidas.
    """
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    return df_copy


def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Rellena valores faltantes según una estrategia.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        strategy (str): Estrategia de relleno ("mean", "median", "mode").

    Returns:
        pd.DataFrame: DataFrame sin valores faltantes.
    """
    df_copy = df.copy()

    for col in df_copy.select_dtypes(include="number").columns:
        if strategy == "mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == "median":
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == "mode":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    return df_copy
