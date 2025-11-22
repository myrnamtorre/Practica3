import pandas as pd

def check_columns_exist(df: pd.DataFrame, columns: list[str]) -> bool:
    """Verifica si las columnas existen en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        columns (list[str]): Columnas a verificar.

    Returns:
        bool: True si todas existen, False en caso contrario.
    """
    return all(col in df.columns for col in columns)
