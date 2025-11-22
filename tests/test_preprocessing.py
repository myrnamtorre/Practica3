import pandas as pd
from ctg_viz.preprocessing import clean_numeric, fill_missing

def test_clean_numeric():
    df = pd.DataFrame({"a": ["1", "2", "x"]})
    result = clean_numeric(df, ["a"])

    # La fila con "x" debe convertirse en NaN
    assert pd.isna(result.loc[2, "a"])

    # El tipo debe ser float
    assert result["a"].dtype == "float64"


def test_fill_missing_mean():
    df = pd.DataFrame({"a": [1, None, 3]})
    result = fill_missing(df, "mean")

    # El valor faltante debe reemplazarse por el promedio (2)
    assert result.loc[1, "a"] == 2
