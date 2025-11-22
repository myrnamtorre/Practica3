import pandas as pd
from ctg_viz.categorization import categorize_quantile, categorize_bins

def test_categorize_quantile():
    df = pd.DataFrame({"x": [10, 20, 30, 40]})
    result = categorize_quantile(df, "x", k=2)

    # Deben generarse 2 categorías
    assert result.nunique() == 2


def test_categorize_bins():
    df = pd.DataFrame({"x": [0, 5, 10, 15]})
    result = categorize_bins(df, "x", bins=2)

    # Deben crearse 2 categorías
    assert result.nunique() == 2
