import pandas as pd
from pandas.api.types import is_integer_dtype


def map_col(series: pd.Series, mapping: pd.Series) -> pd.Series:
    """Smart mapping the values of a column with a series"""
    dtype = mapping.index.dtype
    if is_integer_dtype(dtype):
        dtype = "Int64"
    return series.astype(dtype).map(mapping).fillna(series)
