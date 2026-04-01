import pandas as pd
import numpy as np
from typing import Dict

def build_eunis_code_map(eunis_mapping: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Returns dict: EUNIS_code -> array of raster values (ids) representing that code.
    Handles duplicates (same EUNIS string with multiple ids).
    """
    required = {"Value", "EUNIS"}
    missing = required - set(eunis_mapping.columns)
    if missing:
        raise ValueError(f"eunis_mapping missing columns: {missing}")

    # Ensure integer ids
    df = eunis_mapping.copy()
    df["Value"] = df["Value"].astype(int)

    # Group ids by EUNIS code
    code_map = (
        df.groupby("EUNIS")["Value"]
        .apply(lambda s: np.array(sorted(s.unique()), dtype=np.int32))
        .to_dict()
    )
    return code_map

def build_clc_code_map(clc_mapping: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Returns dict: clc_code -> array of raster values (ids) representing that code.
    Handles duplicates (same clc string with multiple ids).
    """
    required = {"Value", "CODE_18"}
    missing = required - set(clc_mapping.columns)
    if missing:
        raise ValueError(f"clc_mapping missing columns: {missing}")

    # Ensure integer ids
    df = clc_mapping.copy()
    df["Value"] = df["Value"].astype(int)

    # Group ids by EUNIS code
    code_map = (
        df.groupby("CODE_18")["Value"]
        .apply(lambda s: np.array(sorted(s.unique()), dtype=np.int32))
        .to_dict()
    )
    return code_map