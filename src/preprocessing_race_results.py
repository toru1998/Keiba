from pathlib import Path

import pandas as pd
import numpy as np

race_results = pd.read_pickle(
    Path(__file__).resolve().parent.parent / "data" / "race_results.pkl"
)

race_results["sex"] = race_results["性齢"].str[0]
race_results["age"] = race_results["性齢"].str[1:]
