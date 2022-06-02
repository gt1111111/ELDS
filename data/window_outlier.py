import pandas as pd
import numpy as np

class Windows:
    def start(filename):
        df = pd.read_csv(filename, index_col='date')
        df["is_out"] = (df >= df.rolling(5).mean()*1.3) | (df < df.rolling(5).mean()*0.7)
        abDot = [0 if item else 1 for item in df["is_out"]]

        return abDot
