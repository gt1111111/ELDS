import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MEAN:
    def start(filename):
        df = pd.read_csv(filename, index_col='date')
        sns.distplot(df['y'])

        dataMean = df['y'].mean()
        dataStd = df['y'].std()
        lowerL = (dataMean - 3 * dataStd) if (dataMean - 3 * dataStd) > 0 else 0
        upperL = dataMean + 3 * dataStd

        abDot = [0 if item else 1 for item in (df['y'] > upperL) | (df['y'] < lowerL)]

        return abDot

