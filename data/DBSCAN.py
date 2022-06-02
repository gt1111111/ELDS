import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

df = pd.read_csv(r'bridge1.csv', index_col='date')
dbscan = DBSCAN(eps=.3)
dbscan.fit(df)

results = pd.concat((df, pd.DataFrame(dbscan.labels_, index=df.index, columns=['Class'])), axis=1)
outliers = results[results['Class'] == -1]
print(outliers)
np.zeros((len(df),), dtype=int)
plt.scatter(df.index, df.iloc[:, 0], s=10, c=dbscan.labels_)
plt.show()
