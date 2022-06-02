import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Subsetting the dataset
df = pd.read_csv('../data/bridge111.csv', nrows=243)

mean = -0.02816923
std = 0.0198381
df["y"] = (df["y"] - mean) / std
# Creating train and test set
train = df[0:220]
test = df[220:]

# Aggregating the dataset at daily level
df['Timestamp'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.index = df['Timestamp']

train['Timestamp'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
train.index = train['Timestamp']

test['Timestamp'] = pd.to_datetime(test['date'], format='%Y-%m-%d')
test.index = test['Timestamp']
from statsmodels.tsa.api import ExponentialSmoothing

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['y']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['y'], label='Train')
plt.plot(test['y'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('bridge data')
plt.show()

testScore = math.sqrt(mean_squared_error(test['y'], y_hat_avg['Holt_Winter']))
print('Test Score: %.2f RMSE' % (testScore))
n = len(test['y'])
mae = sum(np.abs(test['y'] - y_hat_avg['Holt_Winter'])) / n
mse = sum(np.square(test['y'] - y_hat_avg['Holt_Winter'])) / n
print('Test Score: %.2f mse' % (mse))
print('Test Score: %.2f mae' % (mae))
