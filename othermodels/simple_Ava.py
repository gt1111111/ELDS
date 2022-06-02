import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Subsetting the dataset
df = pd.read_csv('../data/bridge11.csv', nrows=243)

mean=-0.02816923
std=0.0198381
df["y"]=(df["y"]-mean)/std
# Creating train and test set
train = df[0:200]
test = df[200:]

# Aggregating the dataset at daily level
df['Timestamp'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.index = df['Timestamp']

train['Timestamp'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
train.index = train['Timestamp']

test['Timestamp'] = pd.to_datetime(test['date'], format='%Y-%m-%d')
test.index = test['Timestamp']

y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['y'].mean()
plt.figure(figsize=(12,8))
plt.plot(train['y'], label='Train')
plt.plot(test['y'], label='Test')
#plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()
testScore = math.sqrt(mean_squared_error(test['y'], y_hat_avg['avg_forecast']))
print('Test Score: %.2f RMSE' % (testScore))

n = len(test['y'])
mae = sum(np.abs(test['y'] - y_hat_avg['avg_forecast'])) / n
mse = sum(np.square(test['y'] - y_hat_avg['avg_forecast'])) / n
print('Test Score: %.2f mse' % (mse))
print('Test Score: %.2f mae' % (mae))