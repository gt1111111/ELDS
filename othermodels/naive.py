import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Subsetting the dataset
df = pd.read_csv('../data/bridge111.csv', nrows=243)

mean=-0.02816923
std=0.0198381
df["y"]=(df["y"]-mean)/std
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

# Plotting data
train.y.plot(figsize=(15, 8), title='predict', fontsize=14)
test.y.plot(figsize=(15, 8), title='predict', fontsize=14)

dd = np.asarray(train['y'])
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

testScore = math.sqrt(mean_squared_error(test['y'], y_hat['naive']))
print('Test Score: %.2f RMSE' % (testScore))
n = len(test['y'])
mae = sum(np.abs(test['y'] - y_hat['naive'])) / n
mse = sum(np.square(test['y'] - y_hat['naive'])) / n
print('Test Score: %.2f mse' % (mse))
print('Test Score: %.2f mae' % (mae))