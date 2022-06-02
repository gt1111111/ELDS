import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from math import sqrt
# Subsetting the dataset
df = pd.read_csv('../data/beijing.csv', skiprows=lambda x: x % 5 != 0)
# Creating train and test set
train = df[0:len(df)-1000]
test = df[len(df)-1000:]

# Aggregating the dataset at daily level
df['slot'] = pd.to_datetime(df['slot'], format='%Y-%m-%d %H:%M')
df.index = df['slot']

train['slot'] = pd.to_datetime(train['slot'], format='%Y-%m-%d %H:%M')
train.index = train['slot']

test['slot'] = pd.to_datetime(test['slot'], format='%Y-%m-%d %H:%M')
test.index = test['slot']

y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train['count'], order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start=len(df)-1000, end=len(df), dynamic=True)
plt.figure(figsize=(16, 8))
plt.plot(train['count'], label='Train')
plt.plot(test['count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
print(y_hat_avg['SARIMA'])
plt.legend(loc='best')
plt.show()
testScore = math.sqrt(mean_squared_error(test['count'], y_hat_avg['SARIMA']))

print('Test Score: %.2f RMSE' % (testScore))
n = len(test['count'])
mae = sum(np.abs(test['count'] - y_hat_avg['SARIMA'])) / n+1
mse = sum(np.square(test['count'] - y_hat_avg['SARIMA'])) / n+1
print('Test Score: %.2f mse' % (mse))
print('Test Score: %.2f mae' % (mae))















