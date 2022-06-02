import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import SimpleExpSmoothing


def simple_exp(clean=True):
    # Subsetting the dataset
    if clean:
        df = pd.read_csv('data/bridge111.csv', nrows=243)
    else:
        df = pd.read_csv('data/bridge1.csv', nrows=243)

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

    y_hat_avg = test.copy()
    fit = SimpleExpSmoothing(np.asarray(train['y'])).fit(smoothing_level=0.6, optimized=False)
    y_hat_avg['SES'] = fit.forecast(len(test))
    plt.figure(figsize=(16, 8))
    plt.plot(train['y'], label='Train')
    plt.plot(test['y'], label='Test')
    testScore = math.sqrt(mean_squared_error(test['y'], y_hat_avg['SES']))
    print('Test Score: %.2f RMSE' % (testScore))
    plt.xlabel('time')
    plt.ylabel('bridge data')
    plt.plot(y_hat_avg['SES'], label='SES')
    plt.legend(loc='best')
    # plt.show()

    plt.savefig("simple_exp.png")

    rms = sqrt(mean_squared_error(test['y'], y_hat_avg['SES']))
    print(rms)

    n = len(test['y'])
    mae = sum(np.abs(test['y'] - y_hat_avg['SES'])) / n
    mse = sum(np.square(test['y'] - y_hat_avg['SES'])) / n
    print('Test Score: %.2f mse' % (mse))
    print('Test Score: %.2f mae' % (mae))


if __name__ == '__main__':
    simple_exp()
