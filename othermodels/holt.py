import math
from othermodels import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import data


def holt(clean=True):
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

    from statsmodels.tsa.api import Holt

    y_hat_avg = test.copy()
    # smoothing_level:值越高，对最近观察的权重就越大.   smoothing_slope:使用Holt方法的预测在未来会无限期地增加或减少,我们使用具有阻尼参数(0 <φ<1)的阻尼趋势方法来防止预测“失控”
    fit = Holt(np.asarray(train['y'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    y_hat_avg['Holt_linear'] = fit.forecast(len(test))

    plt.figure(figsize=(16, 8))
    plt.plot(train['y'], label='Train')
    plt.plot(test['y'], label='Test')
    plt.xlabel('time')
    plt.ylabel('bridge data')
    plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("holt.png")

    testScore = math.sqrt(mean_squared_error(test['y'], y_hat_avg['Holt_linear']))
    print('Test Score: %.2f RMSE' % (testScore))
    n = len(test['y'])
    print(y_hat_avg['Holt_linear'])
    print(test['y'])

    # mae = sum(np.abs(test['y'].values - y_hat_avg['Holt_linear'].values)) / n
    # mse = sum(np.square(test['y'].values - y_hat_avg['Holt_linear'].values)) / n

    mae = metrics.MAE(y_hat_avg['Holt_linear'].values, test['y'].values)
    mse = metrics.MSE(y_hat_avg['Holt_linear'].values, test['y'].values)
    print(mse)
    print('Test Score: %.2f mse' % (mse))
    print('Test Score: %.2f mae' % (mae))
    print(metrics.metric(y_hat_avg['Holt_linear'].values, test['y'].values))


if __name__ == '__main__':
    holt()
