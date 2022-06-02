from data.window_outlier import Windows
from data.mean import MEAN
from data.lof import LOF

import numpy as np
import pandas as pd


def data_clean():
    filename = "data/bridge1.csv"
    num1 = np.array(LOF.start(filename))
    num2 = np.array(MEAN.start(filename))
    num3 = np.array(Windows.start(filename))
    num = num1+num2+num3
    lloc = np.array(np.where(num >= 2))
    df = pd.read_csv(filename)
    df["roll"] = df.rolling(5).mean()
    for i in lloc[0]:
        print(i)
        if i >= 4:
            df.loc[i, 'y'] = df.loc[i]['roll']

    df = df.drop(['roll'], axis=1)

    df.loc[(df['y']==0.0001)|(df['y']==0.00110),"y"]=None
    df['date']=pd.to_datetime(df['date'])
    df.set_index(['date'],inplace=True)
    df['y']=df.interpolate(method='time')

    df = df.reset_index()

    # df['y'][0] = df['y'][1]
    df.loc[0, 'y'] = df.loc[1]['y']

    df.to_csv('data/bridge111.csv',index=None)


if __name__ == '__main__':
    data_clean()