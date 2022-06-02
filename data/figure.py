import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# #显示所有列
# pd.set_option('display.max_columns', None)
#
# #显示所有行
# pd.set_option('display.max_rows', None)
#
# #设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)
def show_figure(clean=False):
    if clean:
        # df = pd.read_csv('bridge111.csv')
        df = pd.read_csv('data/bridge111.csv')
    else:
        print("gt")
        # df = pd.read_csv('bridge1.csv')
        df = pd.read_csv('data/bridge1.csv')
    x = df['date']
    y = df['y']

    plt.clf()
    my_x_ticks = np.arange(0, 243, 45)
    plt.xticks(my_x_ticks)
    plt.xlabel('time')
    plt.ylabel('beidge data')
    plt.plot(x, y)
    # plt.show()
    plt.savefig("data.png")


if __name__ == '__main__':
    show_figure(True)
