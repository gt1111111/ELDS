import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    preds = np.load("pred.npy")
    trues = np.load("true.npy")
    print(preds)
    for i in range(32):
        new_preds = preds[i, :, 0]
        new_trues = trues[i, :, 0]
        plt.figure(figsize=(16, 8))
        plt.plot(new_preds, label='pred')
        plt.plot(new_trues, label='trues')
        plt.legend()
    plt.show()

    print(np.load("metrics.npy"))
