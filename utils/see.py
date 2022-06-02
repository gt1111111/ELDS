import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def see_graph(pred):
    preds = np.load("pred.npy")
    trues = np.load("true.npy")

    preds = preds[0, :, 0]
    trues = trues[0, :, 0]

    plt.figure(figsize=(16, 8))

    plt.plot(range(15),preds, label='pred')
    plt.plot(range(15),trues, label='trues')
    plt.legend()
    plt.show()


if __name__ == '__main__':



    preds = np.load("pred.npy")
    trues = np.load("true.npy")
    preds = preds[0,:,0]
    trues = trues[0,:,0]

    plt.figure(figsize=(16, 8))
    plt.plot(range(15),preds, label='pred')
    plt.plot(range(15),trues, label='trues')
    plt.legend()
    plt.show()
