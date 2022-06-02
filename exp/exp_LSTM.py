import os.path
import pathlib

import torch
import torch.nn as nn
import numpy as np

from models.LSTM import LSTM
from models.LSTM import Seq2Seq
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
from exp.exp_basic import Exp_Basic
from data.data_factory import data_provider
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

figure_rootPath = ""
figure_dataPath = ""


class ExpLSTM(Exp_Basic):
    def __init__(self, args):
        super(ExpLSTM, self).__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_model(self):
        model = LSTM(self.args)
        global figure_rootPath
        global figure_dataPath
        figure_rootPath = self.args.root_path
        figure_dataPath = self.args.data_path
        return model

    def train(self, train=True):
        _, train_loader = self._get_data(flag='train')
        _, val_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        checkpoint_path = os.path.join(self.args.checkpoint, self.args.setting)
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epoch):
            self.model.train()
            for i, (batch_x, batch_x_dec, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()

                batch_x = batch_x.to(self.args.gpu)
                batch_x_dec = batch_x_dec.to(self.args.gpu)
                batch_y = batch_y.to(self.args.gpu)
                output = self.model(batch_x)
                # output = self.model(batch_x, batch_x_dec, train=True)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            val_loss = self.val(val_loader, criterion)
            self.test(stop=True)
            early_stopping(val_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)
        best_model_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def val(self, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_dec, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(self.args.gpu)
                batch_x_dec = batch_x_dec.to(self.args.gpu)
                batch_y = batch_y.to(self.args.gpu)
                output = self.model(batch_x)
                # output = self.model(batch_x, batch_x_dec, train=False)

                loss = criterion(output, batch_y)
                total_loss.append(loss.item())
        return sum(total_loss) / len(total_loss)

    def test(self, test=False, train=False, stop=False):
        if test:  # if test, then load the trained best model
            print('loading model')
            checkpoint_path = pathlib.Path(self.args.checkpoint, self.args.setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))

        dataset, dataloader = self._get_data(flag='test')

        preds = []
        trues = []

        result_path = os.path.join(self.args.result, self.args.setting)
        pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_dec, batch_y) in enumerate(dataloader):
                batch_x = batch_x.to(self.args.gpu)
                batch_x_dec = batch_x_dec.to(self.args.gpu)
                batch_y = batch_y.to(self.args.gpu)
                # output = self.model(batch_x, batch_x_dec, train=False)
                output = self.model(batch_x)
                preds.append(output.cpu().detach().numpy())
                trues.append(batch_y.cpu().detach().numpy())

                # batch = output.cpu().detach().numpy()
                # value = 0;
                # for i in batch:
                #     value += batch[i]
                # value = value / 12

        print(i)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        predss = np.squeeze(preds)
        truess = np.squeeze(trues)
        # 画图
        if not stop:
            global figure_rootPath
            global figure_dataPath
            df_figure = pd.read_csv(figure_rootPath + figure_dataPath)
            plt.clf()
            mean = dataset.scaler.mean
            std = dataset.scaler.std
            df_figure["count"] = (df_figure["count"] - mean) / std
            length = len(df_figure["count"])
            x = df_figure['slot'][0:int(length*0.8)]
            y = df_figure['count'][0:int(length*0.8)]
            x1 = df_figure['slot'][int(length*0.8):]
            my_x_ticks = np.arange(0, len(df_figure["count"]), 4000)
            plt.xticks(my_x_ticks, rotation=70)


            plt.xlabel('time')
            plt.ylabel('data')
            plt.plot(x, y, color='b', label='real value')
            plt.plot(x1, predss, color='r', label='predict value')
            plt.plot(x1, truess, color='g', label='real value', alpha = 0.4)
            plt.legend()
            plt.show()
        # plt.savefig("lstm.png")
        # plt.figure(figsize=(16, 8))
        # plt.plot(predss, label='pred')
        # plt.plot(truess, label='trues')
        # plt.legend()
        # plt.show()

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        raw_preds = dataset.inverse_transform(preds)
        raw_trues = dataset.inverse_transform(trues)
        mape = np.mean(np.abs((raw_preds - raw_trues) / raw_trues)) * 100
        print('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))

        np.save(result_path + '/preds.npy', preds)
        np.save(result_path + '/trues.npy', trues)

        return preds, trues
