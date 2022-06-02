from data.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer
from utils.toolsnew import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.timefeatures import time_features

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time
import joblib

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Transformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Transformer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train(1)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, batch_y, MS='MS'==self.args.features)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, MS='MS'==self.args.features)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _transforming(self, _data):
        # 调列的顺序，把时间维放首列，被预测维放尾列
        cols = list(_data.columns)
        cols.remove(self.args.target)
        cols.remove('date')
        _data = _data[['date'] + cols + [self.args.target]]

        # 编码时间戳
        _stamp = pd.DataFrame({'date': pd.to_datetime(_data.date)})
        timeenc = 0 if self.args.embed != 'timeF' else 1
        if timeenc == 0:
            _stamp['month'] = _stamp.date.apply(lambda row: row.month, 1)
            _stamp['day'] = _stamp.date.apply(lambda row: row.day, 1)
            _stamp['weekday'] = _stamp.date.apply(lambda row: row.weekday(), 1)
            _stamp['hour'] = _stamp.date.apply(lambda row: row.hour, 1)
            _stamp['minute'] = _stamp.date.apply(lambda row: row.minute, 1)
            _stamp['minute'] = _stamp.minute.map(lambda x: x // 15)
            _stamp = _stamp.drop(['date'], 1).values
        elif timeenc == 1:
            _stamp = time_features(pd.to_datetime(_stamp['date'].values), freq=self.args.freq)
            _stamp = _stamp.transpose(1, 0)

        # 取数据维，并正则化
        # encoder input
        assert self.args.features in ['M', 'MS', 'S']
        if self.args.features == 'M' or self.args.features == 'MS':
            cols_data = _data.columns[1:]
            _data = _data[cols_data]
        else:
            _data = _data[[self.args.target]]
        _data = self.scaler.transform(_data)
        return _data, _stamp

    def infer(self, setting, enc_data, dec_data):
        # 这个接口会反复调用，所以设计思路跟train、val、test不太一样
        if setting != self.args.setting:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            train_data, _ = self._get_data(flag='train')
            self.scaler = train_data.transform()
            joblib.dump(self.scaler, 'api/threshold_performer_std_scaler.bin', compress=True)
            self.unscaler = train_data.inverse_transform
            self.args.setting = setting
        data_x, batch_x_mark = self._transforming(enc_data)
        dec_inp, batch_y_mark = self._transforming(dec_data)

        # 推理
        self.model.eval()
        with torch.no_grad():
            data_x = torch.from_numpy(data_x).float().to(self.device).unsqueeze(dim=0)
            dec_inp = torch.from_numpy(dec_inp).float().to(self.device).unsqueeze(dim=0)
            dec_inp = torch.cat([data_x[:, self.args.label_len:, :], dec_inp], dim=1)
            batch_x_mark = torch.from_numpy(batch_x_mark).float().to(self.device).unsqueeze(dim=0)
            batch_y_mark = torch.from_numpy(batch_y_mark).float().to(self.device).unsqueeze(dim=0)
            batch_y_mark = torch.cat([batch_x_mark[:, self.args.label_len:, :], batch_y_mark], dim=1)
            outputs = self.model(data_x, batch_x_mark, dec_inp, batch_y_mark, MS=True)[0]
        outputs = self.unscaler(torch.cat([outputs, outputs], dim=-1).cpu())[..., :-1]
        return outputs, dec_inp
