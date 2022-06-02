import argparse
import random
import torch
import numpy as np

from exp.exp_LSTM import ExpLSTM


def LSTM(clean=True):
    # fix the random seed for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # parse the arguments
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model_id', type=str, default='LSTM', help='model name')
    parser.add_argument('--model', type=str, default='LSTM', help='model name')

    # data config
    parser.add_argument('--data', type=str, required=False, default='y', help='dataset name')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path')
    if clean:
        parser.add_argument('--data_path', type=str, default='beijing.csv', help='training file name')
    else:
        parser.add_argument('--data_path', type=str, default='beijing.csv', help='training file name')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--features', type=str, default='S', help='task type, options: [M, S, MS]')
    parser.add_argument('--target', type=str, default='count', help='target names')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min')
    parser.add_argument('--checkpoint', type=str, default='checkpoints', help='checkpoint path')
    parser.add_argument('--result', type=str, default='results', help='result path')

    # task config
    parser.add_argument('--sample_rate', type=int, default=1, help='sample rate')
    parser.add_argument('--seq_len', type=int, default=96, help='input length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction length')
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--repeat', type=int, default=1, help='repeat times')
    parser.add_argument('--repeat_id', type=int, default=2, help='repeat id (only used in test)')

    # model config
    parser.add_argument('--d_in', type=int, default=1, help='input size')
    parser.add_argument('--d_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='hidden size')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--bidirectional', type=bool, default=False, help='bidirectional')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # training config
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--train_epoch', type=int, default=10, help='train epoch')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

    # gpu config
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # other config
    parser.add_argument('--desc', type=str, default='', help='description')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # instantiate the experiment
    exp = ExpLSTM(args)

    # run the training or testing
    if args.is_training:
        for i in range(args.repeat):
            # set record of experiment
            args.setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nl{}_{}_{}'.format(
                args.model_id, args.model, args.data, args.features, args.seq_len, args.pred_len,
                args.d_model, args.n_layers, args.desc, i)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))
            exp.train()

            print('>>>>>>>testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))
            exp.test()
            torch.cuda.empty_cache()
    else:
        args.setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nl{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.pred_len,
            args.d_model, args.n_layers, args.desc, args.repeat_id)
        exp.test(test=True)


if __name__ == '__main__':
    LSTM()
