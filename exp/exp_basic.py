import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # short_model_dict_bridge = torch.load(
        #     f"checkpoints\informer_ETTh1_ftS_sl96_ll48_pl15_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0\checkpoint.pth",
        #     )
        # self.model.load_state_dict(short_model_dict_bridge)

    def _build_model(self):
        raise NotImplementedError
        return None

    # def _acquire_device(self):
    #     device = torch.device('cpu')
    #     print('Use CPU')
    #     return device

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    