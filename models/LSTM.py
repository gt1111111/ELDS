import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTM(nn.Module):
    hidden_cell: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=args.d_in, hidden_size=args.d_model, num_layers=args.n_layers,
                            batch_first=True, bidirectional=args.bidirectional)
        if args.bidirectional:
            self.fc = nn.Linear(args.d_model * 2, args.d_out)
        else:
            self.fc = nn.Linear(args.d_model, args.d_out)

    def forward(self, x):
        out, _ = self.lstm(x, self.hidden_cell)
        out = self.fc(out[:, -1:, :])
        print(out)
        return out




class Seq2Seq(nn.Module):
    hidden_cell: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.pred_len = args.pred_len
        self.encoder = nn.LSTM(input_size=args.d_in, hidden_size=args.d_model, num_layers=args.n_layers,
                               batch_first=True, bidirectional=args.bidirectional)
        self.decoder = nn.LSTM(input_size=args.d_in, hidden_size=args.d_model, num_layers=args.n_layers,
                               batch_first=True, bidirectional=args.bidirectional)
        self.fc = nn.Linear(args.d_model, args.d_out)

    def forward(self, x, y, train=True):
        # containing the final hidden state for each element in the sequence.
        #  containing the final cell state for each element in the sequence.
        x, (h, c) = self.encoder(x, self.hidden_cell)
        if train:
            y, _ = self.decoder(y, (h, c))
            out = self.fc(y)
        else:
            output = []
            out = y[:, :1, :]
            for i in range(self.pred_len):
                out, (h, c) = self.decoder(out, (h, c))
                out = self.fc(out)
                output.append(out)
            out = torch.cat(output, dim=1)
        return out
