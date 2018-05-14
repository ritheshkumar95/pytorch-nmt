import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def to_var(arr):
    return Variable(torch.from_numpy(arr)).cuda()


class Encoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.drop = nn.Dropout(cf.src_params['dropout'])
        self.emb = nn.Embedding(
            cf.src_params['vocab_size'],
            cf.src_params['emb_size']
        )

        self.rnn = nn.LSTM(
            cf.src_params['emb_size'],
            cf.src_params['hidden_size'],
            cf.src_params['num_layers'],
            dropout=cf.src_params['dropout'],
            batch_first=True,
            bidirectional=True
        )

    def forward(self, src, lengths):
        sorted_lengths, idxs = torch.sort(
            torch.tensor(lengths).cuda(), dim=0, descending=True
        )
        _, inverse_idxs = torch.sort(
            idxs, dim=0
        )
        src = src[idxs]

        src = self.drop(self.emb(src))
        src = pack_padded_sequence(src, sorted_lengths.tolist(), batch_first=True)
        out, hidden = self.rnn(src)

        f_h = hidden[0][::2, inverse_idxs]
        f_c = hidden[1][::2, inverse_idxs]
        b_h = hidden[0][1::2, inverse_idxs]
        b_c = hidden[1][1::2, inverse_idxs]

        hidden = (torch.cat([f_h, b_h], -1), torch.cat([f_c, b_c], -1))
        return hidden


class Decoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.drop = nn.Dropout(cf.src_params['dropout'])
        self.emb = nn.Embedding(
            cf.trg_params['vocab_size'],
            cf.trg_params['emb_size']
        )

        self.rnn = nn.LSTM(
            cf.trg_params['emb_size'],
            cf.trg_params['hidden_size'],
            cf.trg_params['num_layers'],
            dropout=cf.src_params['dropout'],
            batch_first=True
        )

        self.fc = nn.Linear(
            cf.trg_params['hidden_size'],
            cf.trg_params['vocab_size']
        )

    def forward(self, src, hidden):
        src = self.drop(self.emb(src))
        out, hidden = self.rnn(src, hidden)
        out = self.fc(self.drop(out))
        return out, hidden


class Seq2Seq(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.encoder = Encoder(cf)
        self.decoder = Decoder(cf)
        self.ntokens = cf.trg_params['vocab_size']

    def forward(self, src, src_lengths, trg):
        enc = self.encoder(src, src_lengths)
        return self.decoder(trg, enc)[0]

    def inference(self, src, src_lengths, sos, maxlen=30):
        x_t = np.full((src.size(0), 1), sos).astype('int64')
        x_t = to_var(x_t)
        h_t = self.encoder(src, src_lengths)

        outputs = []
        for i in range(maxlen):
            probs, h_t = self.decoder(x_t, h_t)
            x_t = F.softmax(probs, -1).max(-1)[1]
            outputs.append(x_t)
        return torch.cat(outputs, 1)

    def score(self, src, src_lengths, trg, mask):
        source = trg[:, :-1]
        target = trg[:, 1:]
        mask = mask[:, 1:].contiguous()

        outputs = self.forward(src, src_lengths, source)
        loss = self.criterion(
            outputs.view(-1, self.ntokens),
            target.contiguous().view(-1)
        ) * mask.view(-1)
        loss = loss.sum() / mask.sum()
        return loss


if __name__ == '__main__':
    from config import load_config
    cf = load_config('../config/baseline.py')

    src = np.random.randint(
        0,
        cf.src_params['vocab_size'],
        (64, 20)
    ).astype('int64')
    src_lengths = sorted(
        np.random.randint(1, 20, 64).tolist(),
        reverse=True
    )

    trg = np.random.randint(
        0,
        cf.trg_params['vocab_size'],
        (64, 30)
    ).astype('int64')

    enc = Encoder(cf).cuda()
    dec = Decoder(cf).cuda()

    src_out = enc(to_var(src), src_lengths)
    print("Source encoded size: ", src_out.size())
    dec_out = dec(src_out, to_var(trg))
    print("Target decoded size: ", dec_out.size())
