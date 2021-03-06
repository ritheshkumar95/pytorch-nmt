import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def to_var(arr):
    return Variable(torch.from_numpy(arr)).cuda()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.project = nn.Linear(
            cf.trg_params['hidden_size'],
            cf.src_params['hidden_size'] * 2
        )

    def forward(self, key, value):
        B, T, D = key.size()
        scale = 1. / np.sqrt(D)

        key = self.project(key)
        att_weights = key.bmm(value.transpose(1, 2)) * scale
        att_probs = F.softmax(att_weights, -1)
        att_outputs = att_probs.bmm(value)
        return att_outputs.squeeze(1)


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
            bidirectional=True,
        )

    def forward(self, src, lengths):
        src = self.drop(self.emb(src))
        src = pack_padded_sequence(src, lengths, batch_first=True)
        out, _ = self.rnn(src)
        out = pad_packed_sequence(out, batch_first=True)
        return self.drop(out[0])


class Decoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.drop = nn.Dropout(cf.trg_params['dropout'])
        self.emb = nn.Embedding(
            cf.trg_params['vocab_size'],
            cf.trg_params['emb_size']
        )

        self.rnn = nn.LSTMCell(
            cf.trg_params['emb_size'] + cf.trg_params['hidden_size'],
            cf.trg_params['hidden_size']
        )
        self.att = ScaledDotProductAttention(cf)

        self.att_hidden = nn.Linear(
            cf.trg_params['hidden_size'] + cf.src_params['hidden_size'] * 2,
            cf.trg_params['hidden_size']
        )
        self.fc = nn.Linear(
            cf.trg_params['hidden_size'],
            cf.trg_params['vocab_size']
        )

        self.input_feed_init = nn.Parameter(
            torch.randn(cf.trg_params['hidden_size'])
        )
        self.hidden_init = nn.Parameter(
            torch.randn(cf.trg_params['hidden_size'] * 2)
        )

    def forward(self, src, trg, hidden=None, h_hat_t=None):
        trg = self.drop(self.emb(trg))

        if h_hat_t is None:
            h_hat_t = self.input_feed_init.expand(src.size(0), -1)

        if hidden is None:
            hidden = self.hidden_init.expand(src.size(0), -1)
            hidden = hidden.chunk(2, dim=-1)

        outputs = []
        for i in range(trg.size(1)):
            x_t = trg[:, i]
            hidden = self.rnn(
                torch.cat([x_t, h_hat_t], -1),
                hidden
            )
            h_t = self.drop(hidden[0])

            c_t = self.att(h_t[:, None], src)
            h_hat_t = F.tanh(self.att_hidden(
                torch.cat([h_t, c_t], -1)
            ))
            outputs.append(h_hat_t)

        outputs = self.fc(torch.stack(outputs, 1))
        return outputs, hidden, h_hat_t


class Seq2Seq(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.encoder = Encoder(cf)
        self.decoder = Decoder(cf)
        self.ntokens = cf.trg_params['vocab_size']

    def forward(self, src, src_lengths, trg):
        src = self.encoder(src, src_lengths)
        return self.decoder(src, trg)[0]

    def inference(self, src, src_lengths, sos, maxlen=30):
        src = self.encoder(src, src_lengths)
        x_t = np.full((src.size(0), 1), sos).astype('int64')
        x_t = to_var(x_t)

        h_t = None
        h_hat_t = None
        outputs = []
        for i in range(maxlen):
            probs, h_t, h_hat_t = self.decoder(src, x_t, h_t, h_hat_t)
            x_t = F.softmax(probs, -1).max(-1)[1]
            outputs.append(x_t)
        return torch.cat(outputs, 1)

    def score(self, src, src_lengths, trg):
        source = trg[:, :-1]
        target = trg[:, 1:]

        outputs = self.forward(src, src_lengths, source)
        loss = self.criterion(
            outputs.view(-1, self.ntokens),
            target.contiguous().view(-1)
        )
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
