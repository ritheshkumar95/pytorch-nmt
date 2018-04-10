import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def to_var(arr):
    return Variable(torch.from_numpy(arr)).cuda()


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, key, value):
        B, T, _ = key.size()
        scale = 1. / np.sqrt(key.size(-1))
        att_weights = key.bmm(value.transpose(1, 2)) * scale
        att_probs = F.softmax(att_weights, -1)
        att_outputs = att_probs.bmm(value)
        return att_outputs


class Encoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
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

        self.drop = nn.Dropout(cf.src_params['dropout'])

    def forward(self, src, lengths):
        src = self.drop(self.emb(src))
        src = pack_padded_sequence(src, lengths, batch_first=True)
        out, _ = self.rnn(src)
        out = pad_packed_sequence(out, batch_first=True)
        return self.drop(out[0])


class Decoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.emb = nn.Embedding(
            cf.trg_params['vocab_size'],
            cf.trg_params['emb_size']
        )

        self.rnn = nn.LSTM(
            cf.trg_params['emb_size'] + cf.trg_params['hidden_size'],
            cf.trg_params['hidden_size'],
            cf.trg_params['num_layers'],
            dropout=cf.trg_params['dropout'],
            batch_first=True
        )
        self.att = ScaledDotProductAttention()

        self.att_hidden = nn.Linear(
            cf.trg_params['hidden_size'] * 2,
            cf.trg_params['hidden_size']
        )
        self.fc = nn.Linear(
            cf.trg_params['hidden_size'],
            cf.trg_params['vocab_size']
        )

        self.drop = nn.Dropout(cf.trg_params['dropout'])
        self.input_feed_init = nn.Parameter(
            torch.randn(cf.trg_params['hidden_size'])
        )

    def forward(self, src, trg, hidden=None, h_hat_t=None):
        trg = self.drop(self.emb(trg))

        if h_hat_t is None:
            h_hat_t = self.input_feed_init.expand(src.size(0), 1, -1)

        outputs = []
        for x_t in trg.split(1, dim=1):
            h_t, hidden = self.rnn(
                torch.cat([x_t, h_hat_t], -1),
                hidden
            )
            h_t = self.drop(h_t)

            c_t = self.att(h_t, src)
            h_hat_t = F.tanh(self.att_hidden(
                torch.cat([h_t, c_t], -1)
            ))
            out = self.fc(h_hat_t)

            outputs.append(out)

        return torch.cat(outputs, 1), hidden, h_hat_t


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
