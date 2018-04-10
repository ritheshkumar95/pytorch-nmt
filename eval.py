# coding: utf-8
import os
import argparse
import subprocess
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from config import load_config
from model import Seq2Seq
from beam import BeamSearch
from data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-l', '--load_path', default=None)
    parser.add_argument('-b', '--beam_search', action='store_true')
    parser.add_argument('-w', '--beam_width', type=int, default=4)
    args = parser.parse_args()
    return args


def compute_bleu(refs, hyps):
    open('ref.txt', 'w').write('\n'.join(refs))
    open('hyp.txt', 'w').write('\n'.join(hyps))
    command = '$MOSES_ROOT/scripts/generic/multi-bleu.perl -lc ref.txt < hyp.txt'
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    print(out)
    score = float(out.decode("utf-8").split(' ')[2].split(',')[0])
    return score


class MetricEvaluator(object):
    def __init__(self, loader, beam_search=False, beam_width=4, batch_size=64):
        self.batch_size = batch_size
        self.loader = loader

        # Dumping essential params
        self.word2idx = loader.corpus.trg_params['word2idx']
        self.idx2word = loader.corpus.trg_params['idx2word']
        self.sos = loader.corpus.trg_params['word2idx']['<s>']

        self.beam_search = None
        if beam_search:
            self.beam_search = BeamSearch(
                self.word2idx, beam_width=beam_width
            )

    def compute_scores(self, model, split, compute_ppl=False):
        itr = self.loader.create_epoch_iterator(split, self.batch_size)
        model.eval()

        refs = []
        hyps = []
        costs = []
        for i, (src, src_lengths, trg) in tqdm(enumerate(itr)):
            if compute_ppl:
                loss = model.score(src, src_lengths, trg)
                costs.append(loss.data[0])

            if self.beam_search is None:
                out = model.inference(
                    src, src_lengths,
                    sos=self.sos
                )
                out = out.cpu().data.tolist()
            else:
                src = model.encoder(src, src_lengths)
                out = self.beam_search.search(model.decoder, src)

            trg = trg.cpu().data.tolist()

            for ref, hyp in zip(trg, out):
                refs.append(self.loader.corpus.idx2sent(self.idx2word, ref))
                hyps.append(self.loader.corpus.idx2sent(self.idx2word, hyp))

        score = compute_bleu(refs, hyps)
        return score, costs


if __name__ == '__main__':
    args = parse_args()
    cf = load_config(args.path)

    loader = DataLoader(cf)
    cf.src_params['vocab_size'] = loader.corpus.src_params['vocab_size']
    cf.trg_params['vocab_size'] = loader.corpus.trg_params['vocab_size']

    model = Seq2Seq(cf).cuda()
    model.eval()
    model.criterion = nn.CrossEntropyLoss().cuda()

    if args.load_path:
        f = os.path.join(args.load_path, 'model.pt')
        model.load_state_dict(torch.load(f))

    evaluator = MetricEvaluator(
        loader, args.beam_search, args.beam_width
    )

    print('=' * 89)
    print("Startin evaluation on test set...")
    start = time.time()
    evaluator.compute_scores(model, 'test')
    print("Took {:5.4f}s to validate!".format(time.time() - start))
    print('=' * 89)
