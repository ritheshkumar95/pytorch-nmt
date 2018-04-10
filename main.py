# coding: utf-8
import time
from pathlib import Path
import os
import subprocess
import argparse
import numpy as np

import torch
import torch.nn as nn

from eval import MetricEvaluator
from data import DataLoader
from config import load_config
from model import Seq2Seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-s', '--save_path', default='./')
    parser.add_argument('-l', '--load_path', default=None)
    args = parser.parse_args()
    return args


args = parse_args()
cf = load_config(args.path)
if not Path(args.save_path).exists():
    os.makedirs(args.save_path)

torch.manual_seed(cf.seed)
torch.cuda.manual_seed(cf.seed)

loader = DataLoader(cf)
cf.src_params['vocab_size'] = loader.corpus.src_params['vocab_size']
cf.trg_params['vocab_size'] = loader.corpus.trg_params['vocab_size']

evaluator = MetricEvaluator(loader)

model = Seq2Seq(cf).cuda()
model.criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(model.parameters())

if args.load_path:
    f = os.path.join(args.load_path, 'model.pt')
    model.load_state_dict(torch.load(f))


def evaluate(split):
    start = time.time()
    print('=' * 89)
    print("Startin evaluation on {} set...".format(split))
    score, costs = evaluator.compute_scores(model, split, True)
    print('Validation completed! loss {:5.4f} | ppl {:8.4f} | BLEU {:5.4f}'.format(
            np.mean(costs), np.exp(np.mean(costs)), score)
          )
    print("Took {:5.4f}s to validate!".format(time.time() - start))
    print('=' * 89)
    return score


def train():
    model.train()
    itr = loader.create_epoch_iterator('train', cf.batch_size)

    costs = []
    start_time = time.time()
    for i, (src, src_lengths, trg) in enumerate(itr):
        loss = model.score(src, src_lengths, trg)
        costs.append(loss.data[0])

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % cf.log_interval == 0 and i > 0:
            cur_loss = np.asarray(costs)[-cf.log_interval:].mean()
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} completed | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i * cf.batch_size, len(loader.corpus.data['train']),
                      elapsed * 1000 / cf.log_interval, cur_loss, np.exp(cur_loss))
                  )
            start_time = time.time()

    return np.mean(costs)


best_val_score = -1
try:
    for epoch in range(1, cf.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_score = evaluate('val')
        if val_score > best_val_score:
            best_val_score = val_score
            f = os.path.join(args.save_path, 'model.pt')
            torch.save(model.state_dict(), f)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.save_path, 'model.pt'), 'rb') as f:
    model.load_state_dict(torch.load(f))

# Run on test data.
evaluate('test')
