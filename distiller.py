# coding: utf-8
import time
from pathlib import Path
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import MetricEvaluator
from data import DataLoader
from config import load_config
from model import BiSeq2Seq, Seq2Seq


def kl_div(log_p, log_q):
    return torch.exp(log_p) * (0 * log_p - log_q)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', default=None)
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-s', '--save_path', default='./')
    parser.add_argument('-l', '--load_path', default=None)
    args = parser.parse_args()
    return args


args = parse_args()
cf = load_config(args.path, 'student')
cf_teacher = load_config(
    os.path.join(args.teacher_path, 'birnn.py'),
    'teacher'
)

if not Path(args.save_path).exists():
    os.makedirs(args.save_path)

torch.manual_seed(cf.seed)
torch.cuda.manual_seed(cf.seed)

loader = DataLoader(cf)
cf.src_params['vocab_size'] = loader.corpus.src_params['vocab_size']
cf.trg_params['vocab_size'] = loader.corpus.trg_params['vocab_size']
cf_teacher.src_params['vocab_size'] = loader.corpus.src_params['vocab_size']
cf_teacher.trg_params['vocab_size'] = loader.corpus.trg_params['vocab_size']

evaluator = MetricEvaluator(loader)

bidir_model = BiSeq2Seq(cf_teacher).cuda()
bidir_model.criterion = nn.CrossEntropyLoss().cuda()
bidir_model.load_state_dict(torch.load(
    os.path.join(args.teacher_path, 'model.pt')
))

fwd_model = Seq2Seq(cf).cuda()
fwd_model.criterion = nn.CrossEntropyLoss(reduce=False).cuda()

optimizer = torch.optim.Adam(fwd_model.parameters())

if args.load_path:
    f = os.path.join(args.load_path, 'model.pt')
    fwd_model.load_state_dict(torch.load(f))


def evaluate(split):
    start = time.time()
    print('=' * 89)
    print("Startin evaluation on {} set...".format(split))
    score, costs = evaluator.compute_scores(fwd_model, split, True)
    print('Evaluation completed ({})! loss {:5.4f} | ppl {:8.4f} | BLEU {:5.4f}'.format(
        split, np.mean(costs), np.exp(np.mean(costs)), score)
    )
    print("Took {:5.4f}s to validate!".format(time.time() - start))
    print('=' * 89)
    return score


def train():
    fwd_model.train()
    itr = loader.create_epoch_iterator('train', cf.batch_size)

    costs = []
    start_time = time.time()
    for i, (src, src_lengths, trg, trg_lengths, mask) in enumerate(itr):
        bidir_logits = bidir_model.score(
            src, src_lengths, trg, trg_lengths, mask
        )[1]
        loss, fwd_logits = fwd_model.score(
            src, src_lengths, trg, mask
        )

        fwd_log_p = F.log_softmax(fwd_logits, -1)
        bidir_log_p = F.log_softmax(bidir_logits, -1)

        mask = mask[:, 1:].contiguous()
        kl_cost = kl_div(
            bidir_log_p.detach(),
            fwd_log_p
        )
        kl_cost = (kl_cost * mask.unsqueeze(-1)).sum() / mask.sum()

        costs.append([loss.item(), kl_cost.item()])

        optimizer.zero_grad()
        (loss + kl_cost).backward()
        optimizer.step()

        if i % cf.log_interval == 0 and i > 0:
            cur_loss = np.asarray(costs)[-cf.log_interval:].mean(0)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} completed | ms/batch {:5.2f} | '
                  'loss {} | ppl {}'.format(
                      epoch, i *
                      cf.batch_size, len(loader.corpus.data['train']),
                      elapsed * 1000 / cf.log_interval, cur_loss, np.exp(cur_loss))
                  )
            start_time = time.time()

    return np.mean(costs)


best_val_loss = 999
try:
    for epoch in range(1, cf.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_score = evaluate('val')
        if val_score > best_val_score:
            print("Saving model!")
            best_val_loss = val_loss
            f = os.path.join(args.save_path, 'model.pt')
            torch.save(fwd_model.state_dict(), f)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.save_path, 'model.pt'), 'rb') as f:
    fwd_model.load_state_dict(torch.load(f))

# Run on val data.
evaluate('val')
# Run on test data.
evaluate('test')
