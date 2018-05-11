import time
from collections import Counter
from pathlib import Path
import numpy as np

import torch
from torch.autograd import Variable


def to_var(arr):
    return Variable(torch.from_numpy(arr)).cuda()


class Corpus(object):
    def __init__(self, cf):
        self.base_path = Path(cf.dataset_path)

        self.src_params = cf.src_params
        self.trg_params = cf.trg_params

        dump_file = self.base_path / \
            "processed_{}-{}_data.npy".format(
                self.src_params['lang'], self.trg_params['lang']
            )
        if dump_file.exists():
            data = np.load(dump_file)
            self.src_params = data[0]
            self.trg_params = data[1]
            self.data = data[2]
        else:
            self.process_data()

    def sent2idx(self, word2idx, sent):
        unk = word2idx['<unk>']
        sos = word2idx['<s>']
        eos = word2idx['</s>']
        return [sos] + [word2idx.get(x, unk) for x in sent] + [eos]

    def idx2sent(self, idx2word, sent):
        sent = ' '.join([idx2word[x] for x in sent])
        rem_sos = sent.replace('<s> ', '')
        rem_eos = rem_sos.split(' </s>')[0]
        return rem_eos

    def read_file(self, path):
        data = open(path).read().splitlines()
        return [x.lower().split() for x in data]

    def process_data(self):
        start = time.time()

        src_data = {}
        trg_data = {}

        for split in ['train', 'val', 'test']:
            src_file = self.base_path / (split + '.' + self.src_params['lang'])
            trg_file = self.base_path / (split + '.' + self.trg_params['lang'])
            src_data[split] = self.read_file(src_file)
            trg_data[split] = self.read_file(trg_file)

        src_vocab = Counter()
        trg_vocab = Counter()

        _ = [src_vocab.update(line) for line in src_data['train']]
        _ = [trg_vocab.update(line) for line in trg_data['train']]

        src_vocab_reduced = list(zip(
            *src_vocab.most_common(self.src_params['vocab_size'])
        )) 
        trg_vocab_reduced = list(zip(
            *trg_vocab.most_common(self.trg_params['vocab_size'])
        ))

        essentials = ('<s>', '</s>', '<unk>', '<pad>')
        self.src_params['idx2word'] = essentials + src_vocab_reduced[0][:-4]
        self.trg_params['idx2word'] = essentials + trg_vocab_reduced[0][:-4]

        # Setting the vocab size, in-case it's dynamic
        self.src_params['vocab_size'] = len(self.src_params['idx2word'])
        self.trg_params['vocab_size'] = len(self.trg_params['idx2word'])

        self.src_params['word2idx'] = {
            x: i for i, x in enumerate(self.src_params['idx2word'])
        }
        self.trg_params['word2idx'] = {
            x: i for i, x in enumerate(self.trg_params['idx2word'])
        }

        print(
            "Thresholding source vocab at: ({}/ {}) Last word: ({}, {})".format(
                len(self.src_params['idx2word']),
                len(src_vocab),
                src_vocab_reduced[0][-1],
                src_vocab_reduced[1][-1]
            )
        )
        print(
            "Thresholding target vocab at: ({}/ {}) Last word: ({}, {})".format(
                len(self.trg_params['idx2word']),
                len(trg_vocab),
                trg_vocab_reduced[0][-1],
                trg_vocab_reduced[1][-1]
            )
        )

        src_dict = self.src_params['word2idx']
        trg_dict = self.trg_params['word2idx']

        self.data = {}
        for split in ['train', 'val', 'test']:
            src_data[split] = [self.sent2idx(src_dict, sent)
                               for sent in src_data[split]]
            trg_data[split] = [self.sent2idx(trg_dict, sent)
                               for sent in trg_data[split]]
            self.data[split] = list(zip(src_data[split], trg_data[split]))

        np.random.seed(111)
        np.random.shuffle(self.data['train'])

        print("Finished processing data in {:5.4f}s".format(time.time() - start))
        dump_file = self.base_path / \
            "processed_{}-{}_data.npy".format(
                self.src_params['lang'], self.trg_params['lang']
            )
        np.save(dump_file, (self.src_params, self.trg_params, self.data))


class DataLoader(object):
    def __init__(self, cf):
        self.corpus = Corpus(cf)

    def sort_data(self, src, trg):
        lengths = np.asarray([len(x) for x in src])
        idxs = np.argsort(-lengths)
        sorted_src = np.asarray(src, dtype=object)[idxs]
        sorted_trg = np.asarray(trg, dtype=object)[idxs]
        return sorted_src.tolist(), sorted_trg.tolist()

    def pad_and_mask(self, arr, pad_idx):
        lengths = [len(x) for x in arr]
        maxlen = max(lengths)
        bsz = len(arr)

        padded_arr = np.full((bsz, maxlen), pad_idx, dtype='int64')
        mask = np.zeros((bsz, maxlen), dtype='float32')
        for i in range(bsz):
            padded_arr[i, :lengths[i]] = arr[i]
            mask[i, :lengths[i]] = 1.
        return padded_arr, lengths, mask

    def create_epoch_iterator(self, which_set, batch_size=64):
        data = self.corpus.data[which_set]

        src_pad = self.corpus.src_params['word2idx']['<pad>']
        trg_pad = self.corpus.trg_params['word2idx']['<pad>']

        for i in range(0, len(data), batch_size):
            src, trg = zip(*data[i: i + batch_size])
            src, trg = self.sort_data(src, trg)

            src, src_lengths, _ = self.pad_and_mask(src, src_pad)
            trg, _, mask = self.pad_and_mask(trg, trg_pad)

            yield to_var(src), src_lengths, to_var(trg), to_var(mask)


if __name__ == '__main__':
    from config import load_config
    cf = load_config('config/baseline.py')
    loader = DataLoader(cf)

    src_dict = loader.corpus.src_params['idx2word']
    trg_dict = loader.corpus.trg_params['idx2word']

    itr = loader.create_epoch_iterator('train', 64)

    times = []
    for i in range(1000):
        start = time.time()
        try:
            src, _, trg, mask = itr.__next__()
        except StopIteration:
            itr = loader.create_epoch_iterator('train', 64)
            continue
        times.append(time.time()-start)
    print("Latency of data loader: ", np.mean(times))

    for i in range(64):
        print(loader.corpus.idx2sent(src_dict, src[i].data))
        print(loader.corpus.idx2sent(trg_dict, trg[i].data))
        print(mask[i].data)
        input()
