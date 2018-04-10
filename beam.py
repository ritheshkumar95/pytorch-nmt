import heapq
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def to_numpy(var):
    return var.cpu().data.numpy()


def to_list(var):
    return var.cpu().data.tolist()


def to_var(arr):
    return Variable(torch.from_numpy(arr)).cuda()


class Beam(object):
    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, prefix, hidden, complete):
        heapq.heappush(self.heap, (prob, prefix, hidden, complete))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


class BeamSearch(object):
    def __init__(self, word2idx, beam_width=4, batch_size=256):
        self.beam_width = beam_width
        self.sos = word2idx['<s>']
        self.eos = word2idx['</s>']
        self.batch_size = batch_size

    def search(self, model, src):
        n_beams = src.size(0)
        # Create all initial beams
        prev_beams = [Beam(self.beam_width) for i in range(n_beams)]

        # Seed the beam with sos
        for i, beam in enumerate(prev_beams):
            prev_beams[i].add(
                1.,
                [self.sos],
                (None, None, None),
                False
            )

        # Repeat until all beams terminate
        while True:
            # Create the next iteration of beams
            new_beams = [Beam(self.beam_width) for i in range(n_beams)]

            # Flattening all prefixes, states, scores from all beams to do
            # Batched Beam Search. Pointer is used to invert flattening.
            prefix_flat = []
            hidden_flat = []
            scores_flat = []
            pointers = []
            for i, beam in enumerate(prev_beams):
                scores, prefix, hidden, is_complete = zip(*beam.heap)

                # Only add elements that are still incomplete
                idxs = np.where(np.asarray(is_complete) == False)[0]
                prefix_flat += [prefix[idx] for idx in idxs]
                scores_flat += [scores[idx] for idx in idxs]
                hidden_flat += [hidden[idx] for idx in idxs]
                # Update the pointer list
                pointers += [i] * len(idxs)

                # Add completed elements directly to the new beam
                idxs = np.where(np.asarray(is_complete) == True)[0]
                for idx in idxs:
                    new_beams[i].add(
                        scores[idx],
                        prefix[idx],
                        hidden[idx],
                        True
                    )

            # Termination condition!
            if len(prefix_flat) == 0:
                break

            for i in range(0, len(prefix_flat), self.batch_size):
                batch_pointers = pointers[i: i + self.batch_size]
                batch_scores = scores_flat[i: i + self.batch_size]
                batch_prefix = prefix_flat[i: i + self.batch_size]

                x_t = [x[-1] for x in batch_prefix]
                batch_x_t = to_var(
                    np.asarray(x_t).astype('int64')
                )[:, None]

                h, c, h_hat = zip(*hidden_flat[i: i + self.batch_size])
                try:
                    batch_hidden = (torch.stack(h, 1), torch.stack(c, 1))
                    batch_h_hat = torch.stack(h_hat, 0)
                except AttributeError:
                    batch_hidden = None
                    batch_h_hat = None

                batch_src = src[batch_pointers]

                output, hidden, h_hat = model(
                    batch_src,
                    batch_x_t,
                    batch_hidden,
                    batch_h_hat
                )
                scores, next_words = F.softmax(output[:, 0], 1).topk(self.beam_width)

                scores = to_numpy(scores)
                next_words = to_numpy(next_words)

                # Update the appropriate beam
                for i, ptr in enumerate(batch_pointers):
                    for j in range(self.beam_width):
                        new_beams[ptr].add(
                            batch_scores[i] * scores[i, j],
                            batch_prefix[i] + [next_words[i, j]],
                            (hidden[0][:, i], hidden[1][:, i], h_hat[i]),
                            next_words[i, j] == self.eos
                        )

            del prev_beams
            prev_beams = new_beams

        return [max(beam.heap)[1] for beam in prev_beams]
