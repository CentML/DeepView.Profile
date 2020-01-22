import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

torch.backends.cudnn.benchmark = True


class TransformerWithLoss(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def _cal_loss(self, pred, gold, smoothing):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(Constants.PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(
                pred, gold, ignore_index=Constants.PAD, reduction='sum')

        return loss

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, gold):
        out = self.transformer(src_seq, src_pos, tgt_seq, tgt_pos)
        return self._cal_loss(out, gold, smoothing=True)


def model_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument(
        '-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-trace_to', type=str)
    parser.add_argument('-seed', type=int)
    parser.add_argument('-load_traced', type=str)
    parser.add_argument('-measure', action='store_true')
    opt = parser.parse_args(args=[])

    opt.seed = 1
    opt.embs_share_weight = False
    opt.proj_share_weight = True
    opt.label_smoothing = True

    # Constants from the dataset
    opt.d_word_vec = opt.d_model
    opt.max_token_seq_len = 52
    opt.src_vocab_size = 32317
    opt.tgt_vocab_size = 32317

    opt.warm_up = 10
    opt.measure_for = 100
    return opt


def skyline_model_provider():
    opt = model_config()
    return TransformerWithLoss(Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)).cuda()


def skyline_input_provider(batch_size=64):
    vocab_size = 32000
    src_seq_len = 25
    tgt_seq_len = 25

    device = torch.device('cuda')

    source = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, src_seq_len),
        dtype=torch.int64,
        device=device,
    )
    target = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, tgt_seq_len),
        dtype=torch.int64,
        device=device,
    )

    source_pos_row = torch.arange(
        0, src_seq_len, dtype=torch.int64, device=device).unsqueeze_(0)
    target_pos_row = torch.arange(
        0, tgt_seq_len, dtype=torch.int64, device=device).unsqueeze_(0)

    source_pos_list = [source_pos_row for _ in range(batch_size)]
    target_pos_list = [target_pos_row for _ in range(batch_size)]

    src_pos = torch.cat(source_pos_list, 0)
    tgt_pos = torch.cat(target_pos_list, 0)

    gold = target[:, 1:]
    return source, src_pos, target, tgt_pos, gold


def skyline_iteration_provider(transformer):
    opt = model_config()
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    def iteration(src_seq, src_pos, tgt_seq, tgt_pos, gold):
        optimizer.zero_grad()
        loss = transformer(src_seq, src_pos, tgt_seq, tgt_pos, gold)
        loss.backward()
        optimizer.step_and_update_lr()

    return iteration
