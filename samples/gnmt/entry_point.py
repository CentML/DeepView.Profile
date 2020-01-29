import argparse
from ast import literal_eval

import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist

import seq2seq.data.config as config
import seq2seq.utils as utils
from seq2seq.models.gnmt import GNMT
from seq2seq.train.fp_optimizers import Fp32Optimizer
from seq2seq.train.lr_scheduler import WarmupMultiStepLR
from seq2seq.train.smoothing import LabelSmoothing

torch.backends.cudnn.benchmark = True


def get_args():
    def exclusive_group(group, name, default, help):
        destname = name.replace('-', '_')
        subgroup = group.add_mutually_exclusive_group(required=False)
        subgroup.add_argument(f'--{name}', dest=f'{destname}',
                              action='store_true',
                              help=f'{help} (use \'--no-{name}\' to disable)')
        subgroup.add_argument(f'--no-{name}', dest=f'{destname}',
                              action='store_false', help=argparse.SUPPRESS)
        subgroup.set_defaults(**{destname: default})

    parser = argparse.ArgumentParser(
        description='GNMT training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en',
                         help='path to the directory with training/test data')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it will be \
                         automatically created if it does not exist')
    results.add_argument('--save', default='gnmt',
                         help='defines subdirectory within RESULTS_DIR for \
                         results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--hidden-size', default=1024, type=int,
                       help='model hidden size')
    model.add_argument('--num-layers', default=4, type=int,
                       help='number of RNN layers in encoder and in decoder')
    model.add_argument('--dropout', default=0.2, type=float,
                       help='dropout applied to input of RNN cells')

    exclusive_group(group=model, name='share-embedding', default=True,
                    help='use shared embeddings for encoder and decoder')

    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                       CrossEntropyLoss, if not zero model will be trained \
                       with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp32', choices=['fp16', 'fp32'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='master seed for random number generators, if \
                         "seed" is undefined then the master seed will be \
                         sampled from random.SystemRandom()')

    exclusive_group(group=general, name='eval', default=True,
                    help='run validation and test after every epoch')
    exclusive_group(group=general, name='env', default=False,
                    help='print info about execution env')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--train-batch-size', default=128, type=int,
                          help='training batch size per worker')
    training.add_argument('--train-global-batch-size', default=None, type=int,
                          help='global training batch size, this argument \
                          does not have to be defined, if it is defined it \
                          will be used to automatically \
                          compute train_iter_size \
                          using the equation: train_iter_size = \
                          train_global_batch_size // (train_batch_size * \
                          world_size)')
    training.add_argument('--train-iter-size', metavar='N', default=1,
                          type=int,
                          help='training iter size, training loop will \
                          accumulate gradients over N iterations and execute \
                          optimizer every N steps')
    training.add_argument('--epochs', default=8, type=int,
                          help='max number of training epochs')

    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enables gradient clipping and sets maximum \
                          norm of gradients')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--train-loader-workers', default=2, type=int,
                          help='number of workers for training data loading')
    training.add_argument('--batching', default='bucketing', type=str,
                          choices=['random', 'sharding', 'bucketing'],
                          help='select batching algorithm')
    training.add_argument('--shard-size', default=80, type=int,
                          help='shard size for "sharding" batching algorithm, \
                          in multiples of global batch size')
    training.add_argument('--num-buckets', default=5, type=int,
                          help='number of buckets for "bucketing" batching \
                          algorithm')

    # optimizer
    optimizer = parser.add_argument_group('optimizer setup')
    optimizer.add_argument('--optimizer', type=str, default='Adam',
                           help='training optimizer')
    optimizer.add_argument('--lr', type=float, default=1.00e-3,
                           help='learning rate')

    # scheduler
    scheduler = parser.add_argument_group('learning rate scheduler setup')
    scheduler.add_argument('--warmup-steps', type=str, default='200',
                           help='number of learning rate warmup iterations')
    scheduler.add_argument('--remain-steps', type=str, default='0.666',
                           help='starting iteration for learning rate decay')
    scheduler.add_argument('--decay-interval', type=str, default='None',
                           help='interval between learning rate decay steps')
    scheduler.add_argument('--decay-steps', type=int, default=4,
                           help='max number of learning rate decay steps')
    scheduler.add_argument('--decay-factor', type=float, default=0.5,
                           help='learning rate decay factor')

    # validation
    val = parser.add_argument_group('validation setup')
    val.add_argument('--val-batch-size', default=64, type=int,
                     help='batch size for validation')
    val.add_argument('--max-length-val', default=125, type=int,
                     help='maximum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--min-length-val', default=0, type=int,
                     help='minimum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--val-loader-workers', default=0, type=int,
                     help='number of workers for validation data loading')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--max-length-test', default=150, type=int,
                      help='maximum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--min-length-test', default=0, type=int,
                      help='minimum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--intra-epoch-eval', metavar='N', default=0, type=int,
                      help='evaluate within training epoch, this option will \
                      enable extra N equally spaced evaluations executed \
                      during each training epoch')
    test.add_argument('--test-loader-workers', default=0, type=int,
                      help='number of workers for test data loading')

    # checkpointing
    chkpt = parser.add_argument_group('checkpointing setup')
    chkpt.add_argument('--start-epoch', default=0, type=int,
                       help='manually set initial epoch counter')
    chkpt.add_argument('--resume', default=None, type=str, metavar='PATH',
                       help='resumes training from checkpoint from PATH')
    chkpt.add_argument('--save-all', action='store_true', default=False,
                       help='saves checkpoint after every epoch')
    chkpt.add_argument('--save-freq', default=5000, type=int,
                       help='save checkpoint every SAVE_FREQ batches')
    chkpt.add_argument('--keep-checkpoints', default=0, type=int,
                       help='keep only last KEEP_CHECKPOINTS checkpoints, \
                       affects only checkpoints controlled by --save-freq \
                       option')

    # benchmarking
    benchmark = parser.add_argument_group('benchmark setup')
    benchmark.add_argument('--target-bleu', default=24.0, type=float,
                           help='target accuracy, training will be stopped \
                           when the target is achieved')

    # distributed
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='global rank of the process, do not set!')
    distributed.add_argument('--local_rank', default=0, type=int,
                             help='local rank of the process, do not set!')

    args = parser.parse_args([])

    args.warmup_steps = literal_eval(args.warmup_steps)
    args.remain_steps = literal_eval(args.remain_steps)
    args.decay_interval = literal_eval(args.decay_interval)

    return args


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)

    return criterion


class GNMTWithLoss(nn.Module):
    def __init__(self, gnmt, loss_fn):
        super().__init__()
        self.gnmt = gnmt
        self.loss_fn = loss_fn

    def forward(self, src, src_len, tgt, tgt_len):
        out = self.gnmt(src, src_len, tgt[:-1])
        T, B = out.size(0), out.size(1)
        tgt_labels = tgt[1:]
        loss = self.loss_fn(
            out.view(T * B, -1),
            tgt_labels.contiguous().view(-1))
        return loss / B


def skyline_model_provider():
    args = get_args()
    vocab_size = 32317
    model_config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'batch_first': False,
        'share_embedding': args.share_embedding,
    }
    model = GNMTWithLoss(
        GNMT(vocab_size=vocab_size, **model_config),
        build_criterion(vocab_size, config.PAD, args.smoothing),
    ).cuda()
    model.zero_grad()
    return model


def skyline_input_provider(batch_size=64):
    vocab_size = 32000
    src_len = 25
    tgt_len = 25

    device = torch.device('cuda')

    src = torch.randint(
        low=0,
        high=vocab_size,
        size=(src_len, batch_size),
        dtype=torch.int64,
        device=device,
    )
    tgt = torch.randint(
        low=0,
        high=vocab_size,
        size=(tgt_len, batch_size),
        dtype=torch.int64,
        device=device,
    )

    src_len_tensor = torch.tensor(
        [src_len] * batch_size, dtype=torch.int64, device=device)
    tgt_len_tensor = torch.tensor(
        [tgt_len] * batch_size, dtype=torch.int64, device=device)

    return src, src_len_tensor, tgt, tgt_len_tensor


def skyline_iteration_provider(model):
    args = get_args()
    opt_config = {
        'optimizer': args.optimizer,
        'lr': args.lr,
    }
    scheduler_config = {
        'warmup_steps': args.warmup_steps,
        'remain_steps': args.remain_steps,
        'decay_interval': args.decay_interval,
        'decay_steps': args.decay_steps,
        'decay_factor': args.decay_factor,
    }

    train_loader_len = 437268
    total_train_iters = train_loader_len // args.train_iter_size * args.epochs
    opt_name = opt_config.pop('optimizer')
    optimizer = torch.optim.__dict__[opt_name](model.parameters(), **opt_config)
    scheduler = WarmupMultiStepLR(
        optimizer, total_train_iters, **scheduler_config)
    fp_optimizer = Fp32Optimizer(model, args.grad_clip)

    def iteration(src, src_len, tgt, tgt_len):
        loss = model(src, src_len, tgt, tgt_len)
        loss.backward()
        fp_optimizer.step(optimizer, scheduler, update=True)

    return iteration
