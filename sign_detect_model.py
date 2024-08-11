import sys
import inspect
import os
import os.path as osp
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Callback
import logging
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import pandas as pd
import numpy as np
from more_itertools import flatten
from functools import partial
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.nn import functional as F
import re 
import pickle

def parse_ckpt_path(s):
    pattern = r".*epoch=(\d+)-step=(\d+).ckpt"
    match = re.search(pattern, s)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None

def mkdir(path) : 
    if osp.exists(path) : 
        return 
    try : 
        os.mkdir(path) 
    except FileNotFoundError : 
        parentPath, _ = osp.split(path)
        mkdir(parentPath)
        os.mkdir(path)

def listdir (path) :
    """
    Convenience function to get 
    full path details while calling os.listdir

    Also ensures that the order is always the same.

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    paths = [osp.join(path, f) for f in os.listdir(path)]
    paths.sort()
    return paths

def load_model (expt_log_dir) : 
    ckpt_dir = osp.join(expt_log_dir, 'checkpoints')
    ckpt_paths = listdir(ckpt_dir)
    if any('last.ckpt' in _ for _ in ckpt_paths) : 
        # check for last.ckpt
        ckpt_path = [_ for _ in ckpt_paths if 'last.ckpt' in _][0]
    else  :
        # find the checkpoint with maximum steps 
        valid_paths = [_ for _ in ckpt_paths if parse_ckpt_path(_) is not None]
        ckpt_path = sorted(valid_paths, key=lambda x: parse_ckpt_path(x)[1])[-1]

    print('Loading from checkpoint ...', ckpt_path)

    # load model args
    args_dict = osp.join(expt_log_dir, 'args.pkl') 
    with open(args_dict, 'rb') as fp :
        args = DictWrapper(pickle.load(fp))

    # make model
    model = SignerDetector.load_from_checkpoint(ckpt_path, args=args)
    return model

def tensorApply (thing, fn, 
    predicate=lambda x: True, module=torch) : 
    """ 
    Apply a tensor transformation to a tensor 
    or dictionary or a list.
    """
    Cls = torch.Tensor if module == torch else np.ndarray
    if isinstance(thing, Cls) and predicate(thing): 
        thing = fn(thing)
    elif isinstance(thing, dict) : 
        for k, v in thing.items() : 
            thing[k] = tensorApply(v, fn, predicate, module)
    elif isinstance(thing, list) : 
        for i, _ in enumerate(thing) : 
            thing[i] = tensorApply(thing[i], fn, 
                    predicate, module)
    return thing


def add_prefix_to_path (full_path, prefix) : 
    base, rest = osp.split(full_path)
    return osp.join(base, prefix + rest)

def aggregateDict (listOfDicts, reducer, keys=None, default='') : 
    """ 
    Very handy function to combine a list of dicts
    into a dict with the reducer applied by key.
    """
    def reducerWithDefault (lst) : 
        try : 
            return reducer(lst)
        except Exception : 
            return lst
    if not isinstance(listOfDicts, list) :
        listOfDicts = list(listOfDicts)
    if keys is None : 
        keys = list(set(flatten(map(deepKeys, listOfDicts))))
    aggregator = lambda key : reducerWithDefault(
        list(map(
            partial(deepGet, deepKey=key, default=default), 
            listOfDicts
        ))
    )
    return deepDict(zip(keys, map(aggregator, keys)))

def dictmap (f, d) : 
    new = dict()
    for k, v in d.items() : 
        new[k] = f(k, v)
    return new

def deepKeys (dictionary) : 
    """ iterate over keys of dict of dicts """
    stack = [((), dictionary)]
    while len(stack) > 0 : 
        prevKeys, dictionary = stack.pop()
        for k, v in dictionary.items() : 
            if isinstance(v, dict) :
                stack.append(((*prevKeys, k), v))
            else : 
                yield (*prevKeys, k)

def deepGet (dictionary, deepKey, default) : 
    """ get key in a dict of dicts """
    v = dictionary.get(deepKey[0], default)
    if isinstance(v, dict) and len(deepKey) > 1: 
        return deepGet(v, deepKey[1:], default)
    else : 
        return v

def deepDict (pairs) : 
    """ 
    Create a deep dict a.k.a a dict of dicts
    where a key may be tuple
    """
    d = dict()
    for k, v in pairs : 
        d_ = d
        for k_ in k[:-1] : 
            if k_ not in d_ : 
                d_[k_] = dict()
            d_ = d_[k_]
        d_[k[-1]] = v
    return d

def getAll(thing, key) :
    """
    Traverse a dict or list of dicts
    in preorder and yield all the values
    for given key
    """
    if isinstance(thing, dict) : 
        if key in thing : 
            yield thing[key]
        for val in thing.values() :
            yield from getAll(val, key)
    elif isinstance(thing, list) : 
        for val in thing : 
            yield from getAll(val, key)

def pad_to_len (x, length) :
    x = x[:length]
    L = x.shape[0]
    pad = (*((0,) * (2 * len(x.shape) - 1)), length - L)
    x = torch.nn.functional.pad(x, pad, value=-1)
    return x

class SignerDetector(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        model_args = dict(
            n_layer=args.n_layer, 
            n_head=args.n_head, 
            n_embd=args.n_embd, 
            in_dim=args.in_dim,
            block_size=args.seq_len, 
            bias=False, 
            vocab_size=None, 
            dropout=0.2 if args.use_dropout else 0
        ) 
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)
        self.args = args

    def forward(self, batch, stage='train'):
        prefix = '' if stage == 'train' else 'val_'
        pose_seq = batch['pose_seq']
        B, L, *_ = pose_seq.shape
        pose_seq = pose_seq.reshape(B, L, -1)
        logits, _ = self.model(pose_seq) 
        if stage == 'train' : 
            probs = batch['probs'] 
            loss = F.cross_entropy(logits, probs)
        else :
            label = torch.tensor(batch['label']).to(self.device)
            loss = F.cross_entropy(logits, label)
        return {f'{prefix}loss': loss, 'logits': logits}

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        for k in outputs.keys() : 
            if 'loss' in k: 
                self.log(k, outputs[k], batch_size=self.args.batch_size)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, stage='val')
        for k in outputs.keys() : 
            if 'loss' in k: 
                self.log(k, outputs[k], batch_size=self.args.batch_size)
        return outputs

    def trainable_parameters (self) : 
        return []

    def count_trainable_parameters(self) : 
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available 
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

def dict_as_table_str(d):
    # Determine the maximum width of the keys and values for alignment
    max_key_width = max(len(str(key)) for key in d.keys())
    max_value_width = max(len(str(value)) for value in d.values())

    # Print the table header
    ges = '' 
    res += f"{'Key':<{max_key_width}} | {'Value':<{max_value_width}}\n"
    res += ('-' * (max_key_width + max_value_width + 3)) + "\n"
    for key, value in d.items():
        res += f"{str(key):<{max_key_width}} | {str(value):<{max_value_width}}\n"
    return res

class DictWrapper:
    def __init__(self, d):
        self._dict = d

    def __getattr__(self, name):
        if name in self._dict:
            return self._dict[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name == '_dict':
            super().__setattr__(name, value)
        else:
            self._dict[name] = value

    def __delattr__(self, name):
        if name in self._dict:
            del self._dict[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __contains__ (self, name) : 
        return name in self._dict

    def __repr__ (self) : 
        return dict_as_table_str(self._dict)

def get_parser () : 
    import isl_utils as iu
    parser = argparse.ArgumentParser(description='Signer detector')
    # model
    parser.add_argument('--n_layer', type=int, default=4, help='n layer for baby gpt')
    parser.add_argument('--n_head', type=int, default=4, help='n head for baby gpt')
    parser.add_argument('--n_embd', type=int, default=128, help='n embd for baby gpt')
    parser.add_argument('--in_dim', type=int, default=133 * 2, help='input dimension for baby gpt')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
    parser.add_argument('--beta2', type=float, default=0.95, help='beta 2')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--n_more_steps', type=int, default=1000000, help='number of more steps to train for')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate gradient over')
    parser.add_argument('--ckpt_path', type=str, default=None, help='where to resume training from')
    parser.add_argument('--seq_len', type=int, default=60, help='pose sequence length')
    parser.add_argument('--feature_size', type=int, default=150, help='the input feature size (66 for only pose), (150 for pose + hands)')
    parser.add_argument('--embed_dim', type=int, default=1408, help='embedding dimension for the encoder (pose)')
    parser.add_argument('--use_dropout', action='store_true', help='whether to use dropout or not')
    parser.add_argument('--frequency', type=int, default=20000, help='Plotting frequency')
    parser.add_argument('--dataframe', type=str, default=osp.join(iu.METADATA_DIR, 'unlabelled_track_splits_1M.csv'), help='Path to dataframe with unlabelled hashes') 
    parser.add_argument('--labels', type=str, default='../tracked_splits_probs_train_1M.npy', help='Path to numpy labels from snorkel label model')
    parser.add_argument('--dev_dataframe', type=str, default=osp.join(iu.METADATA_DIR, 'tracked_splits_dev_set.csv'), help='Path to dev dataframe with labelled hashes') 
    parser.add_argument('--mode', type=str, default='probabilistic', choices=['probabilistic', 'onehot'], help='Whether to infer labels argument as probs or labels') 
    return parser

def print_dict_as_table(d):
    # Determine the maximum width of the keys and values for alignment
    max_key_width = max(len(str(key)) for key in d.keys())
    max_value_width = max(len(str(value)) for value in d.values())

    # Print the table header
    print(f"{'Key':<{max_key_width}} | {'Value':<{max_value_width}}")
    print('-' * (max_key_width + max_value_width + 3))

    # Print the key-value pairs
    for key, value in d.items():
        print(f"{str(key):<{max_key_width}} | {str(value):<{max_value_width}}")
