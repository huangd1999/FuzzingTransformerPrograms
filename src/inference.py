import argparse
import copy
from copy import deepcopy
from functools import partial
import itertools
import json
import math
import os
from pathlib import Path
import random

import einops
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformers import Transformer
from models.programs import (
    TransformerProgramModel,
    argmax,
    gumbel_hard,
    gumbel_soft,
    softmax,
)
from utils import code_utils, data_utils, logging, metric_utils
from utils.data_utils import prepare_dataset

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch")

    # Data
    parser.add_argument("--dataset", type=str, default="reverse")
    parser.add_argument("--vocab_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do_lower", type=int, default=0)
    parser.add_argument("--unique", type=int, default=1)
    parser.add_argument("--replace_numbers", type=int, default=0)

    # Model
    parser.add_argument("--n_vars_cat", type=int, default=1)
    parser.add_argument("--n_vars_num", type=int, default=1)
    parser.add_argument("--d_var", type=int, default=None)
    parser.add_argument("--n_heads_cat", type=int, default=2)
    parser.add_argument("--n_heads_num", type=int, default=2)
    parser.add_argument("--d_mlp", type=int, default=64)
    parser.add_argument("--n_cat_mlps", type=int, default=1)
    parser.add_argument("--n_num_mlps", type=int, default=1)
    parser.add_argument("--mlp_vars_in", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--sample_fn", type=str, default="gumbel_soft")
    parser.add_argument("--one_hot_embed", action="store_true")
    parser.add_argument("--count_only", action="store_true")
    parser.add_argument("--selector_width", type=int, default=0)
    parser.add_argument("--attention_type", type=str, default="cat")
    parser.add_argument("--rel_pos_bias", type=str, default="fixed")
    parser.add_argument("--mlp_type", type=str, default="cat")
    parser.add_argument("--autoregressive", action="store_true")

    parser.add_argument(
        "--glove_embeddings", type=str, default="data/glove.840B.300d.txt"
    )
    parser.add_argument("--do_glove", type=int, default=0)

    parser.add_argument("--unembed_mask", type=int, default=1)
    parser.add_argument("--pool_outputs", type=int, default=0)

    # Standard model
    parser.add_argument("--standard", action="store_true")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_head", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--gumbel_samples", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tau_init", type=float, default=3.0)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--pretrain_strategy", type=str, default="finetune")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--tau_end", type=float, default=0.01)
    parser.add_argument("--tau_schedule", type=str, default="geomspace")
    parser.add_argument("--loss_agg", type=str, default="per_token")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_code", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.dataset != "conll_ner":
        args.pretrain_path = f"output/{args.dataset}/model.pt"
        args.dataset_path = f"output/{args.dataset}/data.pt"
    else:
        args.pretrain_path = f"output/conll/model.pt"
        args.dataset_path = f"output/conll/data.pt"
    if "dyck1" in args.dataset:
        args.autoregressive = True
        args.vocab_size = 1
    if "dyck2" in args.dataset:
        args.autoregressive = True
        args.vocab_size = 2

    logging.initialize(args.output_dir)

    if args.standard and args.d_head is None:
        args.d_head = int(args.d_model // args.n_heads)
        logger.info(
            f"setting d_head to {args.d_model} // {args.n_heads} = {args.d_head}"
        )

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def get_sample_fn(name):
    d = {
        "softmax": softmax,
        "gumbel_hard": gumbel_hard,
        "gumbel_soft": gumbel_soft,
    }
    if name not in d:
        raise NotImplementedError(name)
    return d[name]




def run(args):
    set_seed(args.seed)
    dataset = torch.load(args.dataset_path)
    train = dataset["train"]
    test = dataset["test"]
    val = dataset["val"]
    idx_w = dataset["idx_w"]
    w_idx = dataset["w_idx"]
    idx_t = dataset["idx_t"]
    t_idx = dataset["t_idx"]
    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]
    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]
    X_val = dataset["X_val"]
    Y_val = dataset["Y_val"]
    steps = [60,300,600,3600]
    print_information = ""
    for step in steps:
        with open(f"./output/sort/nac_{step}.json", "r") as f:
                dataset = json.load(f)
        
        for i in range(len(dataset)):
            dataset[i] = dataset[i]["test_case"]
        dataset = pd.DataFrame(dataset)
        dataset.columns = ['sent', 'tags']
        train = dataset.copy()
        test = dataset.copy()
        (idx_w, w_idx, idx_t, t_idx, X_train, Y_train, X_test, Y_test) = prepare_dataset(train,test)

        a = set(["".join(str(i) for i in s) for s in train["sent"]])
        b = set(["".join(str(i) for i in s) for s in test["sent"]])

        if args.d_var is None:
            d = max(len(idx_w), X_train.shape[-1])
        else:
            d = args.d_var
        init_emb = None
        if args.glove_embeddings and args.do_glove:
            emb = data_utils.get_glove_embeddings(
                idx_w, args.glove_embeddings, dim=args.n_vars_cat * d
            )
            init_emb = torch.tensor(emb, dtype=torch.float32).T

        unembed_mask = None
        if args.unembed_mask:
            unembed_mask = np.array([t in ("<unk>", "<pad>") for t in idx_t])

        set_seed(args.seed)
        model = TransformerProgramModel(
            d_vocab=len(idx_w),
            d_vocab_out=len(idx_t),
            n_vars_cat=args.n_vars_cat,
            n_vars_num=args.n_vars_num,
            d_var=d,
            n_heads_cat=args.n_heads_cat,
            n_heads_num=args.n_heads_num,
            d_mlp=args.d_mlp,
            n_cat_mlps=args.n_cat_mlps,
            n_num_mlps=args.n_num_mlps,
            mlp_vars_in=args.mlp_vars_in,
            n_layers=args.n_layers,
            n_ctx=X_train.shape[1],
            sample_fn=get_sample_fn(args.sample_fn),
            init_emb=init_emb,
            attention_type=args.attention_type,
            rel_pos_bias=args.rel_pos_bias,
            unembed_mask=unembed_mask,
            pool_outputs=args.pool_outputs,
            one_hot_embed=args.one_hot_embed,
            count_only=args.count_only,
            selector_width=args.selector_width,
        ).to(torch.device(args.device))
        print(args.pretrain_path)
        model.load_state_dict(torch.load(args.pretrain_path))
        error = 0
        return_preds=True
        x_pad_idx=w_idx["<pad>"]
        y_pad_idx=t_idx["<pad>"]
        autoregressive=args.autoregressive
        loss_agg=args.loss_agg
        o_idx=t_idx.get("O", None)
        idx_t=idx_t
        func=torch.argmax
        for split, X, Y in [
            ("test", X_test, Y_test),
        ]:
            dataloader = DataLoader(
                list(zip(X, Y)), batch_size=1, shuffle=False
            )
            out = []
            preds = []
            true = []
            model.eval()
            for x, y in dataloader:
                x = x.to(model.device)
                m = (x != x_pad_idx).float()
                mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
                if autoregressive:
                    mask = torch.tril(mask)
                with torch.no_grad():
                    log_probs = model(x, mask=mask).log_softmax(-1)
                tgts = y.to(model.device)
                if loss_agg == "per_seq":
                    losses = -log_probs.gather(2, tgts.unsqueeze(-1))
                    losses = losses.masked_fill(
                        (tgts == y_pad_idx).unsqueeze(-1), 0.0
                    ).sum(-1)
                else:
                    all_losses = -log_probs.gather(2, tgts.unsqueeze(-1)).squeeze(-1)
                    masked_losses = all_losses.masked_fill((tgts == y_pad_idx), 0.0)
                    lengths = (tgts != y_pad_idx).sum(-1)
                    losses = masked_losses.sum(-1) / lengths
                out.append(losses.detach().cpu().numpy())
                pred = func(log_probs, -1)
                pred[pred == 5] = 0
                if torch.any(pred != tgts):
                    error+=1
        print_information += f"{error}&"
    print(print_information)

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {vars(args)}")
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f)
    run(args)
