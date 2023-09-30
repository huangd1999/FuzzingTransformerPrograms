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



def run_test(
    model,
    X,
    Y,
    batch_size=256,
    return_preds=False,
    x_pad_idx=0,
    y_pad_idx=0,
    autoregressive=False,
    func=torch.argmax,
    loss_agg="per_token",
    o_idx=None,
    idx_t=None,
    output_dir=None,
    
):
    dataloader = DataLoader(
        list(zip(X, Y)), batch_size=batch_size, shuffle=False
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
        preds.append(pred.detach().cpu().numpy())
        true.append(tgts.detach().cpu().numpy())
    preds = np.concatenate(preds, 0)
    true = np.concatenate(true, 0)
    save_result = []
    for pred, label in zip(preds, true):
        save_result.append({"pred": pred.tolist(), "label": label.tolist()})
    with open(f"./{output_dir}/prediction.json", "w") as f:
        json.dump(save_result, f, indent=4)
    m = true != y_pad_idx
    acc = (preds == true)[m].mean()
    metrics = {}
    # if o_idx is not None:
    y_true = [idx_t[y[y != y_pad_idx]].tolist() for y in true]
    y_pred = [
        idx_t[y_hat[y != y_pad_idx]].tolist()
        for y, y_hat in zip(true, preds)
    ]
    metrics = metric_utils.conll_score(y_true=y_true, y_pred=y_pred)
    loss = np.concatenate(out, 0).mean()
    if return_preds:
        return loss, acc, metrics, preds, true
    return loss, acc, metrics



## LSA
class LSA(object):
    def __init__(self,train,input,layers,std=0.05):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.lst=[]
        self.std_lst=[]
        self.mask=[]
        self.neuron_activate_train=[]
        index_lst=[]

        for index,l in layers:
            self.lst.append(Model(inputs=input,outputs=l))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(train).reshape(len(train),l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
            self.std_lst.append(np.std(temp,axis=0))
            self.mask.append((np.array(self.std_lst)>std))
        self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
        self.mask=np.concatenate(self.mask,axis=0)
        #self.lst=list(zip(index_lst,self.lst))

    def fit(self,test,use_lower=False):
        self.neuron_activate_test=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test=np.concatenate(self.neuron_activate_test,axis=1)
        test_score = []
        for test_sample in self.neuron_activate_test[:,self.mask]:
            test_mean = np.zeros_like(test_sample)
            for train_sample in self.neuron_activate_train[:,self.mask]:
                temp = test_sample-train_sample
                kde = stats.gaussian_kde(temp, bw_method='scott')
                test_mean+=kde.evaluate(temp)
            test_score.append(reduce(lambda x,y:np.log(x)+np.log(y),test_mean/len(self.neuron_activate_train)))
        return test_score

## DSA
class DSA(object):
    def __init__(self,train,label,input,layers,std=0.05):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.lst=[]
        self.std_lst=[]
        self.mask=[]
        self.neuron_activate_train=[]
        index_lst=[]

        for index,l in layers:
            self.lst.append(Model(inputs=input,outputs=l))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(train).reshape(len(train),l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
        self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
        self.train_label = np.array(label)

    def fit(self,test,label,use_lower=False):
        self.neuron_activate_test=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test=np.concatenate(self.neuron_activate_test,axis=1)
        test_score = []
        for test_sample,label_sample in zip(self.neuron_activate_test,label):
            dist_a = np.min(((self.neuron_activate_train[self.train_label == label_sample,:]-test_sample)**2).sum(axis=1))
            dist_b = np.min(((self.neuron_activate_train[self.train_label != label_sample,:]-test_sample)**2).sum(axis=1))
            test_score.append(dist_a/dist_b)
        return test_score



# CAM
# def compute_neuron_coverage(activations, threshold=0.5):
#     total_neurons = 0
#     activated_neurons = 0

#     for act in activations.values():
#         if isinstance(act, torch.Tensor):
#             total_neurons += act.numel()
#             activated_neurons += (act > threshold).sum().item()
#         elif isinstance(act, tuple):
#             for tensor in act:
#                 if isinstance(tensor, torch.Tensor):
#                     total_neurons += tensor.numel()
#                     activated_neurons += (tensor > threshold).sum().item()

#     return activated_neurons / total_neurons

# # 2. 修改hook_fn函数
# def hook_fn(module, input, output):
#     layer_name = str(module)
#     # 我们将直接存储激活值，而不是形状
#     layer_outputs[layer_name] = output


# 使用字典来存储每一层的当前输出
layer_outputs = {}

# CTM
# 1. 定义全局变量
cumulative_activations = {}

def hook_fn(module, input, output):
    layer_name = str(module)
    
    # 如果检测到形状变化
    if layer_name in cumulative_activations and cumulative_activations[layer_name].shape != output.shape:
        layer_name = f"{layer_name}_{output.shape}"
    
    # 更新layer_outputs
    if isinstance(output, torch.Tensor):
        layer_outputs[layer_name] = output
    elif isinstance(output, tuple):
        for i, tensor in enumerate(output):
            key = f"{layer_name}_{i}"
            layer_outputs[key] = tensor
    
    # 更新cumulative_activations
    if isinstance(output, torch.Tensor):
        if layer_name not in cumulative_activations:
            cumulative_activations[layer_name] = output > 0.25
        else:
            cumulative_activations[layer_name] |= output > 0.25


# 在compute_neuron_coverage中，更新如何计算activated_neurons
def compute_neuron_coverage(activations):
    total_neurons = 0
    activated_neurons = 0

    for act in activations.values():
        if isinstance(act, torch.Tensor):
            total_neurons += act.numel()
            activated_neurons += act.sum().item()
        elif isinstance(act, tuple):
            for tensor in act:
                if isinstance(tensor, torch.Tensor):
                    total_neurons += tensor.numel()
                    activated_neurons += tensor.sum().item()

    # 添加一个检查以确保total_neurons不为0
    if total_neurons == 0:
        raise ValueError("No neurons were found in the activations. Please check the model and hooks.")

    return activated_neurons / total_neurons

def register_hooks(model):
    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn)

def json_serializable(item):
    if isinstance(item, torch.Tensor):
        return item.cpu().tolist()  # 将张量转换为列表
    raise TypeError(f"Type {type(item)} not serializable")



def coverage_fuzzing(
    model,
    X,
    Y,
    batch_size=1,
    return_preds=False,
    x_pad_idx=0,
    y_pad_idx=0,
    autoregressive=False,
    func=torch.argmax,
    loss_agg="per_token",
    o_idx=None,
    idx_t=None,
    output_dir=None,
    args=None,
    dataset=None,
):
    dataset = dataset
    dataloader = DataLoader(
        list(zip(X, Y)), batch_size=batch_size, shuffle=False
    )
    out = []
    preds = []
    true = []
    model.eval()
    coverage_metric = None
    
    # 为模型的每一层注册钩子
    import time
    fuzzing_times = [60,300,600,3600]
    for step in fuzzing_times:
        start = time.time()
        pre_coverage=0
        save_result = []
        register_hooks(model)
        cumulative_activations.clear()
        for i, (x, y) in tqdm(enumerate(dataloader)):
            x = x.to(model.device)
            m = (x != x_pad_idx).float()
            mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
            if autoregressive:
                mask = torch.tril(mask)
            
            with torch.no_grad():
                log_probs = model(x, mask=mask).log_softmax(-1)
            
            coverage = compute_neuron_coverage(cumulative_activations)
            if pre_coverage < coverage:
                pre_coverage = coverage
                save_result.append({"encode_x":x, "decode_y":y, "inputs": dataset.iloc[i]["sent"], "outputs": dataset.iloc[i]["tags"], "coverage":coverage})
            if coverage == 1.0:
                cumulative_activations.clear()
                print("coverage 1.0")
            if time.time() - start > step:
                print(time.time() - start)
                break
        print(len(save_result))
        # 使用这个函数来保存结果
        with open(f'./{args.output_dir}/nac_{step}_0.25.json', 'w') as f:
            json.dump(save_result, f, indent=4, default=json_serializable)




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

    a = set(["".join(s) for s in train["sent"]])
    b = set(["".join(s) for s in test["sent"]])

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
    model.load_state_dict(torch.load(args.pretrain_path))

    for split, X, Y,train in [
        ("test", X_train, Y_train, train),
    ]:
        coverage_fuzzing(
            model,
            X,
            Y,
            return_preds=True,
            x_pad_idx=w_idx["<pad>"],
            y_pad_idx=t_idx["<pad>"],
            autoregressive=args.autoregressive,
            loss_agg=args.loss_agg,
            o_idx=t_idx.get("O", None),
            idx_t=idx_t,
            output_dir=args.output_dir,
            args=args,
            dataset=train,
        )


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {vars(args)}")
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f)
    run(args)
