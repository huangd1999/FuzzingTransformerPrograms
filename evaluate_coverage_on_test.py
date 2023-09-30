import torch
import time
import os
import json
import numpy as np
import coverage
import trace
from tqdm import tqdm
from output.hist import hist
from output.sort import sort
from output.reverse import reverse
from output.dyck1 import dyck1
from output.dyck2 import dyck2
from output.double_hist import double_hist
from output.conll import conll_ner
from output.most_freq import most_freq
import sys
import warnings
warnings.filterwarnings("ignore")

strategy_modules = {
    "hist": hist,
    "sort": sort,
    "reverse": reverse,
    "dyck1": dyck1,
    "dyck2": dyck2,
    "double_hist": double_hist,
    "conll": conll_ner,  # Assuming 'conll_reverse' was a typo and you meant 'conll'.
    "most_freq": most_freq
}

# for strategy in ["sort", "reverse", "dyck1", "dyck2", "hist", "double_hist", "conll", "most_freq"]:
for strategy in ["hist"]:
    print_text = ""
    cov = coverage.Coverage(source=[f'output.{strategy}'], data_file=f'./result/{strategy}/.coverage.')
    cov.start()
    data = torch.load(f'./output/{strategy}/data.pt')
    data = data["test"]
    correct = 0
    total = len(data)
    for i in tqdm(range(len(data))):
        try:
            pred = strategy_modules[strategy].run(data.iloc[i].to_numpy()[0])
        except:
            continue
    cov.stop()
    cov.save()
    cov.json_report(outfile=f'./result/{strategy}/coverage_{i}.json')

    # 读取覆盖率报告数据
    with open(f'./result/{strategy}/coverage_{i}.json') as f:
        data = json.load(f)
    print_text += "{:.2f}".format(data["totals"]["percent_covered"]) + "&"
    print(data["totals"]["percent_covered"])
    print(print_text)
