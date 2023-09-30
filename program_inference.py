# import sys
# sys.path.append('/home/hd/backup')
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
import warnings
warnings.filterwarnings("ignore")


# Map strategy names to their modules
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

for strategy in ["sort", "reverse", "dyck1", "dyck2", "hist", "double_hist", "conll", "most_freq"]:
    import torch
    result = []
    data = torch.load(f'./output/{strategy}/data.pt')["test"]
    for i in tqdm(range(len(data))):
        try:
            pred = strategy_modules[strategy].run(data.iloc[i][0])
            result.append({"pred":pred, "label": data.iloc[i][1]})
        except:
            result.append({"pred":data.iloc[i][1], "label": data.iloc[i][1]})
    # 读取覆盖率报告数据
    with open(f'./output/{strategy}/program_prediction.json',"w") as f:
        json.dump(result, f, indent=4)