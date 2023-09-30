import json
import pandas as pd
import sys
from numpy import dot
from numpy.linalg import norm


sys.path.append('/home/hd/FuzzingTransformerPrograms/src')
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
from src.utils.data_utils import calculate_similarity_prepare_dataset
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

def cosine_similarity(list1, list2):
    return dot(list1, list2) / (norm(list1) * norm(list2)+1e-8)

for strategy in ["sort", "reverse", "most_freq", "dyck1", "dyck2", "hist", "double_hist", "conll"]:
    try:
        with open(f'./output/{strategy}/program_prediction.json',"r") as f:
            program_result = json.load(f)
        with open(f'./output/{strategy}/prediction.json',"r") as f:
            model_result = json.load(f)
        import torch
        train = torch.load(f'./output/{strategy}/data.pt')["train"]
        program_result = pd.DataFrame(program_result)
        program_result.columns = ['sent', 'tags']
        (idx_w, w_idx, idx_t, t_idx, X_train, Y_train, X_test, Y_test) = calculate_similarity_prepare_dataset(train,program_result)
        total_similarity = 0
        for i in range(len(program_result)):
            if strategy == "sort":
                total_similarity += cosine_similarity(X_test[i][1:-1], model_result[i]["pred"][1:-1])
            elif strategy == "reverse":
                total_similarity += cosine_similarity(X_test[i][1:-1], model_result[i]["pred"][1:-1])
            elif strategy == "dyck1":
                total_similarity += cosine_similarity(X_test[i][1:], model_result[i]["pred"][1:])
            elif strategy == "dyck2":
                total_similarity += cosine_similarity(X_test[i][1:], model_result[i]["pred"][1:])
            elif strategy == "double_hist":
                total_similarity += cosine_similarity(X_test[i][1:], model_result[i]["pred"][1:])
            elif strategy == "conll":
                total_similarity += cosine_similarity(X_test[i][1:-1], model_result[i]["pred"][1:-1])
            elif strategy == "most_freq":
                total_similarity += cosine_similarity(X_test[i][1:], model_result[i]["pred"][1:])
            elif strategy == "hist":
                total_similarity += cosine_similarity(X_test[i][1:], model_result[i]["pred"][1:])
        total_similarity = total_similarity/len(program_result)
        print(f"{strategy} similarity: {total_similarity}")
        # break
    except:
        continue

    