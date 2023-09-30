import sys
sys.path.append('/home/hd/FuzzingTransformerPrograms')
sys.path.append('/home/hd/FuzzingTransformerPrograms/src')
import time
import os
import json
import numpy as np
import coverage
import trace
from output.hist import hist
from output.sort import sort
from output.reverse import reverse
from output.dyck1 import dyck1
from output.dyck2 import dyck2
from output.double_hist import double_hist
from output.conll import conll_ner
from output.most_freq import most_freq
from src.utils.data_utils import make_hist_fuzzer
print("start")

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

for strategy in ["sort"]:
    print_text = ""
    for step in [60,300,600,3600]:
        with open(f"./result/{strategy}/trace_test_cases_{step}.json", "r") as f:
            data = json.load(f)
            
        correct = 0
        total = len(data)
        for i in range(len(data)):
            try:
                pred = strategy_modules[strategy].run(data[i]["inputs"])
                if strategy == "sort" and pred[1:-1] == data[i]["outputs"][1:-1]:
                    correct += 1
                elif strategy == "reverse" and pred[1:-1] == data[i]["outputs"][1:-1]:
                    correct += 1
                elif strategy == "dyck1" and pred[1:] == data[i]["outputs"][1:]:
                    correct += 1
                elif strategy == "dyck2" and pred[1:] == data[i]["outputs"][1:]:
                    correct += 1
                elif strategy == "double_hist" and pred[1:] == data[i]["outputs"][1:]:
                    correct += 1
                elif strategy == "conll" and pred[1:-1] == data[i]["outputs"][1:-1]:
                    correct += 1
                elif strategy == "most_freq" and pred[1:] == data[i]["outputs"][1:]:
                    correct += 1
                elif strategy == "hist" and pred[1:] == data[i]["outputs"][1:]:
                    correct += 1
            except:
                correct+=1
        print_text += f"{total}&{total-correct}&"
        print(f"evaluation: {strategy}, step: {step}, correct: {correct}, total: {total}, accuracy: {correct/total}, error: {total-correct}, error_rate: {100 * (total-correct)/total:.2f}%")
        # print(f"evaluation: {strategy}, step: {step}, correct: {correct}, total: {total}, accuracy: {correct/total}")
    print(print_text)
