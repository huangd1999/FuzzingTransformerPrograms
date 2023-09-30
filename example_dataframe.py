import pandas as pd
import torch
import json
from src.utils import data_utils

data = torch.load(f'./output/sort/data.pt')
BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"

prepare_dataset(
                data['train'],
                test,
                val=val,
                vocab_size=vocab_size,
                unk=unk,
            ),

# with open(f'./result/sort/trace_test_cases_60.json') as f:
#     coverage_data = json.load(f)

# print(coverage_data[:5])


# # 将coverage_data转换为DataFrame
# coverage_df = pd.DataFrame(coverage_data)
# coverage_df.columns = ['sent', 'tags']

# # 将原始数据和coverage_df拼接起来
# combined_data = pd.concat([data, coverage_df], ignore_index=True)
