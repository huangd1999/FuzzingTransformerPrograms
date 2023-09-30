import sys
sys.path.append('/home/hd/backup')
sys.path.append('/home/hd/backup/src')
import time
import torch
import os
import json
import numpy as np
import coverage
import trace
from output.conll import conll_ner
from src.utils.data_utils import get_conll_ner_fuzzer
BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"

dataset = torch.load('./output/conll/data.pt')
# /home/hd/miniconda3/lib/python3.11/trace.py
for step in [60,300,600,3600]:
    filtered_test_cases = []

    start = time.time()

    maintainer = []

    i = 0
    while True:
        if time.time() - start > step:
            break
        try:
            total_coverage = coverage.Coverage(source=['output.conll'], data_file='./result/conll/.coverage.total_'+str(step))
            total_coverage.start()

            fuzzer_sample = {"inputs": [], "outputs": []}
            test_case_length = np.random.randint(0, 32)
            for _ in range(test_case_length):
                sentence_idx = np.random.randint(0, len(dataset["train"]))
                input_idx = np.random.randint(1, len(dataset["train"].iloc[sentence_idx].to_numpy()[0])-1)
                fuzzer_sample["inputs"].append(dataset["train"].iloc[sentence_idx].to_numpy()[0][input_idx])
                fuzzer_sample["outputs"].append(dataset["train"].iloc[sentence_idx].to_numpy()[1][input_idx])
            fuzzer_sample["inputs"] = [BOS] + fuzzer_sample["inputs"] + [EOS]
            fuzzer_sample["outputs"] = [PAD] + fuzzer_sample["outputs"] + [PAD]
            print(fuzzer_sample)
            result = conll_ner.run(fuzzer_sample["inputs"])
            # 停止总体覆盖率收集并保存数据
            total_coverage.stop()
            total_coverage.save()
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            # 初始化trace追踪器
            tracer = trace.Trace(trace=1, count=True,path_covered='conll_ner.py')

            # 启动trace追踪
            tracer.runfunc(conll_ner.run, fuzzer_sample["inputs"])  # 使用conll.run函数并传入data["inputs"]参数

            # 再次运行相同的测试用例
            conll_ner.run(fuzzer_sample["inputs"])
            sys.stdout = original_stdout
            # 停止trace追踪并获取执行行号
            result = tracer.obtain_trace_result()
            print(len(result))
            print(len(set(result)))
            # 停止总体覆盖率收集并保存数据
            total_coverage.stop()
            total_coverage.save()

            # 开始当前测试用例的覆盖率收集
            cov = coverage.Coverage(source=['output.conll'], data_file=f'.coverage.{i}')
            cov.start()

            # 再次运行相同的测试用例
            conll_ner.run(fuzzer_sample["inputs"])

            # 停止当前测试用例的覆盖率收集并保存数据
            cov.stop()
            cov.save()
            cov.json_report(outfile=f'./result/conll/coverage_{i}.json')

            # 读取覆盖率报告数据
            with open(f'./result/conll/coverage_{i}.json') as f:
                data = json.load(f)
            os.remove(f'./result/conll/coverage_{i}.json')
            fuzzing_case = data['files']["output/conll/conll_ner.py"]['executed_lines']
            num_statements = data['files']["output/conll/conll_ner.py"]['summary']['num_statements']
            
            if i == 0:
                total_case = fuzzing_case
            else:
                with open(f'./result/conll/total_coverage_{i-1}.json') as f:
                    data = json.load(f)
                os.remove(f'./result/conll/total_coverage_{i-1}.json')
                total_case = data['files']["output/conll/conll_ner.py"]['executed_lines']
            # print(result)
            if result not in maintainer:
                maintainer.append(result)
                filtered_test_cases.append(fuzzer_sample)
                print(len(set(total_case + fuzzing_case)) / num_statements)
            
            # 合并总体覆盖率和当前测试用例的覆盖率
            total_coverage.combine([cov.config.data_file])
            total_coverage.save()

            # 保存总体覆盖率结果到JSON文件
            total_coverage.json_report(outfile=f'./result/conll/total_coverage_{i}.json')
            if len(set(total_case + fuzzing_case)) == num_statements:
                print("all")
                break

            # 再次启动总体覆盖率收集，准备下一轮迭代
            total_coverage.start()
            i+=1
            # break
        except Exception as e:
            print(e)
    path = f'./result/conll/trace_test_cases_{step}.json'
    with open(path, 'w') as f:
        json.dump(filtered_test_cases, f, indent=4)


# 停止总体覆盖率收集（最终一次）
total_coverage.stop()
total_coverage.save()