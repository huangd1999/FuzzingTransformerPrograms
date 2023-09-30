import sys
sys.path.append('/home/hd/backup')
sys.path.append('/home/hd/backup/src')
import time
import os
import json
import numpy as np
import coverage
import trace
from output.most_freq import most_freq
from src.utils.data_utils import make_most_freq_fuzzer
print("start")

# /home/hd/miniconda3/lib/python3.11/trace.py
for step in [60,300,600,3600]:
    filename = "./result/most_freq/trace_test_cases_"+str(step)+".txt"

    filtered_test_cases = []

    start = time.time()

    maintainer = []

    i = -1
    while True:
        if time.time() - start > step:
            break
        i+=1
        total_coverage = coverage.Coverage(source=['output.most_freq'], data_file='./result/most_freq/.coverage.total_'+str(step))
        total_coverage.start()
        fuzzer_sample = make_most_freq_fuzzer(vocab_size=8, min_length=0, max_length=8)
        print(fuzzer_sample)
        result = most_freq.run(fuzzer_sample["inputs"])
        # 停止总体覆盖率收集并保存数据
        total_coverage.stop()
        total_coverage.save()
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 初始化trace追踪器
        tracer = trace.Trace(trace=1, count=True,path_covered='most_freq.py')

        # 启动trace追踪
        tracer.runfunc(most_freq.run, fuzzer_sample["inputs"])  # 使用most_freq.run函数并传入data["inputs"]参数

        # 再次运行相同的测试用例
        most_freq.run(fuzzer_sample["inputs"])
        sys.stdout = original_stdout
        # 停止trace追踪并获取执行行号
        result = tracer.obtain_trace_result()
        print(len(result))
        print(len(set(result)))
        # 停止总体覆盖率收集并保存数据
        total_coverage.stop()
        total_coverage.save()

        # 开始当前测试用例的覆盖率收集
        cov = coverage.Coverage(source=['output.most_freq'], data_file=f'.coverage.{i}')
        cov.start()

        # 再次运行相同的测试用例
        most_freq.run(fuzzer_sample["inputs"])

        # 停止当前测试用例的覆盖率收集并保存数据
        cov.stop()
        cov.save()
        cov.json_report(outfile=f'./result/most_freq/coverage_{i}.json')

        # 读取覆盖率报告数据
        with open(f'./result/most_freq/coverage_{i}.json') as f:
            data = json.load(f)
        os.remove(f'./result/most_freq/coverage_{i}.json')
        fuzzing_case = data['files']["output/most_freq/most_freq.py"]['executed_lines']
        num_statements = data['files']["output/most_freq/most_freq.py"]['summary']['num_statements']
        
        if i == 0:
            total_case = fuzzing_case
        else:
            with open(f'./result/most_freq/total_coverage_{i-1}.json') as f:
                data = json.load(f)
            os.remove(f'./result/most_freq/total_coverage_{i-1}.json')
            total_case = data['files']["output/most_freq/most_freq.py"]['executed_lines']
        # print(result)
        if result not in maintainer:
            maintainer.append(result)
            filtered_test_cases.append(fuzzer_sample)
            print(len(set(total_case + fuzzing_case)) / num_statements)
        
        # 合并总体覆盖率和当前测试用例的覆盖率
        total_coverage.combine([cov.config.data_file])
        total_coverage.save()

        # 保存总体覆盖率结果到JSON文件
        total_coverage.json_report(outfile=f'./result/most_freq/total_coverage_{i}.json')
        if len(set(total_case + fuzzing_case)) == num_statements:
            print("all")
            break

        # 再次启动总体覆盖率收集，准备下一轮迭代
        total_coverage.start()
        # break
    path = f'./result/most_freq/trace_test_cases_{step}.json'
    with open(path, 'w') as f:
        json.dump(filtered_test_cases, f, indent=4)


# 停止总体覆盖率收集（最终一次）
total_coverage.stop()
total_coverage.save()