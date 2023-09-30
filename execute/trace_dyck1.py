import sys
sys.path.append('/home/hd/backup')
sys.path.append('/home/hd/backup/src')
import time
import os
import json
import numpy as np
import coverage
import trace
from output.dyck1 import dyck1
from src.utils.data_utils import make_dyck_pft_fuzzer
print("start")

# /home/hd/miniconda3/lib/python3.11/trace.py
for step in [60,300,600,3600]:
    filename = "./result/dyck1/trace_test_cases_"+str(step)+".txt"

    filtered_test_cases = []

    start = time.time()

    maintainer = []

    i = -1
    while True:
        if time.time() - start > step:
            break
        i+=1
        total_coverage = coverage.Coverage(source=['output.dyck1'], data_file='./result/dyck1/.coverage.total_'+str(step))
        total_coverage.start()
        fuzzer_sample = make_dyck_pft_fuzzer(vocab_size=1, min_length=0, max_length=16)
        print(fuzzer_sample)
        result = dyck1.run(fuzzer_sample["inputs"])
        # 停止总体覆盖率收集并保存数据
        total_coverage.stop()
        total_coverage.save()
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 初始化trace追踪器
        tracer = trace.Trace(trace=1, count=True,path_covered='dyck1.py')

        # 启动trace追踪
        tracer.runfunc(dyck1.run, fuzzer_sample["inputs"])  # 使用dyck1.run函数并传入data["inputs"]参数

        # 再次运行相同的测试用例
        dyck1.run(fuzzer_sample["inputs"])
        sys.stdout = original_stdout
        # 停止trace追踪并获取执行行号
        result = tracer.obtain_trace_result()
        print(len(result))
        print(len(set(result)))
        # 停止总体覆盖率收集并保存数据
        total_coverage.stop()
        total_coverage.save()

        # 开始当前测试用例的覆盖率收集
        cov = coverage.Coverage(source=['output.dyck1'], data_file=f'.coverage.{i}')
        cov.start()

        # 再次运行相同的测试用例
        dyck1.run(fuzzer_sample["inputs"])

        # 停止当前测试用例的覆盖率收集并保存数据
        cov.stop()
        cov.save()
        cov.json_report(outfile=f'./result/dyck1/coverage_{i}.json')

        # 读取覆盖率报告数据
        with open(f'./result/dyck1/coverage_{i}.json') as f:
            data = json.load(f)
        os.remove(f'./result/dyck1/coverage_{i}.json')
        fuzzing_case = data['files']["output/dyck1/dyck1.py"]['executed_lines']
        num_statements = data['files']["output/dyck1/dyck1.py"]['summary']['num_statements']
        
        if i == 0:
            total_case = fuzzing_case
        else:
            with open(f'./result/dyck1/total_coverage_{i-1}.json') as f:
                data = json.load(f)
            os.remove(f'./result/dyck1/total_coverage_{i-1}.json')
            total_case = data['files']["output/dyck1/dyck1.py"]['executed_lines']
        # print(result)
        if result not in maintainer:
            maintainer.append(result)
            filtered_test_cases.append(fuzzer_sample)
            print(len(set(total_case + fuzzing_case)) / num_statements)
        
        # 合并总体覆盖率和当前测试用例的覆盖率
        total_coverage.combine([cov.config.data_file])
        total_coverage.save()

        # 保存总体覆盖率结果到JSON文件
        total_coverage.json_report(outfile=f'./result/dyck1/total_coverage_{i}.json')
        if len(set(total_case + fuzzing_case)) == num_statements:
            print("all")
            break

        # 再次启动总体覆盖率收集，准备下一轮迭代
        total_coverage.start()
        # break
    path = f'./result/dyck1/trace_test_cases_{step}.json'
    with open(path, 'w') as f:
        json.dump(filtered_test_cases, f, indent=4)


# 停止总体覆盖率收集（最终一次）
total_coverage.stop()
total_coverage.save()