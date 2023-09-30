import sys
sys.path.append('/home/hd/TransformerPrograms')
import time
import os
import json
import numpy as np
import coverage
from programs.rasp.reverse import reverse

vocab = list(range(4))  # example vocab

total_coverage = coverage.Coverage(source=['programs.rasp.reverse'], data_file='.coverage.total')
total_coverage.start()

filtered_test_cases = []

start = time.time()

i=-1
while True:
    if time.time()-start>360:
        break
    i+=1
    # Generate a random test case
    l = np.random.randint(1, 8 - 1)
    # sent = np.random.choice(vocab, size=l, replace=True).tolist()
    temp = np.random.choice(vocab, size=l, replace=True)
    sent = temp.tolist()
    # print(temp.sort())
    # temp = np.sort(temp).tolist()
    sent = ["<s>"] + sent + ["</s>"]
    # temp = ["<s>"] + temp + ["</s>"]
    print(sent)
    result = reverse.run(sent)
    for n in range(1, len(result)-1):
        result[n] = int(result[n])
    print(result)
    # print(temp)

    # Stop total coverage collection and save data
    total_coverage.stop()
    total_coverage.save()

    # Start coverage collection for current test case
    cov = coverage.Coverage(source=['programs.rasp.reverse'], data_file=f'.coverage.{i}')
    cov.start()

    # Run the same test case again
    reverse.run(sent)

    # Stop coverage collection for current test case and save data
    cov.stop()
    cov.save()
    cov.json_report(outfile=f'./result/reverse/coverage_{i}.json')

    with open(f'./result/reverse/coverage_{i}.json') as f:
        data = json.load(f)
    fuzzing_case = data['files']["programs/rasp/reverse/reverse.py"]['executed_lines']
    num_statements = data['files']["programs/rasp/reverse/reverse.py"]['summary']['num_statements']
    if i==0:
        total_case = fuzzing_case
    else:
        with open(f'./result/reverse/total_coverage_{i-1}.json') as f:
            data = json.load(f)
        total_case = data['files']["programs/rasp/reverse/reverse.py"]['executed_lines']
    if set(total_case) != set(total_case+fuzzing_case):
        print(len(set(total_case+fuzzing_case))/num_statements)
        filtered_test_cases.append(sent)
    # Calculate the coverage difference between current test case and total coverage

    # Combine total coverage with current test case coverage
    total_coverage.combine([cov.config.data_file])
    total_coverage.save()

    # Save the results to a JSON file
    total_coverage.json_report(outfile=f'./result/reverse/total_coverage_{i}.json')
    if len(set(total_case+fuzzing_case))==num_statements:
        break

    # Start total coverage collection again for the next iteration
    total_coverage.start()

# Stop total coverage collection for the final time
total_coverage.stop()
total_coverage.save()

# Save the filtered test cases to a file
with open('./result/reverse/filtered_test_cases.txt', 'w') as f:
    for test_case in filtered_test_cases:
        f.write(''.join(str(test_case)) + '\n')