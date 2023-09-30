import sys
sys.path.append('/home/hd/TransformerPrograms')

import os
import json
import numpy as np
import coverage
from output.sort import sort

# # Set seed for reproducibility
# np.random.seed(0)

# # Ensure the directory exists
# os.makedirs('./result/sort/', exist_ok=True)

# vocab = np.array([str(i) for i in range(100 - 3)])

# # Start total coverage collection
# total_coverage = coverage.Coverage(source=['output.sort'], data_file='.coverage.total')

# total_coverage.start()

# for i in range(10):
#     # Generate a random test case
#     l = np.random.randint(1, 20 - 1)
#     sent = np.random.choice(vocab, size=l, replace=True).tolist()
#     sent = ["<s>"] + sent + ["</s>"]
#     print(sent)
#     print(sort.run(sent))

#     # Stop total coverage collection and save data
#     total_coverage.stop()
#     total_coverage.save()

#     # Start coverage collection for current test case
#     cov = coverage.Coverage(source=['output.sort'], data_file=f'.coverage.{i}')
#     cov.start()

#     # Run the same test case again
#     print(sort.run(sent))

#     # Stop coverage collection for current test case and save data
#     cov.stop()
#     cov.save()

#     # Combine total coverage with current test case coverage
#     total_coverage.combine([cov.config.data_file])
#     total_coverage.save()

#     # Save the results to a JSON file
#     cov.json_report(outfile=f'./result/sort/coverage_{i}.json')
#     total_coverage.json_report(outfile=f'./result/sort/total_coverage_{i}.json')

#     # Start total coverage collection again for the next iteration
#     total_coverage.start()
#     total_coverage.combine([f'.coverage.{i}' for i in range(10)])
#     # Load the JSON report for current test case
#     with open(f'./result/sort/coverage_{i}.json') as f:
#         data = json.load(f)

#     # Calculate the total lines, covered lines, and coverage percent for current test case
#     total_lines = 0
#     covered_lines = 0

#     for file_path, file_data in data['files'].items():
#         total_lines += len(file_data['executed_lines']) + len(file_data['missing_lines'])
#         covered_lines += len(file_data['executed_lines'])

#     coverage_percent = (covered_lines / total_lines) * 100

#     # Load the JSON report for total coverage
#     with open(f'./result/sort/total_coverage_{i}.json') as f:
#         total_data = json.load(f)

#     # Calculate the total lines, covered lines, and coverage percent for total coverage
#     total_total_lines = 0
#     total_covered_lines = 0

#     for file_path, file_data in total_data['files'].items():
#         total_total_lines += len(file_data['executed_lines']) + len(file_data['missing_lines'])
#         total_covered_lines += len(file_data['executed_lines'])

#     total_coverage_percent = (total_covered_lines / total_total_lines) * 100

#     print(f"Test Case {i+1}")
#     print(f"Coverage for current test case:")
#     print(f"Total Lines: {total_lines}")
#     print(f"Covered Lines: {covered_lines}")
#     print(f"Coverage Percent: {coverage_percent}")
#     print(f"Total coverage up to this test case:")
#     print(f"Total Lines: {total_total_lines}")
#     print(f"Covered Lines: {total_covered_lines}")
#     print(f"Coverage Percent: {total_coverage_percent}")

# # Stop total coverage collection for the final time
# total_coverage.stop()
# total_coverage.save()



import numpy as np
import json
import coverage
import torch
vocab = list(range(97))  # example vocab

total_coverage = coverage.Coverage(source=['output.sort'], data_file='.coverage.total')
total_coverage.start()

dataset = torch.load('./output/sort/data.pt')

length = len(dataset["train"])
for i in range(length):
    temp = dataset["train"].iloc[i].tolist()
    temp = ["<s>"] + temp + ["</s>"] 
    print(temp)
    result = sort.run(sent)
    for n in range(1, len(result)-1):
        result[n] = int(result[n])

    # Stop total coverage collection and save data
    total_coverage.stop()
    total_coverage.save()

    # Start coverage collection for current test case
    cov = coverage.Coverage(source=['output.sort'], data_file=f'.coverage.{i}')
    cov.start()

    # Run the same test case again
    sort.run(sent)

    # Stop coverage collection for current test case and save data
    cov.stop()
    cov.save()
    cov.json_report(outfile=f'./result/sort/coverage_{i}.json')
    # Combine total coverage with current test case coverage
    total_coverage.combine([cov.config.data_file])
    total_coverage.save()

    # Save the results to a JSON file
    total_coverage.json_report(outfile=f'./result/sort/total_coverage_{i}.json')

    # Start total coverage collection again for the next iteration
    total_coverage.start()

# Stop total coverage collection for the final time
total_coverage.stop()
total_coverage.save()
