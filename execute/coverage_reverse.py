import sys
sys.path.append('/home/hd/TransformerPrograms')

import os
import json
import numpy as np
import coverage
from output.reverse import reverse


vocab = list(range(97))  # example vocab

total_coverage = coverage.Coverage(source=['output.reverse'], data_file='.coverage.total')
total_coverage.start()

for i in range(50):
    # Generate a random test case
    l = np.random.randint(1, 20 - 1)
    # sent = np.random.choice(vocab, size=l, replace=True).tolist()
    temp = np.random.choice(vocab, size=l, replace=True)
    sent = temp.tolist()

    sent = ["<s>"] + sent + ["</s>"]
    # temp = ["<s>"] + temp + ["</s>"]
    print(sent)
    result = reverse.run(sent)
    print(result)
    # for i in range(1, len(result)-1):
    #     result[i] = int(result[i])
    # print(result)

    # Stop total coverage collection and save data
    total_coverage.stop()
    total_coverage.save()

    # Start coverage collection for current test case
    cov = coverage.Coverage(source=['output.reverse'], data_file=f'.coverage.{i}')
    cov.start()

    # Run the same test case again
    reverse.run(sent)

    # Stop coverage collection for current test case and save data
    cov.stop()
    cov.save()
    cov.json_report(outfile=f'./result/reverse/coverage_{i}.json')
    # Combine total coverage with current test case coverage
    total_coverage.combine([cov.config.data_file])
    total_coverage.save()

    # Save the results to a JSON file
    total_coverage.json_report(outfile=f'./result/reverse/total_coverage_{i}.json')

    # Start total coverage collection again for the next iteration
    total_coverage.start()

# Stop total coverage collection for the final time
total_coverage.stop()
total_coverage.save()
