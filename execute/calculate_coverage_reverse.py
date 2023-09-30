import sys
sys.path.append('/home/hd/TransformerPrograms')

import os
import json
import numpy as np
import coverage
from programs.rasp.reverse import reverse

examples = [
    ['<s>', 0, 0, '</s>'],
['<s>', '</s>'],
['<s>', 1, 4, 3, '</s>'],
['<s>', 3, 3, 0, 1, '</s>'],
['<s>', 3, '</s>'],
['<s>', 3, 3, 0, 1,1,4,3],
[1, 3, 3, 0, 1,1,4,3],
]

for example in examples:
    reverse.run(example)