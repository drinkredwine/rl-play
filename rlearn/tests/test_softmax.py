from collections import defaultdict
from pprint import pprint

import numpy as np


def test_softmax():
    results = defaultdict(int)
    for _ in range(10000):
        choice = np.random.choice([1, 2, 3], p=[0.1, 0.3, 0.6])
        results[choice] += 1

    pprint(results)