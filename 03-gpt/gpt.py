import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SEED = 2147483647  # same seed for every random step, so the results are reproducible

with open('03-gpt/input.txt', 'r') as f:
    text = f.read().splitlines()

print(text)