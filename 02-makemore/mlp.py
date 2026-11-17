import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

with open('02-makemore/names.txt', 'r') as f:
    names = f.read().splitlines()

#Dictionaries for char to int and int to char mapping
chars = sorted(set(''.join(names)))

stoi = {}
for i, char in enumerate(chars):
    stoi[char] = i+1
stoi['.'] = 0  #special token marking start and end of a name
itos = {i: s for s, i in stoi.items()}

#build dataset
context_length = 3
def build_dataset(names):
    X = []
    Y = []

    for name in names:
        context = [0]*context_length
        for ch in name + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append (= sliding window)
            
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

random.seed(42)
random.shuffle(names)
n1 = int(0.8*len(names))
n2 = int(0.9*len(names))

Xtr, Ytr = build_dataset(names[:n1]) #80% Training
Xdev, Ydev = build_dataset(names[n1:n2]) # 10% Validation
Xte, Yte = build_dataset(names[n2:]) #10% Test

print(Xtr.size())