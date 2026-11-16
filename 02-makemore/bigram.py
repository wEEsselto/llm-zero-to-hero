import random

import matplotlib.pyplot as plt
import numpy as np
import torch

with open('02-makemore/names.txt', 'r') as f:
    names = f.read().splitlines()

#Dictornaries for char to int and int to char mapping
stoi = {}
itos = {}

chars = sorted(set(''.join(names)))

for i, char in enumerate(chars):
    stoi[char]= i+1
stoi['.'] = 0

itos = {i: s for s, i in stoi.items()}

#Create tensor
N = torch.zeros((27, 27), dtype=torch.int32)

for w in names:
    chs = '.' + w + '.'
    #Loop over Bigrams (= two consecutive characters) in dataset and store count in tensor
    for a, b in zip(chs, chs[1:]):
        N[stoi[a], stoi[b]] += 1

#Visualize N
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')

plt.axis('off')
plt.savefig('02-makemore/bigram_counts.png', bbox_inches='tight')

#Turn counts into probabilities
P=(N+1).float() #+1 so no zero entries("smoothing")
#Normalize each row to sum to 1
P = P / P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)


#Sample Names
for i in range (10):
    out = []
    ix=0
    while True:
        # Draw the next character's index according to the probability distribution in row P[ix]
        ix = torch.multinomial(P[ix], num_samples=1, replacement =True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    #print(''.join(out))

# Negative log likelihood loss: for every bigram that occurs in the data,
#   model gave it a high probability -> small penalty
#   model gave it a low probability  -> big penalty
# The loss is the average penalty over all bigrams (lower is better)
loss=0
n = 0 
for w in names:
    chs = '.' + w + '.'
    #Loop over Bigrams
    for a, b in zip(chs, chs[1:]):
        n += 1
        loss += np.log(P[stoi[a], stoi[b]].item())
loss = -loss / n 
print(loss)