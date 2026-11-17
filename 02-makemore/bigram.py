import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SEED = 2147483647  # same seed for every sampling run, so the results are comparable

with open('02-makemore/names.txt', 'r') as f:
    names = f.read().splitlines()

#Dictionaries for char to int and int to char mapping
chars = sorted(set(''.join(names)))

stoi = {}
for i, char in enumerate(chars):
    stoi[char] = i+1
stoi['.'] = 0  #special token marking start and end of a name

itos = {i: s for s, i in stoi.items()}


# =====================================================================
# COUNTING MODEL
# =====================================================================

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
P = (N+1).float()  #+1 so no zero entries ("smoothing")
#Normalize each row to sum to 1
P = P / P.sum(1, keepdim=True)

# Negative log likelihood loss: for every bigram that occurs in the data,
#   model gave it a high probability -> small penalty
#   model gave it a low probability  -> big penalty
# The loss is the average penalty over all bigrams (lower is better)
log_likelihood = 0
n = 0
for w in names:
    chs = '.' + w + '.'
    #Loop over Bigrams
    for a, b in zip(chs, chs[1:]):
        n += 1
        log_likelihood += torch.log(P[stoi[a], stoi[b]]).item()
loss_count = -log_likelihood / n

#Baseline: every character equally likely -> loss = -log(1/27)
loss_uniform = -math.log(1/27)


# =====================================================================
# 1-LAYER NEURAL NET
# =====================================================================

#Build training set
xs = []
ys = []
for w in names:
    chs = '.' + w + '.'
    #Loop over Bigrams
    for a, b in zip(chs, chs[1:]):
        xs.append(stoi[a])
        ys.append(stoi[b])
xs = torch.tensor(xs)
ys = torch.tensor(ys)

#One-hot encode inputs
xenc = F.one_hot(xs, num_classes=27).float()

#TRAINING LOOP
g = torch.Generator().manual_seed(SEED)
W = torch.randn((27, 27), generator=g, requires_grad=True)  #Weights
lr = 50  #Learning rate (= step size of gradient descent)
losses = []
for k in range(500):
    #FORWARD PASS
    logits = xenc @ W  # @ = mat-mul
    probs = torch.exp(logits)  # makes everything positive
    probs = probs / probs.sum(1, keepdim=True)  # normalize rows
    # Negative log likelihood loss: for every bigram that occurs in the data,
    #   model gave it a high probability -> small penalty
    #   model gave it a low probability  -> big penalty
    # Plus regularization: penalizes large weights, pulls the model toward
    # uniform probabilities
    loss = - torch.log(probs[torch.arange(probs.shape[0]), ys]).mean() + 0.01 * (W**2).mean()
    #BACKWARD PASS
    W.grad = None
    loss.backward()
    #UPDATE
    W.data += -lr * W.grad  #Gradient Descent
    losses.append(loss.item())

loss_nn = losses[-1]


# =====================================================================
# RESULTS: LOSSES AND SAMPLED NAMES
# =====================================================================

print('\nLOSS (negative log likelihood, lower is better)')
print(f'  uniform baseline:  {loss_uniform:.4f}')
print(f'  counting model:    {loss_count:.4f}')
print(f'  neural net:        {loss_nn:.4f}  (after {len(losses)} iterations, incl. regularization)')

#Sample names: uniform baseline
print('\n10 names from the uniform baseline:')
g = torch.Generator().manual_seed(SEED)
p_uniform = torch.ones(27) / 27
for i in range(10):
    out = []
    ix = 0
    while True:
        # Draw the next character's index, every character equally likely
        ix = torch.multinomial(p_uniform, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print('  ' + ''.join(out))

#Sample names: counting model
print('\n10 names from the counting model:')
g = torch.Generator().manual_seed(SEED)
for i in range(10):
    out = []
    ix = 0
    while True:
        # Draw the next character's index according to the probability distribution in row P[ix]
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print('  ' + ''.join(out))

#Sample names: neural net
print('\n10 names from the neural net:')
g = torch.Generator().manual_seed(SEED)
for i in range(10):
    out = []
    ix = 0
    while True:
        # Forward pass for a single character to get its distribution over the next character
        x = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logit = x @ W
        prob = torch.exp(logit)
        prob = prob / prob.sum(1, keepdim=True)
        ix = torch.multinomial(prob[0], num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print('  ' + ''.join(out))


# =====================================================================
# PLOTS
# =====================================================================

#Visualize loss
plt.figure(figsize=(8, 5))
plt.plot(losses, label='neural net (1 layer)')
plt.axhline(loss_count, color='red', linestyle='--', label='counting model')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig('02-makemore/bigram_loss_curve.png', bbox_inches='tight')

#Visualize W in counts: learned probabilities x how often each character occurs as first char
P_nn = W.exp()
P_nn = P_nn / P_nn.sum(1, keepdim=True)
P_nn = P_nn.detach()
N_nn = (P_nn * N.sum(1, keepdim=True)).round().int()
plt.figure(figsize=(16, 16))
plt.imshow(N_nn, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N_nn[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.savefig('02-makemore/bigram_nn_counts.png', bbox_inches='tight')