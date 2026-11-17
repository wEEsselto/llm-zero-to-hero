import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SEED = 2147483647  # same seed for every random step, so the results are reproducible

with open('02-makemore/names.txt', 'r') as f:
    names = f.read().splitlines()

#Dictionaries for char to int and int to char mapping
chars = sorted(set(''.join(names)))

stoi = {}
for i, char in enumerate(chars):
    stoi[char] = i+1
stoi['.'] = 0  #special token marking start and end of a name
itos = {i: s for s, i in stoi.items()}

#Model configuration: all shapes are derived from these
context_length = 3   # how many previous characters the model sees
n_embd = 10          # size of each character's embedding vector
n_hidden = 200       # neurons in the hidden layer
vocab_size = 27


# =====================================================================
# DATASET: TRAIN / DEV / TEST SPLIT
# =====================================================================

def build_dataset(names):
    X = []
    Y = []

    for name in names:
        context = [0]*context_length  # start with "..."
        for ch in name + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append (= sliding window)

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

#Shuffle the names, then split them (not the examples), so no name ends up in two sets
random.seed(42)
random.shuffle(names)
n1 = int(0.8*len(names))
n2 = int(0.9*len(names))

Xtr, Ytr = build_dataset(names[:n1])      # 80% Training: learn the weights
Xdev, Ydev = build_dataset(names[n1:n2])  # 10% Validation: make design decisions
Xte, Yte = build_dataset(names[n2:])      # 10% Test: one final, honest number


# =====================================================================
# MODEL: EMBEDDING -> HIDDEN LAYER (tanh) -> OUTPUT LAYER
# =====================================================================

g = torch.Generator().manual_seed(SEED)
# Embedding table: every character gets a learnable vector in the "character space".
# Similar characters end up close, so the network generalizes to unseen contexts
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((context_length * n_embd, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

def forward(X):
    emb = C[X]  # look up each character's vector: (N, 3, 10)
    h = torch.tanh(emb.view(-1, context_length * n_embd) @ W1 + b1)  # (N, 200)
    logits = h @ W2 + b2  # (N, 27)
    return logits


# =====================================================================
# TRAINING
# =====================================================================

max_steps = 200000
batch_size = 32
lossi = []

for k in range(max_steps):
    # Minibatch: each step uses a random batch of 32 examples.
    # The gradient is noisy but cheap, so for the same compute we get
    # thousands of small steps instead of one exact step on all examples
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

    #FORWARD PASS
    logits = forward(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    #BACKWARD PASS
    for p in parameters:
        p.grad = None
    loss.backward()

    #UPDATE: learning rate decay, 10x smaller for the second half
    lr = 0.1 if k < max_steps // 2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    #track stats
    lossi.append(loss.log10().item())
    if k % 10000 == 0:
        print(f'step {k:6d}/{max_steps}: loss {loss.item():.4f}')


# =====================================================================
# RESULTS: LOSSES AND SAMPLED NAMES
# =====================================================================

@torch.no_grad()  # no gradients needed for evaluation
def split_loss(X, Y):
    return F.cross_entropy(forward(X), Y).item()

print(f'\nMODEL: {sum(p.nelement() for p in parameters)} parameters '
      f'(context {context_length}, embedding {n_embd}, hidden {n_hidden})')

print('\nLOSS (negative log likelihood, lower is better)')
print(f'  train: {split_loss(Xtr, Ytr):.4f}')
print(f'  dev:   {split_loss(Xdev, Ydev):.4f}')
# test set: evaluate only once at the very end, after all tuning is done
# print(f'  test:  {split_loss(Xte, Yte):.4f}')

#Sample names from the MLP
print('\n10 names from the MLP:')
g = torch.Generator().manual_seed(SEED)
for _ in range(10):
    out = []
    context = [0] * context_length  # start with "..."
    while True:
        # Forward pass for a single context (batch size 1)
        logits = forward(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        # Draw the next character's index according to the probability distribution
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]  # slide the window
        out.append(itos[ix])
        if ix == 0:
            break
    print('  ' + ''.join(out))


# =====================================================================
# PLOTS
# =====================================================================

#Visualize loss over steps
plt.figure(figsize=(8, 5))
plt.plot(lossi, linewidth=0.5)
plt.xlabel('step')
plt.ylabel('log10(loss)')
plt.savefig('02-makemore/mlp_loss_curve.png', bbox_inches='tight')

#Visualize embedding: train a second model with a 2D embedding, so the plot
#shows the whole character space (a 10D embedding can't be plotted without losing information)
def train_2d(steps=max_steps):
    g2 = torch.Generator().manual_seed(SEED)
    P = [torch.randn(s, generator=g2) for s in
         [(vocab_size, 2), (context_length * 2, n_hidden), (n_hidden,), (n_hidden, vocab_size), (vocab_size,)]]
    for p in P:
        p.requires_grad = True
    C_, W1_, b1_, W2_, b2_ = P
    fwd = lambda X: torch.tanh(C_[X].view(-1, context_length * 2) @ W1_ + b1_) @ W2_ + b2_
    for k in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g2)
        loss = F.cross_entropy(fwd(Xtr[ix]), Ytr[ix])
        for p in P:
            p.grad = None
        loss.backward()
        lr = 0.1 if k < steps // 2 else 0.01
        for p in P:
            p.data += -lr * p.grad
    with torch.no_grad():
        dev = F.cross_entropy(fwd(Xdev), Ydev).item()
    return C_.detach(), dev

print('\ntraining 2D-embedding model for the plot...')
C2, dev_2d = train_2d()
print(f'  dev loss with 2D embedding: {dev_2d:.4f}')

plt.figure(figsize=(8, 8))
plt.scatter(C2[:, 0], C2[:, 1], s=200)
for i in range(vocab_size):
    plt.text(C2[i, 0].item(), C2[i, 1].item(), itos[i], ha='center', va='center', color='white')
plt.grid(True, alpha=0.3)
plt.savefig('02-makemore/mlp_embeddings.png', bbox_inches='tight')