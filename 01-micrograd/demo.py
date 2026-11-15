"""
Train the same small network on the "two moons" dataset twice:
first with micrograd, then with PyTorch. Then plot what each one learned.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from nn import MLP

random.seed(1337)

STEPS = 100
LEARNING_RATE = 1.0   # decays linearly to 0.1 over the training steps
ALPHA = 1e-4   # regularization: keeps the weights small, which gives a smoother boundary


# ---------------------------------------------------------------- data
# two interleaving half circles: 50 points labelled -1, 50 points labelled +1

t = np.linspace(0, np.pi, 50)
top = np.column_stack([np.cos(t), np.sin(t)])
bottom = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
X = np.vstack([top, bottom]) + np.random.default_rng(1337).normal(scale=0.1, size=(100, 2))
y = np.array([-1.0] * 50 + [1.0] * 50)


# ---------------------------------------------------------------- micrograd

model = MLP(2, [16, 16, 1])

for step in range(STEPS):
    # forward pass
    scores = [model(x) for x in X.tolist()]
    loss = sum((1 - yi * s).relu() for yi, s in zip(y.tolist(), scores)) / len(y)
    loss = loss + ALPHA * sum(p * p for p in model.parameters())

    # backward pass
    model.zero_grad()
    loss.backward()

    # update
    lr = LEARNING_RATE * (1 - 0.9 * step / STEPS)
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 10 == 0:
        print(f"micrograd  step {step:3d}  loss {loss.data:.4f}")


# ---------------------------------------------------------------- PyTorch

torch_model = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
)

# start from the same weights the micrograd model started with
random.seed(1337)
start = MLP(2, [16, 16, 1])
with torch.no_grad():
    for linear, layer in zip(torch_model[::2], start.layers):
        linear.weight.copy_(torch.tensor([[w.data for w in n.w] for n in layer.neurons]))
        linear.bias.copy_(torch.tensor([n.b.data for n in layer.neurons]))

optimizer = torch.optim.SGD(torch_model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=STEPS)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

for step in range(STEPS):
    # forward pass
    scores = torch_model(X_t).squeeze(1)
    loss = torch.relu(1 - y_t * scores).mean()
    loss = loss + ALPHA * sum((p * p).sum() for p in torch_model.parameters())

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()
    scheduler.step()

    if step % 10 == 0:
        print(f"PyTorch    step {step:3d}  loss {loss.item():.4f}")


# ---------------------------------------------------------------- plot
# color every point of a grid by the model's prediction: the boundary is where the score is 0

xx, yy = np.meshgrid(np.linspace(-2, 3, 50), np.linspace(-1.5, 2, 50))
grid = np.column_stack([xx.ravel(), yy.ravel()])

micro_scores = np.array([model(p).data for p in grid.tolist()])
with torch.no_grad():
    torch_scores = torch_model(torch.tensor(grid, dtype=torch.float32)).squeeze(1).numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, scores, title in [(axes[0], micro_scores, "micrograd"), (axes[1], torch_scores, "PyTorch")]:
    bound = np.abs(scores).max()
    ax.contourf(xx, yy, scores.reshape(xx.shape), levels=[-bound, 0, bound], colors=["#fdae61", "#abdda4"])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Spectral, edgecolors="k")
    ax.set_title(title)
fig.tight_layout()
fig.savefig("demo.png", dpi=120)
plt.show()
