import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# Model configuration
n_embd = 384       # size of each token's vector (C)
n_head = 6         # attention heads per block, each of size n_embd // n_head = 64
n_layer = 6        # number of transformer blocks
block_size = 256   # maximum context length (T)

# Training configuration
batch_size = 64      # sequences processed in parallel (B)
max_steps = 5000
eval_interval = 500  # how often to print train/val loss
eval_iters = 200     # batches averaged per loss estimate
learning_rate = 3e-4
dropout = 0.2        # fraction of activations randomly zeroed during training, against overfitting

# Use the Mac's GPU (Apple Silicon) if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)


# =====================================================================
# DATA: CHARACTER-LEVEL TOKENIZER AND BATCHES
# =====================================================================

with open('03-gpt/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)  # 65 characters
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]  # string -> list of token ids

def decode(ids):
    return ''.join(itos[i] for i in ids)  # list of token ids -> string

# 90% train, 10% val. No shuffling: the text is one continuous sequence
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    # B random chunks of length T. The target y is the input x shifted by one character,
    # so every position in a chunk is its own training example
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# =====================================================================
# MODEL
# =====================================================================

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what am I looking for?"
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what do I contain?"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what I pass on"
        # Lower-triangular mask: a token may only look at itself and earlier tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)    # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Attention scores: how much each token attends to each other token.
        # Scaled by 1/sqrt(hs) so softmax doesn't saturate
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # hide the future
        wei = F.softmax(wei, dim=-1)                              # each row sums to 1
        wei = self.dropout(wei)

        return wei @ v  # weighted sum of the values: (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, outputs concatenated and projected back to n_embd."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: each token 'thinks' on its own after attention. 4x wider inside."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention), then computation (feed-forward).
    Pre-norm with residual connections: x = x + sublayer(norm(x))"""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, n_embd // n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # what the token is
        self.position_embedding = nn.Embedding(block_size, n_embd)  # where it is
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)              # final norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # one score per possible next token
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Small initial weights, so the first loss is close to uniform guessing
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                              # (B, T, C)
        pos = self.position_embedding(torch.arange(T, device=device))  # (T, C)
        x = tok + pos                                                # (B, T, C)
        x = self.blocks(x)                                           # (B, T, C)
        logits = self.lm_head(self.ln_f(x))                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy expects (N, classes) and (N,), so flatten B and T
            loss = F.cross_entropy(logits.view(B * T, vocab_size), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # The same sampling loop as in makemore: predict, sample, append, repeat
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # position embedding only covers block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                # only the last position predicts the next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =====================================================================
# TRAINING
# =====================================================================

model = GPT().to(device)
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device: {device}')

@torch.no_grad()
def estimate_loss():
    # Average over many batches: a single batch loss is too noisy
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start = time.time()
best_val = float('inf')

for step in range(max_steps):
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"step {step:5d}: train {losses['train']:.4f}, val {losses['val']:.4f} "
              f"({time.time() - start:.0f}s)")
        # Early stopping: keep the weights with the lowest val loss, not the last ones
        if losses['val'] < best_val:
            best_val = losses['val']
            torch.save(model.state_dict(), '03-gpt/gpt.pt')
            print(f'  saved new best model (val {best_val:.4f})')

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Load the best checkpoint (lowest val loss) for generating text
model.load_state_dict(torch.load('03-gpt/gpt.pt'))
model.eval()  # switch dropout off


# =====================================================================
# RESULTS: GENERATED TEXT
# =====================================================================

context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start from token 0 ('\n')
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))