import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Step 2: data आणि tokenizer ----------
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocab size:", vocab_size)          # 65

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------- Step 3: batching ----------
torch.manual_seed(1337)
block_size = 8
batch_size = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ---------- Step 4: Bigram model ----------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
xb, yb = get_batch("train")
logits, loss = model(xb, yb)
print(logits.shape)    # torch.Size([256, 65])
print(loss.item())     # सुमारे 4.2 ते 4.8
start = torch.zeros((1, 1), dtype=torch.long)    # सुरुवात: token 0 (newline)
out = model.generate(start, max_new_tokens=200)
print(decode(out[0].tolist()))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(10000):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

print("final loss:", loss.item())
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))