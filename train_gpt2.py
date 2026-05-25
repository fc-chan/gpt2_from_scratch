import math
import time
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import tiktoken

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALING_INIT = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALING_INIT = True

    def forward(self, x): # x.shape = (B, T, C)
        B, T, C = x.shape
        qkv = self.c_attn(x) #(B, T, 3C)
        q,k,v = qkv.split(self.config.n_embd, 2) # (B, T, C) * 3

        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) #(B, n_head, T, C / n_head)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        # attention_score = (q @ k.transpose(-2, -1)) / (C // self.config.n_head) ** 0.5
        # mask = torch.tril(torch.ones(T, T, device=x.device))
        # attention_score = attention_score.masked_fill(mask == 0, float("-inf"))
        # x = F.softmax(attention_score, dim=-1) @ v # (B, n_head, T, C / n_head)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(normalized_shape=config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x):
        output = self.ln_1(x)
        output = self.attn(output)
        x = output + x
        output = self.ln_2(x)
        output = self.mlp(output)
        x = output + x
        return x

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.vocab_size, self.config.n_embd),
                "wpe": nn.Embedding(self.config.block_size, self.config.n_embd),
                "h": nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                "ln_f": nn.LayerNorm(normalized_shape=self.config.n_embd),
            }
        )

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALING_INIT"):
                std = 0.02 / (2 * self.config.n_layer) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        # Create an instance
        model = GPT(config=GPT2Config())

        # Load the pretrained model
        pretrained_state_dict = AutoModelForCausalLM.from_pretrained(model_type).state_dict()
        my_state_dict = model.state_dict()
        transposed_key = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        with torch.no_grad():
            for k in my_state_dict:
                if any(k.endswith(suffix) for suffix in transposed_key):
                    my_state_dict[k].copy_(pretrained_state_dict[k].t())
                else:
                    my_state_dict[k].copy_(pretrained_state_dict[k])

        return model

    def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4):
        require_dacay_parapmeters = [p for p in self.parameters() if p.ndim >= 2]
        no_decay_parameters = [p for p in self.parameters() if p.ndim != 2]
        optimizer = torch.optim.AdamW([
            {'params': require_dacay_parapmeters, 'weight_decay': weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, targets=None): # Size of X = (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size

        # Token embedding
        token_embeddings = self.transformer["wte"](idx)  # (B, T, n_embd)
        position_embeddings = self.transformer["wpe"](torch.arange(T, device=idx.device))
        x = token_embeddings + position_embeddings

        # Transformer blocks
        for block in self.transformer["h"]:
            x = block(x)

        # Output
        x = self.transformer["ln_f"](x)
        x = self.lm_head(x)

        # Return logits and loss if targets is provided -> for training
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.shape[-1]), targets.view(-1))
            return x, loss

        return x

# Autodetect of the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device} device")

class DataLoader():
    def __init__(self):
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.data = enc.encode(text)
        self.current_position = 0

    def next_batch(self, batch_size, time_span):
        data = torch.tensor(self.data[self.current_position : self.current_position + batch_size * time_span + 1], dtype=torch.long)
        x = data[: -1].view(batch_size, time_span)
        y = data[1 :].view(batch_size, time_span)
        self.current_position += batch_size * time_span

        if (self.current_position + batch_size * time_span + 1) > len(self.data):
            self.current_position = 0

        return x, y

# Training Process
data_loader = DataLoader()
model = GPT(config=GPT2Config()).to(device)
model = torch.compile(model)

total_batch_size = 16384
batch_size = 8
time_span = 256

assert total_batch_size % (batch_size * time_span) == 0
gradient_accumulation_steps = total_batch_size // (batch_size * time_span)
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

num_epochs = 10
num_iterations = num_epochs * (len(data_loader.data) // (batch_size * time_span))

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 5
max_steps = 50

def get_lr(it): # it is from 0 to num_iterations - 1
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4)
model.train()
for i in range(50):
    torch.mps.synchronize()
    time_start = time.time()
    loss_accumulated = 0.0
    optimizer.zero_grad()

    for micro_step in range(gradient_accumulation_steps):
        x, y = data_loader.next_batch(batch_size, time_span)
        x, y = x.to(device), y.to(device)


        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / gradient_accumulation_steps
        loss_accumulated += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.mps.synchronize()
    time_end = time.time()

    print(f"At {i + 1} training step | The loss is {loss_accumulated.item():.6f} | Time taken for this step is {time_end - time_start:.4f} seconds | Token per second: {(gradient_accumulation_steps * batch_size * time_span) / (time_end - time_start):.2f} | Norm: {norm:.2f} | lr: {lr:.6f}")
"""
# Generating Process

tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

max_length = 30
num_generate = 5

prompt = "Hello, I'm a language model,"
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
idx = idx.repeat(num_generate, 1)
idx = idx.to(device)

with torch.no_grad():
    for _ in range(max_length):
        result = model(idx)
        logits = F.softmax(result, dim=-1)
        logits = logits[:, -1, :]
        next_token = torch.multinomial(logits, num_samples=1, replacement=True)
        idx = torch.cat((idx, next_token), dim=1)

for result in idx:
    print(tokenizer.decode(result.tolist()))
    print("-----------------------------------------------")
"""