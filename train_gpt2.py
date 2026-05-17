from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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

    def forward(self, x): # x.shape = (B, T, C)
        B, T, C = x.shape
        qkv = self.c_attn(x) #(B, T, 3C)
        q,k,v = qkv.split(self.config.n_embd, 2) # (B, T, C) * 3

        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) #(B, n_head, T, C / n_head)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        attention_score = (q @ k.transpose(-2, -1)) / (C // self.config.n_head) ** 0.5
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))
        x = F.softmax(attention_score, dim=-1) @ v # (B, n_head, T, C / n_head)
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

    def forward(self, idx): # Size of X = (B, T)
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
        return x

# Generating Process
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained("gpt2")
model.eval()

max_length = 30
num_generate = 5

prompt = "Hello, I'm"
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
idx = idx.repeat(num_generate, 1)

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