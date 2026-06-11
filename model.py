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

    def forward(self, x, past_kv=None): # x.shape = (B, T, C)
        B, T, C = x.shape
        # Without using KV Cache
        qkv = self.c_attn(x) #(B, T, 3C)
        q,k,v = qkv.split(self.config.n_embd, 2) # (B, T, C) * 3

        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) #(B, n_head, T, C / n_head)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        is_causal = True
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
            is_causal = False

        # attention_score = (q @ k.transpose(-2, -1)) / (C // self.config.n_head) ** 0.5
        # mask = torch.tril(torch.ones(T, T, device=x.device))
        # attention_score = attention_score.masked_fill(mask == 0, float("-inf"))
        # x = F.softmax(attention_score, dim=-1) @ v # (B, n_head, T, C / n_head)


        x = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        x = x.transpose(1, 2).contiguous().view(B, T, C)

        x = self.c_proj(x)

        return x, (k, v)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(normalized_shape=config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        output = self.ln_1(x)
        output, (k, v) = self.attn(output, past_kv)
        x = output + x
        output = self.ln_2(x)
        output = self.mlp(output)
        x = output + x
        return x, (k, v)

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
    def from_pretrained(cls, url=None, model_type="gpt2"):
        # Create an instance
        model = GPT(config=GPT2Config())
        my_state_dict = model.state_dict()

        # Load the pretrained model
        if url is None:
            pretrained_state_dict = AutoModelForCausalLM.from_pretrained(model_type).state_dict()
            transposed_key = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

            with torch.no_grad():
                for k in my_state_dict:
                    if any(k.endswith(suffix) for suffix in transposed_key):
                        my_state_dict[k].copy_(pretrained_state_dict[k].t())
                    else:
                        my_state_dict[k].copy_(pretrained_state_dict[k])
        else:
            pretrained_state_dict = torch.load(url, map_location="cpu")
            # Remove the prefix from torch.comiple
            cleaned_state_dict = {
                k.replace("_orig_mod.", ""): v 
                for k, v in pretrained_state_dict.items()
            }
            model.load_state_dict(cleaned_state_dict)
        return model

    def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4):
        require_dacay_parapmeters = [p for p in self.parameters() if p.ndim >= 2]
        no_decay_parameters = [p for p in self.parameters() if p.ndim != 2]
        optimizer = torch.optim.AdamW([
            {'params': require_dacay_parapmeters, 'weight_decay': weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False): # Size of X = (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size

        # Token embedding
        token_embeddings = self.transformer["wte"](idx)  # (B, T, n_embd)
        past_length = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        position_embeddings = self.transformer["wpe"](torch.arange(past_length, past_length + T, device=idx.device))
        x = token_embeddings + position_embeddings

        # Transformer blocks
        new_kvs = []
        for i, block in enumerate(self.transformer["h"]):
            layer_past = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, layer_past)
            new_kvs.append(new_kv)

        # Output
        x = self.transformer["ln_f"](x)
        x = self.lm_head(x)

        # Return logits and loss if targets is provided -> for training
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.shape[-1]), targets.view(-1))
            return x, loss

        if use_cache:
            return x, new_kvs

        return x
