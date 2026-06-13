import time
import tiktoken
import matplotlib.pyplot as plt
from model import GPT
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
torch.manual_seed(42)

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

model = GPT.from_pretrained(url=None, model_type="gpt2").to(device)

prompt = "I am"
tokenizer = tiktoken.get_encoding("gpt2")
input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

def generate_naive(model, input_ids, n_tokens):
    input_ids = input_ids.clone()
    torch.manual_seed(42)
    with torch.no_grad():
        sync()
        start = time.time()

        for i in range(n_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        sync()
        end = time.time()
        time_taken = (end - start) * 1000 # in ms
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text, time_taken

def generate_with_cache(model, input_ids, n_tokens):
    input_ids = input_ids.clone()
    torch.manual_seed(42)

    with torch.no_grad():
        sync()
        start = time.time()

        logits, past_kvs = model(input_ids, use_cache=True)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=-1)

        for _ in range(n_tokens - 1):
            logits, past_kvs = model(next_token, past_kvs=past_kvs, use_cache=True)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)

        sync()
        end = time.time()
        time_taken = (end - start) * 1000
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text, time_taken

# --------- Warm Up -----------------------
generate_naive(model, input_ids, n_tokens=5)
generate_with_cache(model, input_ids, n_tokens=5)
# -----------------------------------------

naive = {}
cache = {}
for n_tokens in [50, 100, 200, 400, 800]:
    generated_text, time_taken = generate_naive(model, input_ids, n_tokens=n_tokens)
    naive[n_tokens] = time_taken
    generated_text, time_taken = generate_with_cache(model, input_ids, n_tokens=n_tokens)
    cache[n_tokens] = time_taken

plt.plot(list(cache.keys()), list(cache.values()), "o-", label="Cache")
plt.plot(list(naive.keys()), list(naive.values()), "s-", label="Naive")
plt.xlabel("n_tokens")
plt.ylabel("Latency (ms)")
plt.title("KV Cache vs Naive Generation Latency (GPT-2 124M)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kv_cache_comparison.png")

# Calculation for SpeedUp
for n in naive:
    speedup = naive[n]/cache[n]
    print(f"{n:^10}{naive[n]:^15.1f}{cache[n]:^15.1f}{speedup:^10.2f}")