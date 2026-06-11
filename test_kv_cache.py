import torch
import tiktoken
import torch.nn.functional as F
from model import GPT

torch.manual_seed(42)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device} device")

tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained(url="./trained_model/model_final.pth")
model.eval().to(device)

prompt = "First Citizen:"
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

n_generate = 20

#--------------Naive (no cache)----------------
torch.manual_seed(42)
idx_naive = torch.clone(idx)

with torch.no_grad():
    for _ in range(n_generate):
        logits = model(idx_naive)
        logits = logits[:,-1,:]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx_naive = torch.cat((idx_naive, next_token), dim=-1)

#-------------With cache----------------
torch.manual_seed(42)
idx_cache = torch.clone(idx)

with torch.no_grad():
    # Prefill: use the all prompt to get the inital cache
    logits, past_kvs = model(idx_cache, use_cache=True)
    logits = logits[:,-1,:]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    idx_cache = torch.cat((idx_cache, next_token), dim=-1)

    # Decode: each time only use the last token and the cache
    for _ in range(n_generate - 1):
        logits, past_kvs = model(next_token, past_kvs=past_kvs, use_cache=True)
        logits = logits[:,-1,:]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx_cache = torch.cat((idx_cache, next_token), dim=-1)

print("Naive: ", tokenizer.decode(idx_naive[0].tolist()))
print("Cache: ", tokenizer.decode(idx_cache[0].tolist()))
print("Match: ", torch.equal(idx_naive, idx_cache))
assert torch.equal(idx_naive, idx_cache), "KV cache output diverges from naive!"