import tiktoken
import torch
import torch.nn.functional as F
from train_gpt2 import GPT, device
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
