import tiktoken
import torch
import torch.nn.functional as F
from model import GPT
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained(url=None, model_type="gpt2")
# model = GPT.from_pretrained(url="./trained_model/model_step_50.pth")

model.eval()
# Autodetect of the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
model.to(device)

max_length = 30
num_generate = 5

prompt = "Hello, I'm a language model, and I can generate text based on the input prompt. I can"
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

with open("generated_output.txt", "w") as f:
    for result in idx:
        decoded_text = tokenizer.decode(result.tolist())
        f.write(decoded_text + "\n")
        f.write("-----------------------------------------------\n")
