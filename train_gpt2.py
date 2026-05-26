import platform

import torch
import tiktoken
import math
import time
import matplotlib.pyplot as plt
from model import GPT, GPT2Config

# Autodetect of the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device} device")
if device == "cuda":
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

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
if device == "cuda" and platform.system() == "Linux":
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
loss_accumulated_list = []
for i in range(50):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    time_start = time.time()
    loss_accumulated = 0.0
    optimizer.zero_grad()

    for micro_step in range(gradient_accumulation_steps):
        x, y = data_loader.next_batch(batch_size, time_span)
        x, y = x.to(device), y.to(device)


        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / gradient_accumulation_steps
        loss_accumulated += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    time_end = time.time()

    log_directory = "log.txt"
    message = f"At {i + 1} training step | The loss is {loss_accumulated.item():.6f} | Time taken for this step is {time_end - time_start:.4f} seconds | Token per second: {(gradient_accumulation_steps * batch_size * time_span) / (time_end - time_start):.2f} | Norm: {norm:.2f} | lr: {lr:.6f}\n"
    print(message)
    with open(log_directory, "a") as f:
        f.write(message)

    loss_accumulated_list.append(loss_accumulated.item())

    # Save the model every 1000 steps
    if (i + 1) % 1000 == 0:
        torch.save(model.state_dict(), f"./trained_model/model_step_{i + 1}.pth")

torch.save(model.state_dict(), f"./trained_model/model_final.pth")

# Plot the training loss
plt.plot(loss_accumulated_list)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png")