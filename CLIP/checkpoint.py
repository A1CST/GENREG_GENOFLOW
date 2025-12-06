import torch
ckpt = torch.load("checkpoint_step_288000.pth", map_location="cpu")

print("TOP LEVEL KEYS:", ckpt.keys())

for k, v in ckpt.items():
    print(k, type(v))