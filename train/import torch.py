import torch

if torch.backends.mps.is_available():
    print("Metal Performance Shaders (MPS) GPU is available")
else:
    print("CPU version installed")