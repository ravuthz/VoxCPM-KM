import subprocess
import torch

print("=== NVIDIA-SMI ===")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
except FileNotFoundError:
    print("nvidia-smi not found \n")

print("=== PyTorch CUDA Check ===")
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("GPU not support, using CPU")
print("\n")
