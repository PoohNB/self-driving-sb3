import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Print GPU device name
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Create a tensor on CPU
    x = torch.rand(3, 3)
    print("Tensor on CPU:", x)

    # Move tensor to GPU
    x = x.to('cuda')
    print("Tensor on GPU:", x)

    # Perform a basic operation on GPU
    y = torch.rand(3, 3).to('cuda')
    z = x + y
    print("Result of tensor addition on GPU:", z)

    # Move result back to CPU
    z = z.to('cpu')
    print("Result moved back to CPU:", z)
else:
    print("CUDA is not available.")
