"""Demo of GPU monitoring during image analysis."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.gpu_monitor import GPUMonitor
import torch
import time

print("=" * 70)
print("GPU Monitor Demo for Image Engine")
print("=" * 70)

# Check if GPU is available
if not torch.cuda.is_available():
    print("\nNo GPU detected. This demo requires a CUDA-capable GPU.")
    sys.exit(1)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# Create monitor
monitor = GPUMonitor(update_interval=0.5)
monitor.start()

print("Starting GPU workload simulation...")
print("(This simulates what happens during ML analysis)\n")

# Simulate GPU workload
try:
    for i in range(10):
        # Create tensors on GPU (simulating ML inference)
        x = torch.randn(2000, 2000).cuda()
        y = torch.randn(2000, 2000).cuda()
        z = torch.matmul(x, y)

        # Print current status
        status = monitor.get_status_string()
        print(f"Batch {i+1}/10: {status}")

        # Small delay to let monitoring catch up
        time.sleep(0.5)

        # Clean up
        del x, y, z
        torch.cuda.empty_cache()

finally:
    monitor.stop()

print("\n" + "=" * 70)
print("Final GPU Statistics:")
print("=" * 70)
monitor.print_stats()

print("\nThis is what you'll see during actual image processing!")
print("Run: python -m src.cli analyze \"D:\\Pictures\\Camera Roll\"")
