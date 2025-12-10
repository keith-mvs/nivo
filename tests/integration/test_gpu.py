"""Quick GPU test for Image Engine."""

import torch
import sys
import time

print("=" * 60)
print("GPU Test for Image Engine")
print("=" * 60)

# Test 1: PyTorch CUDA availability
print("\n1. PyTorch CUDA Detection:")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ERROR: CUDA not available!")
    sys.exit(1)

# Test 2: Create tensors on GPU
print("\n2. GPU Tensor Test:")
try:
    # Create a large tensor on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()

    # Perform computation
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start

    print(f"   [OK] GPU computation successful")
    print(f"   Time: {gpu_time*1000:.2f}ms")
    print(f"   GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    # Clear
    del x, y, z
    torch.cuda.empty_cache()

except Exception as e:
    print(f"   [FAIL] GPU computation failed: {e}")
    sys.exit(1)

# Test 3: Load ML Analyzer
print("\n3. ML Analyzer GPU Test:")
try:
    from src.core.analyzers.ml_vision import MLVisionAnalyzer

    analyzer = MLVisionAnalyzer(use_gpu=True, batch_size=4)
    print(f"   [OK] ML Analyzer initialized")
    print(f"   Device: {analyzer.device}")
    print(f"   Batch size: {analyzer.batch_size}")

    # Check if it's actually on GPU
    if analyzer.device.type == 'cuda':
        print(f"   [OK] Analyzer is configured for GPU!")
    else:
        print(f"   [FAIL] Analyzer is on CPU (expected GPU)")

except Exception as e:
    print(f"   [FAIL] ML Analyzer failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("GPU Test Complete")
print("=" * 60)

# Print instructions
print("\nTo monitor GPU during actual processing:")
print("  1. Open another terminal")
print("  2. Run: nvidia-smi -l 1")
print("  3. Run your image analysis")
print("\nThe GPU will only be used during ML analysis (Phase 3/3)")
