# TensorRT Quick Start Guide
**Quick implementation guide for YOLOv8 TensorRT optimization**

## Immediate Implementation: YOLOv8 TensorRT Export

### Why This First?
- **Native Ultralytics support** - no external dependencies
- **3-5x speedup** expected
- **5-10 minute implementation**
- **Zero code complexity** - same API
- **No CUDA version issues**

---

## Step 1: Test Current Performance

```bash
# Activate venv
& C:\Users\kjfle\.venv\Scripts\Activate.ps1

# Run baseline benchmark
python test_ml_quick.py

# Note the timing:
# - Total processing time
# - Images per second
# - GPU memory usage
```

**Expected Baseline:**
- 10 images, batch_size=8
- ~10-15 seconds total
- ~0.7-1.0 images/sec

---

## Step 2: Implement TensorRT Export

### Option A: Automatic Export (Recommended)

Edit `C:\Users\kjfle\.projects\nivo\src\analyzers\ml_vision_yolo.py`:

```python
# Around line 150 in _load_yolo_model():

def _load_yolo_model(self):
    """Load YOLO model with automatic TensorRT conversion."""
    try:
        from ultralytics import YOLO
        from pathlib import Path

        print(f"Loading YOLO model: {self.yolo_model_name}")

        # Check for TensorRT engine file
        if self.yolo_model_name.endswith('.pt'):
            engine_name = self.yolo_model_name.replace('.pt', '_fp16.engine')
            engine_path = Path(engine_name)

            # Export to TensorRT if engine doesn't exist
            if not engine_path.exists():
                print(f"\nExporting {self.yolo_model_name} to TensorRT FP16...")
                print("This is a one-time process (~30-60 seconds)...")

                # Load PyTorch model temporarily
                temp_model = YOLO(self.yolo_model_name)

                # Export to TensorRT with FP16 precision
                temp_model.export(
                    format='engine',
                    device=0 if self.device.type == 'cuda' else 'cpu',
                    half=True,  # FP16 precision
                    workspace=4,  # 4GB workspace
                    verbose=True
                )

                print(f"TensorRT engine saved: {engine_path}")
            else:
                print(f"Found existing TensorRT engine: {engine_path}")

            # Load TensorRT engine
            model_to_load = str(engine_path)
        else:
            # Load .pt file directly (fallback)
            model_to_load = self.yolo_model_name

        self._yolo_model = YOLO(model_to_load)
        self._yolo_model.to(self.device)

        print(f"YOLO model loaded successfully")
        if model_to_load.endswith('.engine'):
            print("Using TensorRT acceleration (FP16)")

    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        raise
```

### Option B: Manual Export (Testing)

```python
# Run once to create engine file
from ultralytics import YOLO

# Load PyTorch model
model = YOLO("yolov8n.pt")

# Export to TensorRT FP16
model.export(
    format='engine',
    device=0,  # CUDA device 0
    half=True,  # FP16 precision for RTX 2080 Ti
    workspace=4,  # 4GB workspace
    verbose=True
)

# Creates: yolov8n.engine (~20-40 MB)
```

**Then modify code to load the .engine file:**
```python
# In ml_vision_yolo.py
self._yolo_model = YOLO("yolov8n.engine")
```

---

## Step 3: Test TensorRT Performance

```bash
# Run same benchmark
python test_ml_quick.py

# Compare results:
# - Total processing time (expect 60-80% reduction)
# - Images per second (expect 3-5x increase)
# - GPU memory usage (may be slightly lower)
```

**Expected Results:**
- 10 images, batch_size=8
- ~3-5 seconds total (vs 10-15 baseline)
- **3-5 images/sec** (vs 0.7-1.0 baseline)
- **3-5x speedup confirmed**

---

## Step 4: Benchmark on Larger Dataset

```bash
# Test on 100 images
python -c "
from src.analyzers.ml_vision_yolo import YOLOVisionAnalyzer
from pathlib import Path
import time

# Get 100 test images
test_dir = Path('D:/Pictures/Camera Roll')
images = list(test_dir.glob('*.jpg'))[:100]

# Initialize analyzer
analyzer = YOLOVisionAnalyzer(
    use_gpu=True,
    batch_size=16,  # Larger batches for TensorRT
    precision='fp16'
)

# Benchmark
start = time.time()
results = analyzer.analyze_batch([str(p) for p in images])
elapsed = time.time() - start

print(f'\n=== Benchmark Results ===')
print(f'Images: {len(images)}')
print(f'Time: {elapsed:.2f}s')
print(f'Speed: {len(images)/elapsed:.2f} img/sec')
print(f'GPU Memory: {analyzer.get_memory_usage()}')
"
```

---

## Verification Checklist

✓ **Performance Increase**
- [ ] 3-5x faster object detection
- [ ] 2-3x faster overall pipeline (with CLIP overhead)

✓ **Accuracy Maintained**
- [ ] Same objects detected as PyTorch version
- [ ] Similar confidence scores
- [ ] No quality degradation

✓ **GPU Efficiency**
- [ ] GPU utilization stays high (>80%)
- [ ] Memory usage stable or lower
- [ ] No OOM errors

✓ **File Management**
- [ ] `.engine` file created (20-40 MB)
- [ ] `.pt` file can be deleted (optional)
- [ ] Engine loads faster than .pt on subsequent runs

---

## Troubleshooting

### Error: "TensorRT not found"
```bash
# Install tensorrt
pip install tensorrt

# Or from NVIDIA wheels
pip install tensorrt --extra-index-url https://pypi.nvidia.com
```

### Error: "CUDA error during export"
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show RTX 2080 Ti

# If False, check CUDA installation
nvidia-smi
```

### Error: "Engine file too large"
```python
# Reduce workspace size
model.export(format='engine', half=True, workspace=2)  # 2GB instead of 4GB
```

### Performance Not Improved
```python
# Verify engine is being used
from ultralytics import YOLO
model = YOLO("yolov8n.engine")
print(model.predictor.model)  # Should mention TensorRT

# Check if FP16 is active
# TensorRT should show FP16 in verbose output during export
```

### Engine File Not Loading
```bash
# Check engine was created
ls yolov8n*.engine

# If missing, re-export
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True)"
```

---

## Configuration Options

### Precision Modes

**FP16 (Recommended for RTX 2080 Ti):**
```python
model.export(format='engine', half=True)  # 2-3x speedup, <1% accuracy loss
```

**FP32 (Baseline):**
```python
model.export(format='engine', half=False)  # 1.5-2x speedup, no accuracy loss
```

**INT8 (Maximum Speed, Phase 3):**
```python
model.export(
    format='engine',
    int8=True,
    data='coco.yaml',  # Requires calibration dataset
)
# 5-8x speedup, 2-5% accuracy loss
```

### Batch Sizes

**Small Batches (Low Latency):**
```python
batch_size=4  # Good for real-time use
```

**Medium Batches (Balanced):**
```python
batch_size=8  # Current default
```

**Large Batches (Maximum Throughput):**
```python
batch_size=16  # Best for bulk processing
# May require adjusting workspace size
```

### Dynamic vs Static Shapes

**Static (Faster, Recommended):**
```python
model.export(format='engine', half=True, dynamic=False)
# Fixed input size, maximum optimization
```

**Dynamic (Flexible):**
```python
model.export(format='engine', half=True, dynamic=True)
# Variable input sizes, slightly slower
```

---

## Next Steps After Validation

1. **Update default config** to use TensorRT
2. **Document speedup** in CLAUDE.md
3. **Commit changes** with benchmark results
4. **Consider CLIP optimization** (Phase 2)

---

## Rollback Plan

If TensorRT causes issues:

```python
# In ml_vision_yolo.py, comment out TensorRT logic:

def _load_yolo_model(self):
    """Load YOLO model (PyTorch only)."""
    from ultralytics import YOLO
    self._yolo_model = YOLO(self.yolo_model_name)  # Load .pt directly
    self._yolo_model.to(self.device)
    print("YOLO model loaded (PyTorch mode)")
```

No other code changes needed - same API.

---

## Expected Timeline

- **Export engine:** 30-60 seconds (one-time)
- **Code changes:** 5-10 minutes
- **Testing:** 5-10 minutes
- **Validation:** 10-20 minutes

**Total: 20-40 minutes** for 3-5x speedup on object detection.
