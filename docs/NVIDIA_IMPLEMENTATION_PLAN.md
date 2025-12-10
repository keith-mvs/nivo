# NVIDIA Build API + TensorRT Implementation Plan

**Hardware**: RTX 2080 Ti (Compute Capability 7.5)
**CUDA**: 12.1
**Driver**: 581.57
**GPU Memory**: 11 GB

---

## Phase 2: NVIDIA Build API Integration

### Overview
NVIDIA Build APIs provide production-ready AI models via cloud endpoints:
- **Retail Object Detection**: Identify products, packages, brands
- **Vision-Language Models**: Generate natural language descriptions
- **No local model hosting**: API-based inference

### 2.1 Research & Requirements

**NVIDIA Build API Catalog**:
```
Available Models:
1. nvidia/retail-object-detection
   - Purpose: Detect products, packages, shelves
   - Input: Image (base64 or URL)
   - Output: Bounding boxes, classes, confidence scores
   - Pricing: Free tier available

2. nvidia/nemotron-nano-12b-v2-vl (Vision-Language)
   - Purpose: Generate image descriptions
   - Input: Image + optional prompt
   - Output: Natural language text
   - Use case: Video frame captions, searchable descriptions
```

**API Requirements**:
- NVIDIA Developer Account (free)
- API Key from build.nvidia.com
- Python requests library
- Rate limiting: 100 requests/minute (free tier)

### 2.2 Setup Instructions

**Step 1: Get API Key**
```bash
# 1. Visit https://build.nvidia.com
# 2. Sign up/log in with NVIDIA account
# 3. Navigate to "API Keys"
# 4. Generate new API key
# 5. Save to environment variable
```

**Step 2: Set Environment Variable**
```powershell
# Windows (persistent)
[System.Environment]::SetEnvironmentVariable('NVIDIA_API_KEY', 'your-api-key', 'User')

# Or create .env file
echo "NVIDIA_API_KEY=your-api-key" > .env
```

**Step 3: Install Dependencies**
```bash
pip install python-dotenv requests httpx pillow
```

### 2.3 Implementation Architecture

**File Structure**:
```
src/
├── analyzers/
│   ├── ml_vision.py           (existing)
│   ├── ml_vision_yolo.py      (existing)
│   └── nvidia_build/
│       ├── __init__.py
│       ├── client.py          (API client)
│       ├── retail_detector.py (product detection)
│       └── vision_language.py (image captioning)
├── utils/
│   └── nvidia_auth.py         (API key management)
└── config/
    └── nvidia_config.yaml     (API settings)
```

**API Client Base Class**:
```python
# src/analyzers/nvidia_build/client.py
import os
import requests
import base64
from typing import Dict, Any, Optional
from pathlib import Path

class NVIDIABuildClient:
    """Base client for NVIDIA Build API."""

    BASE_URL = "https://ai.api.nvidia.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to API."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
```

**Retail Detection Implementation**:
```python
# src/analyzers/nvidia_build/retail_detector.py
from .client import NVIDIABuildClient
from typing import List, Dict

class RetailObjectDetector(NVIDIABuildClient):
    """Detect retail products using NVIDIA Build API."""

    ENDPOINT = "cv/nvidia/retail-object-detection"

    def detect_products(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect products in image.

        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence score

        Returns:
            {
                "detections": [
                    {
                        "class": "bottle",
                        "confidence": 0.95,
                        "bbox": [x, y, w, h]
                    }
                ],
                "product_count": 3,
                "dominant_products": ["bottle", "package"]
            }
        """
        image_b64 = self._encode_image(image_path)

        payload = {
            "image": image_b64,
            "confidence_threshold": confidence_threshold
        }

        response = self._post(self.ENDPOINT, payload)

        # Parse response
        detections = response.get("predictions", [])

        # Filter by confidence
        filtered = [
            d for d in detections
            if d.get("confidence", 0) >= confidence_threshold
        ]

        # Aggregate results
        product_classes = [d["class"] for d in filtered]

        return {
            "detections": filtered,
            "product_count": len(filtered),
            "dominant_products": list(set(product_classes))[:5]
        }
```

**Vision-Language Implementation**:
```python
# src/analyzers/nvidia_build/vision_language.py
from .client import NVIDIABuildClient

class VisionLanguageModel(NVIDIABuildClient):
    """Generate image descriptions using NVIDIA Build API."""

    ENDPOINT = "vlm/nvidia/nemotron-nano-12b-v2-vl"

    def describe_image(
        self,
        image_path: str,
        prompt: str = "Describe what is happening in this image in one sentence."
    ) -> str:
        """
        Generate natural language description.

        Args:
            image_path: Path to image
            prompt: Optional custom prompt

        Returns:
            Text description of image
        """
        image_b64 = self._encode_image(image_path)

        payload = {
            "image": image_b64,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.3  # Lower = more deterministic
        }

        response = self._post(self.ENDPOINT, payload)
        return response.get("text", "")
```

### 2.4 Integration with Video Analyzer

**Enhanced VideoAnalyzer**:
```python
# src/analyzers/video_analyzer.py (additions)
from .nvidia_build.retail_detector import RetailObjectDetector
from .nvidia_build.vision_language import VisionLanguageModel

class VideoAnalyzer:
    def __init__(self, ..., use_nvidia_build: bool = False):
        # Existing init
        ...

        # Add NVIDIA Build analyzers
        self.use_nvidia_build = use_nvidia_build
        if use_nvidia_build:
            self.retail_detector = RetailObjectDetector()
            self.vlm = VisionLanguageModel()

    def _analyze_frame_nvidia(self, frame_path: str) -> Dict:
        """Analyze frame with NVIDIA Build APIs."""
        results = {}

        # Product detection
        if self.retail_detector:
            products = self.retail_detector.detect_products(frame_path)
            results["retail_products"] = products

        # Image description
        if self.vlm:
            description = self.vlm.describe_image(frame_path)
            results["description"] = description

        return results
```

### 2.5 Configuration

**config/nvidia_config.yaml**:
```yaml
nvidia_build:
  enabled: false  # Enable when API key available
  api_key_env: "NVIDIA_API_KEY"

  retail_detection:
    enabled: true
    confidence_threshold: 0.7
    max_detections: 10

  vision_language:
    enabled: true
    default_prompt: "Describe the activity in this video frame"
    max_tokens: 100
    temperature: 0.3

  rate_limiting:
    max_requests_per_minute: 100
    retry_attempts: 3
    retry_delay: 1.0

  caching:
    enabled: true
    cache_dir: ".cache/nvidia_build"
    ttl_hours: 24
```

### 2.6 Testing Plan

**Unit Tests**:
```python
# tests/test_nvidia_build.py
def test_retail_detector():
    detector = RetailObjectDetector()
    result = detector.detect_products("test_product.jpg")
    assert "detections" in result
    assert "product_count" in result

def test_vision_language():
    vlm = VisionLanguageModel()
    description = vlm.describe_image("test_scene.jpg")
    assert len(description) > 0
    assert isinstance(description, str)
```

---

## Phase 3: TensorRT Optimization

### Overview
TensorRT optimizes PyTorch models for NVIDIA GPUs:
- **3-5x faster inference** vs PyTorch
- **Lower GPU memory** usage
- **FP16 precision** using Tensor Cores
- **INT8 quantization** for even more speed

### 3.1 Model Export to ONNX

**Target Models**:
1. CLIP (scene classification) - openai/clip-vit-base-patch32
2. DETR (object detection) - facebook/detr-resnet-50

**CLIP Export Script**:
```python
# scripts/export_clip_onnx.py
import torch
from transformers import CLIPModel, CLIPProcessor

def export_clip_to_onnx():
    """Export CLIP model to ONNX format."""

    model_name = "openai/clip-vit-base-patch32"

    # Load model
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Create dummy input
    dummy_image = torch.randn(1, 3, 224, 224)

    # Export vision encoder only (we don't need text)
    torch.onnx.export(
        model.vision_model,
        dummy_image,
        "models/clip_vision_fp32.onnx",
        input_names=['pixel_values'],
        output_names=['image_embeds'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_embeds': {0: 'batch_size'}
        },
        opset_version=14
    )

    print("CLIP exported to models/clip_vision_fp32.onnx")

if __name__ == "__main__":
    export_clip_to_onnx()
```

**DETR Export Script**:
```python
# scripts/export_detr_onnx.py
import torch
from transformers import DetrForObjectDetection

def export_detr_to_onnx():
    """Export DETR model to ONNX format."""

    model_name = "facebook/detr-resnet-50"

    # Load model
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.eval()

    # Dummy input
    dummy_image = torch.randn(1, 3, 800, 800)

    # Export
    torch.onnx.export(
        model,
        dummy_image,
        "models/detr_fp32.onnx",
        input_names=['pixel_values'],
        output_names=['logits', 'pred_boxes'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        },
        opset_version=14
    )

    print("DETR exported to models/detr_fp32.onnx")

if __name__ == "__main__":
    export_detr_to_onnx()
```

### 3.2 ONNX to TensorRT Conversion

**Installation**:
```bash
# TensorRT already included with CUDA 12.1
# Verify installation
python -c "import tensorrt; print(tensorrt.__version__)"
```

**Conversion Script**:
```python
# scripts/convert_to_tensorrt.py
import tensorrt as trt
import numpy as np

class TensorRTConverter:
    """Convert ONNX models to TensorRT engines."""

    def __init__(self, precision: str = "fp16"):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.precision = precision

    def build_engine(
        self,
        onnx_path: str,
        engine_path: str,
        max_batch_size: int = 16
    ):
        """
        Build TensorRT engine from ONNX.

        Args:
            onnx_path: Path to ONNX model
            engine_path: Output path for TensorRT engine
            max_batch_size: Maximum batch size for inference
        """
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            2 << 30  # 2GB
        )

        # Set precision
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision (Tensor Cores)")
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision (requires calibration)")

        # Build engine
        print("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        print(f"Engine saved to {engine_path}")

        # Print engine info
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        print(f"Engine max batch size: {engine.max_batch_size}")
        print(f"Engine num bindings: {engine.num_bindings}")

# Usage
converter = TensorRTConverter(precision="fp16")
converter.build_engine(
    "models/clip_vision_fp32.onnx",
    "models/clip_vision_fp16.trt",
    max_batch_size=16
)
```

### 3.3 TensorRT Inference Wrapper

**Implementation**:
```python
# src/analyzers/tensorrt_engine.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    """TensorRT inference engine wrapper."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate GPU memory
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate device memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'name': binding, 'mem': device_mem, 'dtype': dtype})
            else:
                self.outputs.append({'name': binding, 'mem': device_mem, 'dtype': dtype, 'size': size})

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        # Copy input to GPU
        cuda.memcpy_htod(self.inputs[0]['mem'], input_data)

        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Copy output from GPU
        output = np.empty(self.outputs[0]['size'], dtype=self.outputs[0]['dtype'])
        cuda.memcpy_dtoh(output, self.outputs[0]['mem'])

        return output
```

### 3.4 Integration with Analyzers

**TensorRT ML Vision Analyzer**:
```python
# src/analyzers/ml_vision_tensorrt.py
from .tensorrt_engine import TensorRTInference
import numpy as np

class TensorRTMLVisionAnalyzer:
    """ML vision analysis using TensorRT optimized models."""

    def __init__(self):
        self.clip_engine = TensorRTInference("models/clip_vision_fp16.trt")
        self.detr_engine = TensorRTInference("models/detr_fp16.trt")

    def classify_scene(self, image: np.ndarray) -> str:
        """Classify scene using TensorRT CLIP."""
        # Preprocess image
        image_tensor = self._preprocess_clip(image)

        # Inference
        embeddings = self.clip_engine.infer(image_tensor)

        # Get scene class
        scene = self._clip_postprocess(embeddings)
        return scene

    def detect_objects(self, image: np.ndarray) -> list:
        """Detect objects using TensorRT DETR."""
        # Preprocess
        image_tensor = self._preprocess_detr(image)

        # Inference
        outputs = self.detr_engine.infer(image_tensor)

        # Parse outputs
        objects = self._detr_postprocess(outputs)
        return objects
```

### 3.5 Benchmarking

**Benchmark Script**:
```python
# scripts/benchmark_tensorrt.py
import time
import numpy as np
from src.analyzers.ml_vision import MLVisionAnalyzer
from src.analyzers.ml_vision_tensorrt import TensorRTMLVisionAnalyzer

def benchmark():
    """Compare PyTorch vs TensorRT performance."""

    # Create test data
    test_images = [np.random.randn(3, 224, 224).astype(np.float32) for _ in range(100)]

    # PyTorch baseline
    pytorch_analyzer = MLVisionAnalyzer("cuda")
    start = time.time()
    for img in test_images:
        pytorch_analyzer.classify_scene(img)
    pytorch_time = time.time() - start

    # TensorRT optimized
    tensorrt_analyzer = TensorRTMLVisionAnalyzer()
    start = time.time()
    for img in test_images:
        tensorrt_analyzer.classify_scene(img)
    tensorrt_time = time.time() - start

    # Results
    print(f"PyTorch: {pytorch_time:.2f}s")
    print(f"TensorRT: {tensorrt_time:.2f}s")
    print(f"Speedup: {pytorch_time / tensorrt_time:.2f}x")

if __name__ == "__main__":
    benchmark()
```

---

## Implementation Roadmap

### Week 1: NVIDIA Build API
- [ ] Day 1-2: Get API key, set up environment
- [ ] Day 3-4: Implement client + retail detector
- [ ] Day 5-6: Implement vision-language model
- [ ] Day 7: Testing and documentation

### Week 2: TensorRT Setup
- [ ] Day 1-2: Export models to ONNX
- [ ] Day 3-4: Convert ONNX to TensorRT (FP16)
- [ ] Day 5-6: Implement inference wrapper
- [ ] Day 7: Integration testing

### Week 3: Optimization & Benchmarking
- [ ] Day 1-2: Benchmark TensorRT vs PyTorch
- [ ] Day 3-4: Optimize batch processing
- [ ] Day 5-6: INT8 calibration (optional)
- [ ] Day 7: Final documentation

---

## Success Criteria

### NVIDIA Build API
- ✓ API integration working
- ✓ Retail detection functional
- ✓ Vision-language descriptions generated
- ✓ Rate limiting implemented
- ✓ Error handling robust

### TensorRT
- ✓ 3-5x speedup vs PyTorch (target)
- ✓ <1 second/frame inference (target)
- ✓ FP16 precision working
- ✓ Batch processing optimized
- ✓ Memory usage reduced

### Integration
- ✓ Video analyzer enhanced
- ✓ Database schema updated for new tags
- ✓ Search supports new categories
- ✓ Documentation complete
- ✓ Tests passing

---

## Next Immediate Steps

1. **Get NVIDIA API Key** (5 min)
   - Visit build.nvidia.com
   - Sign up and generate key

2. **Create Implementation Files** (30 min)
   - Set up directory structure
   - Create base classes
   - Add configuration files

3. **Test API Connection** (15 min)
   - Simple test request
   - Verify authentication
   - Check rate limits

**Ready to begin?** Confirm API key acquisition and we'll start implementation.
