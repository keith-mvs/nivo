"""Export CLIP vision model to ONNX format for TensorRT conversion."""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import CLIPModel, CLIPProcessor


def export_clip_to_onnx(
    model_name: str = "openai/clip-vit-base-patch32",
    output_path: str = "models/clip_vision_fp32.onnx",
    opset_version: int = 14
):
    """
    Export CLIP vision encoder to ONNX format.

    Args:
        model_name: HuggingFace model identifier
        output_path: Output ONNX file path
        opset_version: ONNX opset version (14 recommended for TensorRT 8+)
    """
    print("=" * 70)
    print("CLIP MODEL EXPORT TO ONNX")
    print("=" * 70)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading CLIP model: {model_name}")
    # Force safetensors to avoid CVE-2025-32434
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Model type: {type(model)}")

    # We only need the vision encoder for image analysis
    vision_model = model.vision_model

    print(f"\nVision model parameters: {sum(p.numel() for p in vision_model.parameters()):,}")

    # Create dummy input (CLIP expects 224x224 images)
    dummy_image = torch.randn(1, 3, 224, 224)

    print(f"\nDummy input shape: {dummy_image.shape}")
    print(f"Input format: [batch_size, channels, height, width]")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = vision_model(pixel_values=dummy_image)
        pooled_output = output.pooler_output
        print(f"Output shape: {pooled_output.shape}")
        print(f"Output dtype: {pooled_output.dtype}")

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    print(f"Output path: {output_path}")

    torch.onnx.export(
        vision_model,
        dummy_image,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['pooler_output', 'last_hidden_state'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"\n[SUCCESS] CLIP exported to: {output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")

        # Print model info
        print(f"\nModel info:")
        print(f"  IR version: {onnx_model.ir_version}")
        print(f"  Producer: {onnx_model.producer_name}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")

        # Print inputs/outputs
        print(f"\n  Inputs:")
        for input_tensor in onnx_model.graph.input:
            print(f"    - {input_tensor.name}: {input_tensor.type}")

        print(f"\n  Outputs:")
        for output_tensor in onnx_model.graph.output:
            print(f"    - {output_tensor.name}: {output_tensor.type}")

    except ImportError:
        print("[WARNING] onnx package not found, skipping verification")
        print("          Install with: pip install onnx")
    except Exception as e:
        print(f"[ERROR] ONNX verification failed: {e}")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Convert to TensorRT")
    print(f"  python scripts/tensorrt/convert_to_tensorrt.py \\")
    print(f"         --onnx {output_path} \\")
    print(f"         --output models/clip_vision_fp16.trt \\")
    print(f"         --precision fp16")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export CLIP to ONNX")
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output",
        default="models/clip_vision_fp32.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    export_clip_to_onnx(
        model_name=args.model,
        output_path=args.output,
        opset_version=args.opset
    )
