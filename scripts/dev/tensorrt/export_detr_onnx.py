"""Export DETR object detection model to ONNX format."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import DetrForObjectDetection, DetrImageProcessor


def export_detr_to_onnx(
    model_name: str = "facebook/detr-resnet-50",
    output_path: str = "models/detr_fp32.onnx",
    image_size: int = 800,
    opset_version: int = 14
):
    """
    Export DETR object detection model to ONNX format.

    Args:
        model_name: HuggingFace model identifier
        output_path: Output ONNX file path
        image_size: Input image size (DETR uses variable sizes, 800 is standard)
        opset_version: ONNX opset version
    """
    print("=" * 70)
    print("DETR MODEL EXPORT TO ONNX")
    print("=" * 70)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading DETR model: {model_name}")
    # Force safetensors to avoid CVE-2025-32434
    model = DetrForObjectDetection.from_pretrained(model_name, use_safetensors=True)
    processor = DetrImageProcessor.from_pretrained(model_name)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    # DETR can handle variable input sizes, but we'll use standard 800x800
    dummy_image = torch.randn(1, 3, image_size, image_size)

    print(f"\nDummy input shape: {dummy_image.shape}")
    print(f"Image size: {image_size}x{image_size}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=dummy_image)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        print(f"Logits shape: {logits.shape}")  # [batch, num_queries, num_classes]
        print(f"Boxes shape: {pred_boxes.shape}")  # [batch, num_queries, 4]

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    print(f"Output path: {output_path}")

    torch.onnx.export(
        model,
        dummy_image,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits', 'pred_boxes'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"\n[SUCCESS] DETR exported to: {output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")

        print(f"\nModel info:")
        print(f"  IR version: {onnx_model.ir_version}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")

    except ImportError:
        print("[WARNING] onnx package not found, skipping verification")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Convert to TensorRT")
    print(f"  python scripts/tensorrt/convert_to_tensorrt.py \\")
    print(f"         --onnx {output_path} \\")
    print(f"         --output models/detr_fp16.trt \\")
    print(f"         --precision fp16")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export DETR to ONNX")
    parser.add_argument(
        "--model",
        default="facebook/detr-resnet-50",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output",
        default="models/detr_fp32.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=800,
        help="Input image size"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    export_detr_to_onnx(
        model_name=args.model,
        output_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset
    )
