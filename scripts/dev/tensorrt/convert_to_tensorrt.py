"""Convert ONNX models to TensorRT engines for optimized inference."""

import argparse
import tensorrt as trt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TensorRTConverter:
    """Convert ONNX models to TensorRT engines."""

    def __init__(self, verbose: bool = True):
        """
        Initialize TensorRT converter.

        Args:
            verbose: Enable verbose logging
        """
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    def convert(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",
        workspace_size_mb: int = 4096,
        max_batch_size: int = 16
    ):
        """
        Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode (fp32, fp16, int8)
            workspace_size_mb: Max workspace size in MB
            max_batch_size: Maximum batch size for optimization
        """
        print("=" * 70)
        print("TENSORRT CONVERSION")
        print("=" * 70)

        # Validate inputs
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        if precision not in ["fp32", "fp16", "int8"]:
            raise ValueError(f"Invalid precision: {precision}. Must be fp32, fp16, or int8")

        print(f"\nInput ONNX: {onnx_path}")
        print(f"Output TRT: {output_path}")
        print(f"Precision: {precision}")
        print(f"Max batch size: {max_batch_size}")
        print(f"Workspace: {workspace_size_mb} MB")

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create builder and network
        print("\nInitializing TensorRT builder...")
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        print(f"Parsing ONNX model: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("[ERROR] Failed to parse ONNX model")
                for i in range(parser.num_errors):
                    print(f"  Error {i}: {parser.get_error(i)}")
                raise RuntimeError("ONNX parsing failed")

        print(f"[OK] ONNX model parsed successfully")
        print(f"  Network inputs: {network.num_inputs}")
        print(f"  Network outputs: {network.num_outputs}")

        # Create builder config
        config = builder.create_builder_config()

        # Set workspace size
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size_mb * 1024 * 1024
        )

        # Set precision
        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                print("[WARNING] Platform does not support fast FP16, using FP32")
            else:
                print("[OK] Enabling FP16 mode")
                config.set_flag(trt.BuilderFlag.FP16)

        elif precision == "int8":
            if not builder.platform_has_fast_int8:
                print("[WARNING] Platform does not support fast INT8, using FP32")
            else:
                print("[OK] Enabling INT8 mode")
                config.set_flag(trt.BuilderFlag.INT8)
                print("[WARNING] INT8 requires calibration, proceeding without calibrator")

        # Create optimization profile for dynamic shapes
        print("\nCreating optimization profile...")
        profile = builder.create_optimization_profile()

        # Set dynamic shape ranges for each input
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            print(f"  Input {i}: {input_name}, shape: {input_shape}")

            # Create min, opt, max shapes
            # For dynamic batch: (1, C, H, W), (8, C, H, W), (max_batch_size, C, H, W)
            min_shape = [1] + list(input_shape[1:])
            opt_shape = [8] + list(input_shape[1:])
            max_shape = [max_batch_size] + list(input_shape[1:])

            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"    Min: {min_shape}")
            print(f"    Opt: {opt_shape}")
            print(f"    Max: {max_shape}")

        config.add_optimization_profile(profile)

        # Build engine
        print("\nBuilding TensorRT engine (this may take several minutes)...")
        print("Building with optimizations...")

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        print("[OK] Engine built successfully")

        # Save engine
        print(f"\nSaving engine to: {output_path}")

        # Handle IHostMemory object (TensorRT 10.x)
        if hasattr(serialized_engine, '__len__'):
            engine_bytes = serialized_engine
        else:
            # TensorRT 10.x returns IHostMemory, need to cast to bytes
            engine_bytes = bytes(memoryview(serialized_engine))

        with open(output_path, "wb") as f:
            f.write(engine_bytes)

        engine_size_mb = len(engine_bytes) / (1024 * 1024)
        print(f"[OK] Engine saved ({engine_size_mb:.2f} MB)")

        # Print engine info
        print("\n" + "=" * 70)
        print("CONVERSION COMPLETE")
        print("=" * 70)
        print(f"\nTensorRT Engine: {output_path}")
        print(f"Size: {engine_size_mb:.2f} MB")
        print(f"Precision: {precision}")
        print(f"Max batch size: {max_batch_size}")

        print("\nNext steps:")
        print("1. Create inference wrapper for the engine")
        print("2. Benchmark against PyTorch baseline")
        print("3. Integrate into video analyzer")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output TensorRT engine"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision mode (default: fp16)"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4096,
        help="Workspace size in MB (default: 4096)"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum batch size (default: 16)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    converter = TensorRTConverter(verbose=args.verbose)
    converter.convert(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        workspace_size_mb=args.workspace,
        max_batch_size=args.max_batch_size
    )


if __name__ == "__main__":
    main()
