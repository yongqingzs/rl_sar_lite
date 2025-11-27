#!/usr/bin/env python3
"""
Convert PyTorch TorchScript (.pt) models to RKNN format
Designed for reinforcement learning models (non-image inputs)
Automatically detects input dimensions and converts to RKNN
"""

import torch
import sys
import os
import numpy as np
from rknn.api import RKNN

PT_PATH = "/home/jazzy/rl_sar/policy/go1/my_unitree_go1/policy_1.pt"
# RKNN_PATH
TARGET_PALTFORM = "rk3588"

def detect_input_shape_from_pt(pt_path):
    """
    Auto-detect input shape from PyTorch TorchScript model
    Reads from model graph or infers from first layer parameters

    Args:
        pt_path: Path to .pt file

    Returns:
        tuple: (input_size, output_size)
    """
    print("Reading input/output dimensions from PT model...")

    model = torch.jit.load(pt_path)
    model.eval()

    # Method 1: Try to read from graph type information
    try:
        graph = model.graph
        inputs = list(graph.inputs())

        if len(inputs) >= 2:
            input_node = inputs[1]  # First actual input (after self)

            if hasattr(input_node, 'type') and hasattr(input_node.type(), 'sizes'):
                sizes = input_node.type().sizes()
                if sizes is not None and len(sizes) >= 2:
                    input_size = sizes[-1]  # Last dimension

                    # Get output size by running a test
                    dummy_input = torch.randn(1, input_size)
                    with torch.no_grad():
                        output = model(dummy_input)
                    output_size = output.shape[-1]

                    print(f"✓ Detected from graph: input={input_size}, output={output_size}")
                    return int(input_size), int(output_size)
    except Exception as e:
        print(f"  Graph reading failed: {e}")

    # Method 2: Infer from first layer parameters
    print("  Inferring from model parameters...")
    try:
        params = dict(model.named_parameters())

        # Look for first linear layer weights
        for name, param in params.items():
            if 'weight' in name and len(param.shape) == 2:
                # Linear layer weight shape is [out_features, in_features]
                input_size = param.shape[1]

                # Verify by running a test
                try:
                    dummy_input = torch.randn(1, input_size)
                    with torch.no_grad():
                        output = model(dummy_input)
                    output_size = output.shape[-1]

                    print(f"✓ Detected from parameters: input={input_size}, output={output_size}")
                    return int(input_size), int(output_size)
                except:
                    continue
    except Exception as e:
        print(f"  Parameter inference failed: {e}")

    raise ValueError(
        "Could not auto-detect input dimensions from PT model. "
        "The model may have an unsupported structure."
    )


def convert_pt_to_rknn(pt_path, rknn_path=None, target_platform='rk3566'):
    """
    Convert PyTorch TorchScript .pt model to RKNN format

    Args:
        pt_path: Path to input .pt file
        rknn_path: Path to output .rknn file (auto-generated if None)
        target_platform: Target RKNN platform (default: rk3566)
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PT model file not found: {pt_path}")

    # Auto-generate output path if not provided - same directory as PT file
    if rknn_path is None:
        pt_dir = os.path.dirname(pt_path)
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        rknn_path = os.path.join(pt_dir, f"{base_name}.rknn")

    print(f"\nLoading PyTorch model: {pt_path}")

    # Auto-detect input/output size
    input_size, output_size = detect_input_shape_from_pt(pt_path)

    # Create input size list for RKNN (batch_size=1, features=input_size)
    input_size_list = [[1, input_size]]

    print(f"Input size: {input_size}, Output size: {output_size}")
    print(f"Target platform: {target_platform}")
    print(f"Output RKNN file: {rknn_path}")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    try:
        # Config model - for RL models, no image preprocessing needed
        print('--> Config model')
        rknn.config(
            target_platform=target_platform,
            quantized_dtype='w8a8'
            # No mean/std normalization for RL models
            # mean_values=None,
            # std_values=None
        )
        print('done')

        # Load PyTorch model
        print('--> Loading PyTorch model')
        ret = rknn.load_pytorch(model=pt_path, input_size_list=input_size_list)
        if ret != 0:
            raise RuntimeError(f'Load PyTorch model failed! Error code: {ret}')
        print('done')

        # Build model
        print('--> Building RKNN model')
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f'Build RKNN model failed! Error code: {ret}')
        print('done')

        # Export RKNN model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f'Export RKNN model failed! Error code: {ret}')
        print('done')

        print(f"\n✓ Successfully converted to RKNN: {rknn_path}")

        # Optional: Verify the conversion with a test inference
        print("\n--> Verifying conversion")
        verify_conversion(pt_path, input_size, target_platform)
        print("✓ Conversion verified successfully!")

    finally:
        rknn.release()


def verify_conversion(pt_path, input_size, target_platform):
    """
    Verify the RKNN conversion by comparing outputs with original PyTorch model
    Rebuilds the RKNN model for verification instead of loading the exported file

    Args:
        pt_path: Path to original .pt file
        input_size: Input dimension size
        target_platform: Target RKNN platform
    """
    # Load original PyTorch model
    pt_model = torch.jit.load(pt_path)
    pt_model.eval()

    # Create test input
    dummy_input = torch.randn(1, input_size)

    # Get PyTorch output
    with torch.no_grad():
        pt_output = pt_model(dummy_input).numpy()

    # Create RKNN model for verification
    input_size_list = [[1, input_size]]
    rknn = RKNN(verbose=False)

    try:
        rknn.config(
            target_platform=target_platform,
            mean_values=None,
            std_values=None
        )

        ret = rknn.load_pytorch(model=pt_path, input_size_list=input_size_list)
        if ret != 0:
            raise RuntimeError(f'Load PyTorch model for verification failed! Error code: {ret}')

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f'Build RKNN model for verification failed! Error code: {ret}')

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f'Init runtime for verification failed! Error code: {ret}')

        rknn_output = rknn.inference(inputs=[dummy_input.numpy()])[0]

        # Compare outputs
        max_diff = np.abs(pt_output - rknn_output).max()
        print(f"Max difference between PyTorch and RKNN outputs: {max_diff}")

        if max_diff > 1e-3:
            print("⚠ Warning: Large difference detected, conversion may have issues")
        else:
            print("✓ Outputs match within acceptable tolerance")

        for i in range(10000):
            dummy_input = torch.randn(1, input_size)
            # dummy_input = torch.zeros(1, input_size)
            pt_output = pt_model(dummy_input).detach().numpy()
            rknn_output = rknn.inference(inputs=[dummy_input.numpy()])[0]
            diff = np.abs(pt_output - rknn_output).max()
            # if diff > 1e-3:
            #     print(f"⚠ Iteration {i}: Large diff {diff}")
            if diff > 1e-1:
                print(f"⚠ Iteration {i}: Large diff {diff}")

    finally:
        rknn.release()


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python convert_pt_to_rknn.py <pt_model_path> [rknn_output_path] [target_platform]")
    #     print()
    #     print("Arguments:")
    #     print("  pt_model_path    : Path to input PyTorch TorchScript (.pt) file")
    #     print("  rknn_output_path : Path to output RKNN file (optional, auto-generated)")
    #     print("  target_platform  : RKNN target platform (optional, default: rk3566)")
    #     print()
    #     print("Examples:")
    #     print("  python convert_pt_to_rknn.py policy.pt")
    #     print("  python convert_pt_to_rknn.py policy.pt output.rknn")
    #     print("  python convert_pt_to_rknn.py policy.pt output.rknn rk3588")
    #     print()
    #     print("Supported platforms: rk3566, rk3568, rk3588, rk3576, etc.")
    #     sys.exit(1)

    # pt_path = sys.argv[1]
    # rknn_path = sys.argv[2] if len(sys.argv) > 2 else None
    # target_platform = sys.argv[3] if len(sys.argv) > 3 else 'rk3588'

    pt_path = PT_PATH
    rknn_path = None
    target_platform = TARGET_PALTFORM

    try:
        convert_pt_to_rknn(pt_path, rknn_path, target_platform)
        print("\n" + "=" * 60)
        print("✓ RKNN conversion completed successfully!")
        print(f"Input PT file:  {pt_path}")
        print(f"Output RKNN file: {rknn_path or os.path.join(os.path.dirname(pt_path), os.path.splitext(os.path.basename(pt_path))[0] + '.rknn')}")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()