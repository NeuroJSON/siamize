"""
Rewrite an existing SIAM v0.3 ONNX so it's compatible with Apple's Core ML
MLProgram format.

The blocker: nn.InstanceNorm3d (used throughout SIAM's encoder/decoder)
exports as an `InstanceNormalization` node with rank-5 input
[N, C, D, H, W]. Core ML's MLProgram format accepts InstanceNorm only at
rank 3 or 4. ORT 1.26's CoreML EP doesn't partition the unsupported op
out -- it sends the whole graph to Apple's mlcompilerd, which rejects
the compile with:

    "Unable to parse ML Program: in operation .../InstanceNormalization:
    parameter x[0] has invalid rank 5, expected [3, 4]."

The rewrite this script applies, per InstanceNormalization node with
rank-5 input:

    [N, C, D, H, W]            (original input)
        |
    Shape  ----------> [N, C, D, H, W]  (capture for later)
        |
    Reshape([0, 0, -1])  ------>  [N, C, D*H*W]   (rank-3, Core ML OK)
        |
    InstanceNormalization              (same scale + bias from original)
        |
    Reshape(orig_shape) -------->  [N, C, D, H, W]
        |
    (downstream graph)

The math is identical: InstanceNorm reduces (mean + variance) over all
spatial elements (D, H, W) regardless of how they're laid out. Scale
and bias weights pass through unchanged.

Usage:
    python rewrite_for_coreml.py in.onnx out.onnx [--check]

The `--check` flag re-runs `onnx.checker.check_model` after the rewrite
to validate the resulting graph is well-formed ONNX.

To produce a fp16 variant for Apple Silicon's ANE (which is fp16-native),
combine this with onnxconverter-common's float16.convert_float_to_float16:

    python rewrite_for_coreml.py fold_0_fp32.onnx fold_0_fp32_coreml.onnx
    python -c "
        import onnx
        from onnxconverter_common import float16
        m = onnx.load('fold_0_fp32_coreml.onnx')
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m16, 'fold_0_fp16_coreml.onnx')
    "
"""

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _build_rank_map(model: onnx.ModelProto) -> dict:
    """Map every ONNX tensor name to its rank (number of dims), as inferred
    from initializers, graph inputs/outputs, and shape-inferred value_info."""
    rank = {}

    for init in model.graph.initializer:
        rank[init.name] = len(init.dims)

    for vi in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        if vi.type.tensor_type.HasField("shape"):
            rank[vi.name] = len(vi.type.tensor_type.shape.dim)

    return rank


def rewrite_instance_norm_rank5(model: onnx.ModelProto) -> int:
    """Replace every rank-5 InstanceNormalization node in `model` (in place)
    with a Shape + Reshape + InstanceNorm(rank-3) + Reshape sequence.

    Returns the number of nodes rewritten."""
    # Shape inference is required to know each tensor's rank. Without
    # it, value_info entries for intermediate tensors are missing and
    # we can't tell whether an InstanceNorm input is rank-4 (already OK)
    # or rank-5 (needs rewrite).
    model_with_shapes = shape_inference.infer_shapes(model)
    rank = _build_rank_map(model_with_shapes)

    graph = model.graph
    new_nodes = []
    rewrite_count = 0

    for node in graph.node:
        if node.op_type != "InstanceNormalization":
            new_nodes.append(node)
            continue

        in_name = node.input[0]
        in_rank = rank.get(in_name)

        if in_rank is None:
            print(
                f"  warning: could not determine rank of {in_name}; "
                f"leaving InstanceNorm '{node.name or '?'}' as-is",
                file=sys.stderr,
            )
            new_nodes.append(node)
            continue

        if in_rank <= 4:
            # Already CoreML-compatible.
            new_nodes.append(node)
            continue

        if in_rank != 5:
            print(
                f"  warning: InstanceNorm '{node.name or '?'}' has rank "
                f"{in_rank}; rewrite only handles rank 5",
                file=sys.stderr,
            )
            new_nodes.append(node)
            continue

        # ---- emit the rewrite ----
        base = node.name or in_name
        # Per-node unique names so the rewrite is safe to apply when
        # the model has many InstanceNorm layers (SIAM has ~30+).
        shape_out = f"{base}__orig_shape"
        target_shape_name = f"{base}__pre_reshape_target"
        pre_reshape_out = f"{base}__pre_reshaped"
        in_out = f"{base}__in_out"

        # Constant initializer [0, 0, -1] for the pre-reshape:
        # 0 = copy from input (keeps N and C), -1 = infer (collapses D*H*W).
        pre_target = numpy_helper.from_array(
            np.array([0, 0, -1], dtype=np.int64), name=target_shape_name
        )
        graph.initializer.append(pre_target)

        # 1. Capture original shape so we can restore it after the IN.
        shape_node = helper.make_node(
            "Shape",
            inputs=[in_name],
            outputs=[shape_out],
            name=f"{base}__shape",
        )
        new_nodes.append(shape_node)

        # 2. Pre-reshape: [N, C, D, H, W] -> [N, C, D*H*W]
        pre_reshape_node = helper.make_node(
            "Reshape",
            inputs=[in_name, target_shape_name],
            outputs=[pre_reshape_out],
            name=f"{base}__pre_reshape",
        )
        new_nodes.append(pre_reshape_node)

        # 3. InstanceNormalization (rank-3 now). Preserve attributes
        #    (specifically `epsilon`) from the original node.
        in_attrs = list(node.attribute)
        # The scale and bias inputs (node.input[1], node.input[2]) are
        # shape [C] and pass through unchanged. Output name is internal;
        # the post-reshape will re-use the original node's output name.
        new_in = helper.make_node(
            "InstanceNormalization",
            inputs=[pre_reshape_out, node.input[1], node.input[2]],
            outputs=[in_out],
            name=base,  # same name as original so downstream stays referenceable
        )
        new_in.attribute.extend(in_attrs)
        new_nodes.append(new_in)

        # 4. Post-reshape: back to [N, C, D, H, W].
        post_reshape_node = helper.make_node(
            "Reshape",
            inputs=[in_out, shape_out],
            outputs=list(node.output),  # keep the original output name
            name=f"{base}__post_reshape",
        )
        new_nodes.append(post_reshape_node)

        rewrite_count += 1

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Bump opset to >=11 to be sure Shape + Reshape with allowzero are
    # supported per spec. Opset 17 (siamize's default) is well above
    # this floor.
    return rewrite_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to source ONNX (typically fold_N_fp32.onnx)")
    ap.add_argument("output", help="path to write CoreML-friendly rewritten ONNX")
    ap.add_argument(
        "--check",
        action="store_true",
        help="run onnx.checker.check_model on the rewritten graph",
    )
    args = ap.parse_args()

    print(f"loading {args.input}", flush=True)
    model = onnx.load(args.input)

    print("rewriting rank-5 InstanceNormalization -> rank-3 ...")
    n = rewrite_instance_norm_rank5(model)
    print(f"  rewrote {n} InstanceNorm node(s)")

    if args.check:
        print("validating with onnx.checker ...")
        onnx.checker.check_model(model)

    onnx.save(model, args.output)
    print(f"saved {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")

    if n == 0:
        print(
            "note: no rank-5 InstanceNorm nodes were found. Input model "
            "may already be CoreML-compatible, OR shape inference couldn't "
            "annotate ranks. Try re-running on the fp32 (pre-cast) ONNX "
            "where dynamic_axes are present.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
