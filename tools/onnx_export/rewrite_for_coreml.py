"""
Rewrite an existing SIAM v0.3 ONNX so it's compatible with Apple's Core ML
MLProgram format.

Two rewrites are applied, both required for SIAM's 3D U-Net to compile
under the CoreML EP in onnxruntime.

(1) Rank-5 InstanceNormalization -> rank-3
--------------------------------------------
nn.InstanceNorm3d exports as `InstanceNormalization` at rank 5
[N, C, D, H, W]. Core ML MLProgram supports InstanceNorm only at
rank 3 or 4. Pattern applied per node:

    [N, C, D, H, W]                              (original input)
        |
    Shape  ----------> [N, C, D, H, W]           (capture for later)
        |
    Reshape([0, 0, -1])  ------>  [N, C, D*H*W]  (rank-3, Core ML OK)
        |
    InstanceNormalization                        (same scale + bias)
        |
    Reshape(orig_shape) -------->  [N, C, D, H, W]
        |
    (downstream graph)

The math is identical: InstanceNorm reduces (mean + variance) over all
spatial elements (D, H, W) regardless of how they're laid out.

(2) Rank-5 ConvTranspose (kernel == stride, pads = 0) -> PixelShuffle3D
-----------------------------------------------------------------------
SIAM's decoder has six ConvTranspose3D upsamplers with `kernel == stride`
and zero padding. Core ML MLProgram's `conv_transpose` rejects these.
The PixelShuffle3D equivalent uses only ops Core ML supports:

    [N, C_in, D, H, W]
        |
    Conv 1x1x1   (weight = transpose+reshape of original ConvT weight)
        |
    [N, C_out * kD*kH*kW, D, H, W]
        |
    Reshape  --->  [N, C_out, kD, kH, kW, D, H, W]
        |
    Transpose perm=[0,1,5,2,6,3,7,4]
        |
    [N, C_out, D, kD, H, kH, W, kW]
        |
    Reshape  --->  [N, C_out, D*kD, H*kH, W*kW]
        |
    Add bias                                     (broadcast on C axis)
        |
    (downstream graph)

The math identity
    Y[n, oc, d*kD+i, h*kH+j, w*kW+k]
        = B[oc] + sum_ic X[n, ic, d, h, w] * W[ic, oc, i, j, k]
holds bit-for-bit at fp32 and within fp16 rounding when inputs are fp16.
The Core ML siamize models are fp16 fixed-shape [1, 1, 256, 256, 192],
so all Reshape targets are compile-time constants.

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


def _build_shape_map(model: onnx.ModelProto) -> dict:
    """Map every ONNX tensor name to its full shape (list of ints or
    None for symbolic dims), used by the ConvTranspose rewrite which
    needs concrete D/H/W to build static reshape targets."""
    shape = {}
    for init in model.graph.initializer:
        shape[init.name] = list(init.dims)
    for vi in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        if vi.type.tensor_type.HasField("shape"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.HasField("dim_value") else None)
            shape[vi.name] = dims
    return shape


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


def rewrite_conv_transpose_rank5_pixel_shuffle(model: onnx.ModelProto) -> int:
    """Replace every rank-5 ConvTranspose with `kernel == stride`, zero
    pads, unit dilations, and group 1 with a PixelShuffle3D equivalent
    (Conv 1x1x1 + Reshape + Transpose + Reshape + Add).

    Only ConvTranspose nodes whose input shape is fully static (all
    dims known) are rewritten -- the Reshape targets are baked in as
    constant initializers. The CoreML siamize ONNX is fixed-shape, so
    every ConvT in the SIAM v0.3 decoder qualifies.

    Returns the number of nodes rewritten."""
    model_with_shapes = shape_inference.infer_shapes(model)
    shape_map = _build_shape_map(model_with_shapes)
    init_by_name = {i.name: i for i in model.graph.initializer}

    graph = model.graph
    new_nodes = []
    new_initializers = []
    rewrite_count = 0

    for node in graph.node:
        if node.op_type != "ConvTranspose":
            new_nodes.append(node)
            continue

        in_name = node.input[0]
        in_shape = shape_map.get(in_name)

        if in_shape is None or len(in_shape) != 5 or any(d is None for d in in_shape):
            new_nodes.append(node)
            continue

        attrs = {a.name: a for a in node.attribute}
        kernel_shape = (
            list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else None
        )
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1, 1]
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0] * 6
        dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1, 1]
        group = attrs["group"].i if "group" in attrs else 1
        output_padding = (
            list(attrs["output_padding"].ints) if "output_padding" in attrs else [0] * 3
        )

        # Only rewrite the PixelShuffle-equivalent pattern.
        if (
            kernel_shape is None
            or kernel_shape != strides
            or any(p != 0 for p in pads)
            or any(d != 1 for d in dilations)
            or any(op != 0 for op in output_padding)
            or group != 1
        ):
            print(
                f"  skipping ConvTranspose '{node.name or '?'}': "
                f"kernel={kernel_shape} strides={strides} pads={pads} "
                f"dilations={dilations} group={group} output_padding={output_padding} "
                f"-- not PixelShuffle-equivalent",
                file=sys.stderr,
            )
            new_nodes.append(node)
            continue

        weight_init = init_by_name.get(node.input[1])
        if weight_init is None:
            print(
                f"  warning: ConvTranspose '{node.name or '?'}' weight "
                f"'{node.input[1]}' is not a constant initializer; skipping",
                file=sys.stderr,
            )
            new_nodes.append(node)
            continue
        bias_init = init_by_name.get(node.input[2]) if len(node.input) > 2 else None

        N, C_in, D, H, W = (int(d) for d in in_shape)
        kD, kH, kW = kernel_shape
        W_arr = numpy_helper.to_array(weight_init)
        # ConvTranspose weight layout: [C_in, C_out, kD, kH, kW] (group=1)
        if W_arr.ndim != 5 or W_arr.shape[0] != C_in:
            print(
                f"  warning: ConvTranspose '{node.name or '?'}' weight "
                f"shape {W_arr.shape} doesn't match expected "
                f"[{C_in}, C_out, {kD}, {kH}, {kW}]; skipping",
                file=sys.stderr,
            )
            new_nodes.append(node)
            continue
        C_out = W_arr.shape[1]

        # Rebuild as a Conv 1x1x1 weight of shape [C_out*kD*kH*kW, C_in, 1, 1, 1]
        # such that channel index `oc * (kD*kH*kW) + (i*kH*kW + j*kW + k)`
        # of the Conv output equals sum_ic X[ic] * W[ic, oc, i, j, k].
        W_new = (
            W_arr.transpose(1, 2, 3, 4, 0)
            .reshape(C_out * kD * kH * kW, C_in, 1, 1, 1)
            .copy()
        )

        base = node.name or in_name
        conv_w_name = f"{base}__pxshuf_conv_w"
        new_initializers.append(numpy_helper.from_array(W_new, name=conv_w_name))

        # 1. Conv 1x1x1 (no bias -- bias is applied after the shuffle).
        conv_out = f"{base}__pxshuf_conv_out"
        new_nodes.append(
            helper.make_node(
                "Conv",
                inputs=[in_name, conv_w_name],
                outputs=[conv_out],
                name=f"{base}__pxshuf_conv",
                kernel_shape=[1, 1, 1],
                pads=[0] * 6,
                strides=[1, 1, 1],
                dilations=[1, 1, 1],
                group=1,
            )
        )

        # 2. Reshape to [N, C_out, kD, kH, kW, D, H, W].
        re1_target_name = f"{base}__pxshuf_re1_target"
        new_initializers.append(
            numpy_helper.from_array(
                np.array([N, C_out, kD, kH, kW, D, H, W], dtype=np.int64),
                name=re1_target_name,
            )
        )
        re1_out = f"{base}__pxshuf_re1_out"
        new_nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[conv_out, re1_target_name],
                outputs=[re1_out],
                name=f"{base}__pxshuf_re1",
            )
        )

        # 3. Transpose perm=[0,1,5,2,6,3,7,4] -> [N, C_out, D, kD, H, kH, W, kW].
        tr_out = f"{base}__pxshuf_tr_out"
        new_nodes.append(
            helper.make_node(
                "Transpose",
                inputs=[re1_out],
                outputs=[tr_out],
                name=f"{base}__pxshuf_tr",
                perm=[0, 1, 5, 2, 6, 3, 7, 4],
            )
        )

        # 4. Reshape to [N, C_out, D*kD, H*kH, W*kW].
        re2_target_name = f"{base}__pxshuf_re2_target"
        new_initializers.append(
            numpy_helper.from_array(
                np.array([N, C_out, D * kD, H * kH, W * kW], dtype=np.int64),
                name=re2_target_name,
            )
        )
        # If there's no bias to add, the final Reshape produces the original
        # output name directly; otherwise an Add tail finishes the chain.
        re2_out_name = (
            list(node.output)[0] if bias_init is None else f"{base}__pxshuf_re2_out"
        )
        new_nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[tr_out, re2_target_name],
                outputs=[re2_out_name],
                name=f"{base}__pxshuf_re2",
            )
        )

        # 5. Add bias broadcast on C axis. Reshape to [1, C_out, 1, 1, 1]
        #    so the broadcast is unambiguous at rank 5.
        if bias_init is not None:
            B_arr = numpy_helper.to_array(bias_init).reshape(1, C_out, 1, 1, 1).copy()
            bias_name = f"{base}__pxshuf_bias"
            new_initializers.append(numpy_helper.from_array(B_arr, name=bias_name))
            new_nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[re2_out_name, bias_name],
                    outputs=list(node.output),
                    name=f"{base}__pxshuf_add_bias",
                )
            )

        rewrite_count += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)

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
    n_in = rewrite_instance_norm_rank5(model)
    print(f"  rewrote {n_in} InstanceNorm node(s)")

    print("rewriting rank-5 ConvTranspose -> PixelShuffle3D ...")
    n_ct = rewrite_conv_transpose_rank5_pixel_shuffle(model)
    print(f"  rewrote {n_ct} ConvTranspose node(s)")

    if args.check:
        print("validating with onnx.checker ...")
        onnx.checker.check_model(model)

    onnx.save(model, args.output)
    print(f"saved {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")

    if n_in == 0 and n_ct == 0:
        print(
            "note: no rank-5 InstanceNorm or ConvTranspose nodes were found. "
            "Input model may already be CoreML-compatible, OR shape inference "
            "couldn't annotate ranks. Try re-running on the fp32 (pre-cast) "
            "ONNX where dynamic_axes are present.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
