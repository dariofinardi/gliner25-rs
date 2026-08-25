#!/usr/bin/env python3
# Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0
"""
Derives the FP16 variants of an ONNX export that only ships FP32.

`export_boundary_v1.py` emits all three precisions when it runs from the
PyTorch checkpoint. This script is for the case where you have the FP32 export
and not the checkpoint — the published `jugaadsrl/gliner2.5-multi-v1-onnx`, for
instance, ships FP32 only. It reads `*_fp32.onnx` and writes, beside them:

    *_fp16.onnx             keep_io_types=True  — FP16 weights, FP32 in/out
    *_fp16_iobinding.onnx   keep_io_types=False — FP16 throughout

The two are not interchangeable. `_fp16` casts at every graph boundary, which
CoreML requires and which costs a conversion per fragment. `_fp16_iobinding`
leaves the boundaries in FP16 so a bound chain can hand one fragment's output
straight to the next without touching it — that is what `ExecutionMode::IoBinding`
in `gliner25-rs` needs to pay off.

Usage:

    python downcast_fp16.py <export_dir> [--out DIR] [--only STEM ...]

Runs in place by default. `--out` writes elsewhere and copies the tokenizer and
manifest across, so the result is a directory the engine can load as it stands.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SIDECAR_THRESHOLD = 1.9 * 1024**3  # protobuf caps a single file at 2 GiB


def fix_constant_of_shape(model) -> int:
    """
    Repairs `ConstantOfShape` nodes after FP16 conversion.

    `ConstantOfShape` takes its output type from the `value` attribute, and with
    the attribute absent the spec mandates float32. The converter rewrites the
    output's `value_info` to float16 and leaves the attribute alone, so the
    graph contradicts itself and the session refuses to build:

        Type Error: Type (tensor(float16)) of output arg (val_684) of node
        (node_ConstantOfShape_676) does not match expected type (tensor(float))

    The attribute is materialised here in the type actually declared.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    declared = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(model.graph.value_info) + list(model.graph.output)
    }
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        if declared.get(node.output[0]) != TensorProto.FLOAT16:
            continue
        current = None
        for attr in list(node.attribute):
            if attr.name == "value":
                current = numpy_helper.to_array(attr.t)
                node.attribute.remove(attr)
        scalar = np.float16(0.0) if current is None else current.astype(np.float16).reshape(-1)[0]
        node.attribute.append(
            helper.make_attribute(
                "value", numpy_helper.from_array(np.array([scalar], dtype=np.float16), "value")
            )
        )
        fixed += 1
    return fixed


def convert(fp32_path: Path, keep_io_types: bool, out_path: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    # load_external_data=True is required: a fragment past the protobuf limit
    # keeps its weights in a sidecar, and converting the graph alone would
    # produce a model whose tensors are still FP32 on disk.
    model = onnx.load(str(fp32_path), load_external_data=True)
    model = convert_float_to_float16(model, keep_io_types=keep_io_types)
    fixed = fix_constant_of_shape(model)

    # Half the weights, so a graph that needed a sidecar in FP32 may not in
    # FP16. Decide from the converted size rather than from what the source did.
    size = model.ByteSize()
    external = size > SIDECAR_THRESHOLD
    if external:
        onnx.save(
            model,
            str(out_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=out_path.name + ".data",
            size_threshold=1024,
        )
    else:
        onnx.save(model, str(out_path))

    mb = out_path.stat().st_size / 1e6
    data = out_path.with_name(out_path.name + ".data")
    if data.exists():
        mb += data.stat().st_size / 1e6
    label = "keep_io=FP32" if keep_io_types else "full FP16 I/O"
    notes = []
    if fixed:
        notes.append(f"{fixed} ConstantOfShape repaired")
    if external:
        notes.append("weights in sidecar")
    tail = f"  [{', '.join(notes)}]" if notes else ""
    print(f"    {label:14} -> {out_path.name}  ({mb:.1f} MB){tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write elsewhere (default: in place)")
    ap.add_argument("--only", nargs="*", default=None, help="convert just these stems")
    args = ap.parse_args()

    src: Path = args.export_dir
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 2
    out: Path = args.out or src
    out.mkdir(parents=True, exist_ok=True)

    fragments = sorted(p for p in src.glob("*_fp32.onnx"))
    if not fragments:
        print(f"no *_fp32.onnx in {src}", file=sys.stderr)
        return 2
    if args.only:
        wanted = set(args.only)
        fragments = [p for p in fragments if p.name[: -len("_fp32.onnx")] in wanted]

    print(f"{len(fragments)} fragment(s) in {src}")
    for fp32 in fragments:
        stem = fp32.name[: -len("_fp32.onnx")]
        print(f"  {stem}")
        convert(fp32, True, out / f"{stem}_fp16.onnx")
        convert(fp32, False, out / f"{stem}_fp16_iobinding.onnx")

    if out != src:
        # The engine reads these beside the fragments; without them the
        # converted directory is not loadable on its own.
        for name in ("tokenizer.json", "boundary_manifest.json"):
            s = src / name
            if s.exists():
                shutil.copy2(s, out / name)
                print(f"  copied {name}")

    print(f"\ndone -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
