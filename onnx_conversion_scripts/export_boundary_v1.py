"""
GLiNER2.5 *boundary* ONNX Exporter v1
=====================================
Exports a checkpoint with `"architecture": "boundary"` (BoundaryExtractor,
e.g. `gliner2.5-multi-v1`) into ONNX fragments runnable by the `gliner25` crate.

The boundary architecture has nothing in common with the span one: no
`span_rep`, no `count_lstm`, no exhaustive span enumeration. Instead:

    encoder (mDeBERTa-v3)
        -> token_embeddings [1, seq_len, H]
        |
        +- routed_gather(indices, mask)   (token_pooling="first")
        |     -> text_states  [1, L, H]     one row per word of the text
        |     -> query_states [1, Q, H]     one row per [E]/[C]/[R] marker
        |     -> cls_states   [1, K, H]     one row per classification choice
        |
        +- boundary_head(text_states, text_mask, query_states, query_mask)
        |     -> cand_indices    [1, Q, C, 2]  HALF-OPEN (start, end) pairs
        |     -> pair_logits     [1, Q, C]     query x candidate logits
        |     -> cand_valid      [1, Q, C]
        |     -> null_logits     [1, Q]        per-query abstention
        |     -> count_log_rates [1, Q]        expected count per query
        |
        +- classifier(cls_states) -> logits [K]

C is constant (`pool_size`, typically 192): the candidate pool is shared across
queries and has a fixed size. Decoding - sigmoid, per-query threshold, overlap
policy, ranking - is left to the caller.


Why the length buckets
----------------------
`torch.export` specialises `num_words` to a constant: the candidate-pool
builder contains a Python loop over a symbolic dimension, and no `export_mode`
removes it on this branch. It is not something a caller can work around.

The boundary head weighs under 1 MB, though (the encoder is 99.9% of the
parameters), so one copy per length bucket is exported and the runtime picks
the smallest bucket that fits the text, padding with `text_mask=0`.

Masked padding is verified to be equivalent: for the same real words, padding
to a larger bucket - even with noise in the padded rows - yields the **exact
same candidate set** and logits to within ~5e-06. See `verify_parity.py`.

A welcome side effect: static shapes in L and C make the graph ideal for
IOBinding, TensorRT and QNN, all of which degrade with dynamic shapes.

The smallest bucket is 32 because `select_top_boundaries` computes
`k = min(pool_boundary_top_k, n_boundaries)`: below that threshold the graph
would be traced with a reduced `k` and would no longer hold for longer texts.


Stable sort
-----------
The pool uses `torch.sort(..., stable=True)`; ONNX has no stable Sort and the
`aten.sort.stable` operator has no translation. `stable=True` is stripped
during export. Exact score ties happen in practice only between positions
masked to `MASK_LOGIT`, which `cand_valid` discards anyway; measured parity
against PyTorch is ~1e-05.


Usage:
    python export_boundary_v1.py \
        --model_path fastino/gliner2.5-multi-v1 \
        --out_dir models/gliner2.5-multi-v1-onnx

    # custom buckets
    python export_boundary_v1.py --model_path ... --out_dir ... \
        --buckets 64,128,256,512
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from gliner2 import AutoExtractor

DEFAULT_BUCKETS = (64, 128, 256, 512)
MIN_BUCKET = 32
OPSET = 18  # the dynamo path cannot downgrade to 17 (Squeeze axes-as-input)


# ─────────────────────────────────────────────────────────────────────────────
# Export-time patches
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _unstable_sort():
    """Strips `stable=True` from torch.sort/argsort for the duration of the export."""
    _sort, _argsort = torch.sort, torch.argsort

    def sort(*args, **kwargs):
        kwargs.pop("stable", None)
        return _sort(*args, **kwargs)

    def argsort(*args, **kwargs):
        kwargs.pop("stable", None)
        return _argsort(*args, **kwargs)

    torch.sort, torch.argsort = sort, argsort
    try:
        yield
    finally:
        torch.sort, torch.argsort = _sort, _argsort


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 1 – Encoder
# ─────────────────────────────────────────────────────────────────────────────
class EncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,        # [1, seq_len]  int64
        attention_mask: torch.Tensor,   # [1, seq_len]  int64
    ) -> torch.Tensor:                  # [1, seq_len, H]
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 2 – RoutedGather
# Mirrors `gather_routed` in BoundaryExtractorModel._encode_core: takes the
# first sub-token of each slot (token_pooling="first") and zeroes the masked
# rows. A single graph serves text / query / cls.
# ─────────────────────────────────────────────────────────────────────────────
class RoutedGatherWrapper(nn.Module):
    def forward(
        self,
        hidden_state: torch.Tensor,  # [1, seq_len, H]
        indices: torch.Tensor,       # [1, S]  int64
        mask: torch.Tensor,          # [1, S]  int64 (0/1)
    ) -> torch.Tensor:               # [1, S, H]
        h = hidden_state.shape[-1]
        safe = indices.clamp(0, hidden_state.shape[1] - 1)
        states = hidden_state.gather(1, safe.unsqueeze(-1).expand(-1, -1, h))
        return states * mask.unsqueeze(-1).to(states.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 3 – BoundaryHead
# Returns flat tensors instead of ExtractorOutput/CandidateTensorBatch.
# ─────────────────────────────────────────────────────────────────────────────
class BoundaryHeadWrapper(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(
        self,
        text_states: torch.Tensor,   # [1, L, H]
        text_mask: torch.Tensor,     # [1, L]  int64
        query_states: torch.Tensor,  # [1, Q, H]
        query_mask: torch.Tensor,    # [1, Q]  int64
    ):
        out = self.head(
            text_states, text_mask.bool(),
            query_states, query_mask.bool(),
            targets=None,
            return_candidates=True,
            collect_diagnostics=False,
        )
        c = out.candidates
        return (
            c.indices,        # [1, Q, C, 2]  int64
            c.pair_logits,    # [1, Q, C]
            c.valid_mask,     # [1, Q, C]  bool
            out.null_logits,       # [1, Q]
            out.count_log_rates,   # [1, Q]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 4 – Classifier
# ─────────────────────────────────────────────────────────────────────────────
class ClassifierWrapper(nn.Module):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, choice_states: torch.Tensor) -> torch.Tensor:
        # [K, H] -> [K]
        return self.classifier(choice_states).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def _export(
    module: nn.Module,
    args: tuple,
    out_path: Path,
    input_names: list,
    output_names: list,
    *,
    dynamic_axes: dict | None = None,
    dynamic_shapes: dict | None = None,
    dynamo: bool = False,
) -> None:
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        dynamo=dynamo,
    )
    if dynamo:
        kwargs["dynamic_shapes"] = dynamic_shapes
    else:
        kwargs["dynamic_axes"] = dynamic_axes
    with torch.no_grad():
        torch.onnx.export(module, args, str(out_path), **kwargs)
    print(f"    FP32 -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def _fix_constant_of_shape(model) -> int:
    """
    Repairs `ConstantOfShape` nodes after FP16 conversion.

    `ConstantOfShape` derives its output type from the `value` attribute; when
    the attribute is absent the ONNX spec mandates `float32`. The FP16 converter
    rewrites the output's `value_info` declaring it `float16` but leaves the
    attribute alone, and the graph becomes inconsistent::

        Type Error: Type (tensor(float16)) of output arg (val_684) of node
        (node_ConstantOfShape_676) does not match expected type (tensor(float))

    Here the attribute is materialised in the type actually declared.
    """
    import numpy as np
    import onnx
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


def _convert_fp16(fp32_path: Path, keep_io_types: bool, out_path: Path) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(fp32_path))
    model = convert_float_to_float16(model, keep_io_types=keep_io_types)
    fixed = _fix_constant_of_shape(model)
    onnx.save(model, str(out_path))
    label = "fp16 (keep_io=FP32)" if keep_io_types else "fp16 (full FP16 IO)"
    note = f", {fixed} ConstantOfShape repaired" if fixed else ""
    print(f"    {label} -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB){note}")
    return out_path


def _both_fp16(fp32_path: Path, out_dir: Path, stem: str) -> None:
    _convert_fp16(fp32_path, True, out_dir / f"{stem}_fp16.onnx")
    _convert_fp16(fp32_path, False, out_dir / f"{stem}_fp16_iobinding.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────────────────────
def export_boundary(model_path: str, out_dir: Path, buckets: tuple[int, ...]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("GLiNER2.5 boundary ONNX Exporter v1")
    print("=" * 64)
    print(f"Model   : {model_path}")
    print(f"Output  : {out_dir}")
    print(f"Buckets : {list(buckets)}")
    print()

    bad = [b for b in buckets if b < MIN_BUCKET]
    if bad:
        raise SystemExit(
            f"ERROR: buckets {bad} are below the minimum of {MIN_BUCKET}.\n"
            f"  select_top_boundaries uses k=min(pool_boundary_top_k, n_boundaries):\n"
            f"  below {MIN_BUCKET} words the graph would be traced with a reduced k."
        )

    print("Loading AutoExtractor...")
    model = AutoExtractor.from_pretrained(model_path)
    model.eval()

    arch = getattr(model, "architecture", None) or type(model).__name__
    if getattr(model, "boundary_head", None) is None:
        raise SystemExit(
            f"ERROR: {model_path} is not a boundary model (architecture={arch}).\n"
            f"  Span checkpoints belong to the gliner2-rs crate instead."
        )

    head = model.boundary_head
    head.eval()
    head.collect_diagnostics = False

    settings = model.boundary_settings
    H = model.encoder.config.hidden_size
    pool_size = int(settings.pool_size)

    print(f"hidden_size      = {H}")
    print(f"candidate_pool   = {settings.candidate_pool}")
    print(f"pool_size (C)    = {pool_size}")
    print(f"pool_top_k       = {settings.pool_boundary_top_k}")
    print(f"abstention       = {settings.enable_abstention}")
    print(f"count_head       = {settings.enable_count_head}")
    print(f"overlap_policy   = {settings.overlap_policy}")
    print()

    if settings.candidate_pool != "shared":
        raise SystemExit(
            f"ERROR: candidate_pool='{settings.candidate_pool}' is not supported.\n"
            f"  Only the 'shared' branch produces a constant-size pool."
        )

    SEQ, Q, K = 64, 4, 5

    # ═════════════════════════════════════════════════════════════════════════
    # 1. ENCODER  (dynamic in seq_len)
    # ═════════════════════════════════════════════════════════════════════════
    print("--- 1. encoder ---")
    enc32 = out_dir / "encoder_fp32.onnx"
    _export(
        EncoderWrapper(model.encoder),
        (torch.randint(0, 1000, (1, SEQ)), torch.ones((1, SEQ), dtype=torch.long)),
        enc32,
        ["input_ids", "attention_mask"],
        ["last_hidden_state"],
        dynamic_axes={
            "input_ids":         {0: "batch", 1: "seq_len"},
            "attention_mask":    {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
    )
    _both_fp16(enc32, out_dir, "encoder")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 2. ROUTED GATHER  (dynamic; serves text / query / cls)
    # ═════════════════════════════════════════════════════════════════════════
    print("--- 2. routed_gather ---")
    rg32 = out_dir / "routed_gather_fp32.onnx"
    _export(
        RoutedGatherWrapper(),
        (
            torch.randn(1, SEQ, H),
            torch.randint(0, SEQ, (1, 20)),
            torch.ones(1, 20, dtype=torch.long),
        ),
        rg32,
        ["last_hidden_state", "indices", "mask"],
        ["states"],
        dynamic_axes={
            "last_hidden_state": {1: "seq_len"},
            "indices":           {1: "num_slots"},
            "mask":              {1: "num_slots"},
            "states":            {1: "num_slots"},
        },
    )
    _both_fp16(rg32, out_dir, "routed_gather")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 3. BOUNDARY HEAD  (one graph per bucket; num_queries dynamic)
    # ═════════════════════════════════════════════════════════════════════════
    print("--- 3. boundary_head (per bucket) ---")
    nq = torch.export.Dim("num_queries", min=1, max=256)
    for L in buckets:
        stem = f"boundary_head_L{L}"
        print(f"  bucket L={L}")
        fp32 = out_dir / f"{stem}_fp32.onnx"
        with _unstable_sort():
            _export(
                BoundaryHeadWrapper(head),
                (
                    torch.randn(1, L, H),
                    torch.ones(1, L, dtype=torch.long),
                    torch.randn(1, Q, H),
                    torch.ones(1, Q, dtype=torch.long),
                ),
                fp32,
                ["text_states", "text_mask", "query_states", "query_mask"],
                ["cand_indices", "pair_logits", "cand_valid",
                 "null_logits", "count_log_rates"],
                dynamic_shapes={
                    "text_states":  {1: L},
                    "text_mask":    {1: L},
                    "query_states": {1: nq},
                    "query_mask":   {1: nq},
                },
                dynamo=True,
            )
        _both_fp16(fp32, out_dir, stem)
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 4. CLASSIFIER
    # ═════════════════════════════════════════════════════════════════════════
    print("--- 4. classifier ---")
    cls32 = out_dir / "classifier_fp32.onnx"
    _export(
        ClassifierWrapper(model.classifier),
        (torch.randn(K, H),),
        cls32,
        ["choice_states"],
        ["logits"],
        dynamic_axes={"choice_states": {0: "num_choices"}, "logits": {0: "num_choices"}},
    )
    _both_fp16(cls32, out_dir, "classifier")
    print()

    _copy_tokenizer(model_path, out_dir)
    _write_manifest(out_dir, model, settings, H, buckets, pool_size)
    _print_summary(out_dir, buckets)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest consumed by the Rust runtime
# ─────────────────────────────────────────────────────────────────────────────
def _write_manifest(
    out_dir: Path,
    model,
    settings,
    hidden_size: int,
    buckets: tuple[int, ...],
    pool_size: int,
) -> None:
    manifest = {
        "architecture": "boundary",
        "exporter": "export_boundary_v1.py",
        "opset": OPSET,
        "hidden_size": hidden_size,
        "pool_size": pool_size,
        "pool_boundary_top_k": int(settings.pool_boundary_top_k),
        "length_buckets": list(buckets),
        "min_bucket": MIN_BUCKET,
        "enable_abstention": bool(settings.enable_abstention),
        "enable_count_head": bool(settings.enable_count_head),
        "enable_relations": bool(settings.enable_relations),
        "enable_records": bool(settings.enable_records),
        "overlap_policy": str(settings.overlap_policy),
        "token_pooling": str(getattr(model.processor, "token_pooling", "first")),
        "max_position_embeddings": int(model.encoder.config.max_position_embeddings),
        "notes": {
            "padding": "pad the text to the smallest bucket that fits it, "
                       "with text_mask=0 on the padded rows",
            "stable_sort": "stable=True stripped at export; measured parity ~1e-05",
            "decoding": "sigmoid(pair_logits) -> per-query threshold -> overlap_policy -> stable ranking",
        },
    }
    (out_dir / "boundary_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"boundary_manifest.json written ({len(buckets)} buckets)")


def _copy_tokenizer(model_path: str, out_dir: Path) -> None:
    src = Path(model_path) / "tokenizer.json"
    if src.exists():
        shutil.copy(src, out_dir / "tokenizer.json")
        print(f"tokenizer.json copied from {model_path}")
        return
    try:
        from huggingface_hub import hf_hub_download

        shutil.copy(hf_hub_download(model_path, "tokenizer.json"), out_dir / "tokenizer.json")
        print(f"tokenizer.json downloaded from the HuggingFace Hub: {model_path}")
    except Exception as e:
        print(f"WARN: could not copy tokenizer.json: {e}")


def _print_summary(out_dir: Path, buckets: tuple[int, ...]) -> None:
    print("=" * 64)
    print("Boundary export v1 complete")
    print()
    total = sum(f.stat().st_size for f in out_dir.glob("*.onnx"))
    print(f"{len(list(out_dir.glob('*.onnx')))} ONNX files, {total / 1e9:.2f} GB total")
    print()
    print("Execution chain:")
    print("  encoder(input_ids, attention_mask) -> last_hidden_state")
    print("  routed_gather(last_hidden_state, idx, mask) -> text/query/cls states")
    print(f"  boundary_head_L<{'|'.join(map(str, buckets))}>(text, text_mask, query, query_mask)")
    print("      -> cand_indices, pair_logits, cand_valid, null_logits, count_log_rates")
    print("  classifier(choice_states) -> logits")
    print("=" * 64)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLiNER2.5 boundary ONNX Exporter v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_path", required=True,
                   help="Local path or HuggingFace repo id of the boundary checkpoint")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--buckets", default=",".join(map(str, DEFAULT_BUCKETS)),
                   help=f"Comma-separated length buckets (default {DEFAULT_BUCKETS}, min {MIN_BUCKET})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_boundary(
        model_path=args.model_path,
        out_dir=Path(args.out_dir),
        buckets=tuple(sorted(int(x) for x in args.buckets.split(","))),
    )
