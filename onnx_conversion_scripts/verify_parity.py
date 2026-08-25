"""
PyTorch vs ONNX parity check for the GLiNER2.5 boundary export.

Compares every ONNX fragment against the matching PyTorch module on random
inputs, and additionally checks the assumption the bucketing rests on: padding
the text to a larger bucket, with `text_mask = 0` on the padded rows, must not
change the result.

Usage:
    python verify_parity.py \
        --model_path fastino/gliner2.5-multi-v1 \
        --onnx_dir models/gliner2.5-multi-v1-onnx

Exits with status 1 if any check exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# **Relative** tolerances, scaled by the reference's largest magnitude. An
# absolute threshold is unusable here because the fragments operate at very
# different scales.
TOL_FP32 = 1e-5
TOL_FP16 = 5e-3

# Fragments that emit probabilities live in [0,1], so relative and absolute
# error coincide. With random inputs the dot products over 768 dimensions have
# magnitude ~sqrt(768), the logits saturate, and an FP16 perturbation around the
# sigmoid transition is worth a few thousandths of a probability. That is the
# cost of half precision on a probability, not a defect of the export.
TOL_FP16_PROB = 2e-2

# Boundary candidate pool: FP32 must match exactly; in FP16 rounding can swap a
# near-tied candidate right at the `pool_size` cut. Up to 1% is a precision
# effect, not a defect of the export.
TOL_POOL_FP32 = 0.0
TOL_POOL_FP16 = 1e-2


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(sess, feeds: dict) -> list[np.ndarray]:
    typed = {}
    for i in sess.get_inputs():
        v = feeds[i.name]
        want = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }[i.type]
        typed[i.name] = np.asarray(v).astype(want)
    return sess.run(None, typed)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, float, float, bool]] = []

    def add(self, fragment: str, variant: str, delta: float, tol: float) -> None:
        self.rows.append((fragment, variant, delta, tol, delta <= tol))

    @staticmethod
    def relative(ref: np.ndarray, got: np.ndarray) -> float:
        """Largest error, scaled by the reference's magnitude."""
        ref = ref.astype(np.float64)
        got = got.astype(np.float64)
        scale = max(float(np.abs(ref).max()), 1e-12)
        return float(np.abs(ref - got).max() / scale)

    def failed(self) -> bool:
        return any(not ok for *_, ok in self.rows)

    def show(self) -> None:
        print()
        print(f"{'fragment':<26} {'variant':<18} {'rel. error':>12} {'tolerance':>12}  result")
        print("-" * 82)
        for frag, var, d, tol, ok in self.rows:
            print(f"{frag:<26} {var:<18} {d:>12.3e} {tol:>12.3e}  {'OK' if ok else 'FAILED'}")
        print()
        print("FAILED" if self.failed() else "all checks passed")


def _compare_candidate_pool(ref_idx, ref_log, ref_val, got_idx, got_log, got_val, num_queries):
    """
    Compares the candidate pool as a **set**, not positionally.

    Pool order carries no meaning: it comes from an `argsort` over frequently
    near-tied scores, and sort stability is exactly what the export removes
    (ONNX has no stable Sort). Under FP16 rounding permutes the ties. Comparing
    `cand_indices` element by element would fail a perfectly correct graph.

    What matters is that the set of proposed `(start, end)` pairs matches and
    that the logit attached to each pair is the same.

    Logits are compared **after the sigmoid**: that is the quantity the decoder
    thresholds on, and therefore the only one whose deviation is interpretable.
    A 1.6e-2 gap on a raw logit, which looks large, is worth less than half a
    percentage point of probability.

    Returns `(fraction of missing candidates, max |probability delta|)`.
    """
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        z = np.exp(x)
        return float(z / (1.0 + z))

    missing_total = 0
    expected_total = 0
    worst = 0.0
    for q in range(num_queries):
        ref_pairs = {
            tuple(int(v) for v in pair): float(logit)
            for pair, logit, ok in zip(ref_idx[0, q], ref_log[0, q], ref_val[0, q])
            if ok
        }
        got_pairs = {
            tuple(int(v) for v in pair): float(logit)
            for pair, logit, ok in zip(got_idx[0, q], got_log[0, q], got_val[0, q])
            if ok
        }
        common = set(ref_pairs) & set(got_pairs)
        expected_total += len(ref_pairs)
        missing_total += len(ref_pairs) - len(common)
        for key in common:
            worst = max(worst, abs(_sigmoid(ref_pairs[key]) - _sigmoid(got_pairs[key])))
    fraction_missing = missing_total / max(expected_total, 1)
    return fraction_missing, worst


def _variants(onnx_dir: Path, stem: str) -> list[tuple[str, Path, float]]:
    out = []
    for suffix, tol in (("_fp32", TOL_FP32), ("_fp16", TOL_FP16), ("_fp16_iobinding", TOL_FP16)):
        p = onnx_dir / f"{stem}{suffix}.onnx"
        if p.exists():
            out.append((suffix.lstrip("_"), p, tol))
    return out


def _compare(
    report: Report,
    stem: str,
    onnx_dir: Path,
    feeds: dict,
    ref: np.ndarray,
    out_index: int = 0,
    tol_fp16: float = TOL_FP16,
) -> None:
    for name, path, tol in _variants(onnx_dir, stem):
        if name != "fp32":
            tol = tol_fp16
        got = _run(_session(path), feeds)[out_index]
        report.add(stem, name, Report.relative(ref, got), tol)


# ─────────────────────────────────────────────────────────────────────────────
# boundary
# ─────────────────────────────────────────────────────────────────────────────
def verify_boundary(model_path: str, onnx_dir: Path) -> Report:
    import json

    from gliner2 import AutoExtractor

    manifest = json.loads((onnx_dir / "boundary_manifest.json").read_text())
    model = AutoExtractor.from_pretrained(model_path)
    model.eval()
    head = model.boundary_head
    head.eval()
    head.collect_diagnostics = False
    H = model.encoder.config.hidden_size

    torch.manual_seed(0)
    report = Report()
    SEQ, Q, K = 40, 4, 5

    ids = torch.randint(5, 1000, (1, SEQ))
    mask = torch.ones(1, SEQ, dtype=torch.long)
    with torch.no_grad():
        hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    _compare(report, "encoder", onnx_dir,
             {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}, hidden.numpy())

    idx = torch.randint(0, SEQ, (1, 20))
    rmask = torch.ones(1, 20, dtype=torch.long)
    ref = hidden.gather(1, idx.clamp(0, SEQ - 1).unsqueeze(-1).expand(-1, -1, H)) * rmask.unsqueeze(-1)
    _compare(report, "routed_gather", onnx_dir,
             {"last_hidden_state": hidden.numpy(), "indices": idx.numpy(), "mask": rmask.numpy()},
             ref.numpy())

    choice = torch.randn(K, H)
    with torch.no_grad():
        ref = model.classifier(choice).squeeze(-1)
    _compare(report, "classifier", onnx_dir, {"choice_states": choice.numpy()}, ref.numpy())

    # ── heads, one per bucket ─────────────────────────────────────────────
    for L in manifest["length_buckets"]:
        ts = torch.randn(1, L, H)
        tm = torch.ones(1, L, dtype=torch.bool)
        qs = torch.randn(1, Q, H)
        qm = torch.ones(1, Q, dtype=torch.bool)
        with torch.no_grad():
            out = head(ts, tm, qs, qm, targets=None,
                       return_candidates=True, collect_diagnostics=False)
        ref_idx = out.candidates.indices.numpy()
        ref_log = out.candidates.pair_logits.numpy()

        feeds = {
            "text_states": ts.numpy(), "text_mask": tm.numpy(),
            "query_states": qs.numpy(), "query_mask": qm.numpy(),
        }
        ref_val = out.candidates.valid_mask.numpy()
        for name, path, tol in _variants(onnx_dir, f"boundary_head_L{L}"):
            got = _run(_session(path), feeds)
            g_idx, g_log, g_val = got[0], got[1], got[2].astype(bool)
            miss, delta = _compare_candidate_pool(
                ref_idx, ref_log, ref_val, g_idx, g_log, g_val, Q
            )
            pool_tol = TOL_POOL_FP32 if name == "fp32" else TOL_POOL_FP16
            prob_tol = TOL_FP32 if name == "fp32" else TOL_FP16_PROB
            report.add(f"boundary_head_L{L}[pool]", name, miss, pool_tol)
            report.add(f"boundary_head_L{L}[prob]", name, delta, prob_tol)

    # ── the bucketing assumption: masked padding is transparent ───────────
    buckets = sorted(manifest["length_buckets"])
    for real in (buckets[0] - 8, buckets[0]):
        if real < manifest["min_bucket"]:
            continue
        target = next(b for b in buckets if b >= real)
        ts = torch.randn(1, real, H)
        qs = torch.randn(1, Q, H)
        qm = torch.ones(1, Q, dtype=torch.bool)
        with torch.no_grad():
            a = head(ts, torch.ones(1, real, dtype=torch.bool), qs, qm,
                     targets=None, return_candidates=True, collect_diagnostics=False)
            pad = target - real
            # noise in the padding: if masking works it must not matter
            ts_p = torch.cat([ts, torch.randn(1, pad, H)], 1)
            tm_p = torch.cat([torch.ones(1, real, dtype=torch.bool),
                              torch.zeros(1, pad, dtype=torch.bool)], 1)
            b = head(ts_p, tm_p, qs, qm, targets=None,
                     return_candidates=True, collect_diagnostics=False)
        miss, delta = _compare_candidate_pool(
            a.candidates.indices.numpy(), a.candidates.pair_logits.numpy(),
            a.candidates.valid_mask.numpy(),
            b.candidates.indices.numpy(), b.candidates.pair_logits.numpy(),
            b.candidates.valid_mask.numpy(), Q,
        )
        report.add(f"padding {real}->{target}[pool]", "pytorch", miss, TOL_POOL_FP32)
        report.add(f"padding {real}->{target}[prob]", "pytorch", delta, TOL_FP32)

    return report


def main() -> int:
    p = argparse.ArgumentParser(description="PyTorch vs ONNX parity (boundary)")
    p.add_argument("--model_path", required=True,
                   help="Local path or HuggingFace repo id of the boundary checkpoint")
    p.add_argument("--onnx_dir", required=True, help="Directory holding the ONNX export")
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)
    print(f"checkpoint : {args.model_path}")
    print(f"onnx       : {onnx_dir}")

    report = verify_boundary(args.model_path, onnx_dir)
    report.show()
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
