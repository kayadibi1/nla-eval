from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/workspace/nla-research")
QWEN_RESULTS = ROOT / "phase1_results_v2.parquet"
QWEN_ACTS    = ROOT / "phase1_activations_v2.parquet"
GEMMA_RESULTS = ROOT / "phase1_results_gemma_v2.parquet"
GEMMA_ACTS   = ROOT / "phase1_activations_gemma_v2.parquet"

SUBGROUPS = ["A", "B", "C", "D", "E1", "E2"]


def _welch_t(a, b) -> tuple[float, float, float]:
    from scipy import stats
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(getattr(res, "df", float("nan"))), float(res.pvalue)


def _spearman(a, b) -> tuple[float, float]:
    from scipy import stats
    res = stats.spearmanr(a, b)
    return float(res.statistic if hasattr(res, "statistic") else res.correlation), float(res.pvalue)


def cohens_d(a, b) -> float:
    sa = st.stdev(a) if len(a) > 1 else 0.0
    sb = st.stdev(b) if len(b) > 1 else 0.0
    pooled = math.sqrt((sa ** 2 + sb ** 2) / 2) if (sa + sb) > 0 else float("nan")
    return (st.mean(a) - st.mean(b)) / pooled if pooled and not math.isnan(pooled) else float("nan")


def load_model(results_path: Path, acts_path: Path, label: str) -> dict:
    print(f"[{label}] loading {results_path.name} + {acts_path.name}", file=sys.stderr)
    res = pq.read_table(str(results_path)).to_pylist()
    acts_t = pq.read_table(str(acts_path), columns=["idx", "activation_vector"])
    acts = {int(r["idx"]): r["activation_vector"] for r in acts_t.to_pylist()}


    idxs_sorted = sorted(acts.keys())
    G = np.asarray([acts[i] for i in idxs_sorted], dtype=np.float32)
    G_n = G / np.linalg.norm(G, axis=1, keepdims=True).clip(min=1e-12)
    cos_mat = G_n @ G_n.T
    iu = np.triu_indices(len(G_n), k=1)
    pair_cos = cos_mat[iu]
    mean_pair_cos = float(pair_cos.mean())


    for r in res:
        c = r.get("reconstruction_cos")
        r["abc"] = (float(c) - mean_pair_cos) if (c is not None and not math.isnan(c)) else float("nan")
        sg = r.get("tier_subgroup") or r["tier"]
        r["_sg"] = sg

    return {
        "label": label,
        "rows": res,
        "mean_pair_cos": mean_pair_cos,
        "n_pairs": len(pair_cos),
    }


def per_subgroup(rows: list[dict], col: str) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by.setdefault(r["_sg"], []).append(float(v))
    return by


def desc(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    sx = sorted(xs)
    n = len(sx)

    def q(p):
        i = p * (n - 1)
        lo, hi = math.floor(i), math.ceil(i)
        return sx[lo] if lo == hi else sx[lo] + (i - lo) * (sx[hi] - sx[lo])

    return {
        "n": n, "mean": st.mean(xs),
        "median": st.median(xs),
        "stdev": st.stdev(xs) if n > 1 else 0.0,
        "min": min(xs), "max": max(xs),
        "p25": q(0.25), "p75": q(0.75),
    }


def emit_per_subgroup_table(model: dict, lines: list[str]) -> None:
    label = model["label"]
    rows = model["rows"]
    mse = per_subgroup(rows, "reconstruction_mse")
    cos = per_subgroup(rows, "reconstruction_cos")
    abc = per_subgroup(rows, "abc")
    lines.append(f"### {label} - per-subgroup metrics")
    lines.append("")
    lines.append(f"`mean_pairwise_cos(gold) = {model['mean_pair_cos']:.4f}` "
                 f"(over all {model['n_pairs']:,} pairs in the 500-input set)")
    lines.append("")
    lines.append("| sub | n | ABC mean | ABC median | ABC stdev | cos mean | raw MSE mean |")
    lines.append("|---|---|---|---|---|---|---|")
    for sg in SUBGROUPS:
        if sg not in abc:
            continue
        a = desc(abc[sg]); c = desc(cos[sg]); m = desc(mse[sg])
        lines.append(
            f"| {sg} | {a['n']} | {a['mean']:+.3f} | {a['median']:+.3f} | "
            f"{a['stdev']:.3f} | {c['mean']:.3f} | {m['mean']:.3f} |"
        )
    lines.append("")


def emit_pairwise_welch_abc(model: dict, lines: list[str]) -> None:
    abc = per_subgroup(model["rows"], "abc")
    keys = [k for k in SUBGROUPS if k in abc]
    pairs = list(combinations(keys, 2))
    n_tests = len(pairs)
    bonf = 0.05 / max(n_tests, 1)
    lines.append(f"### {model['label']} - pairwise Welch's t-test on ABC")
    lines.append(f"Bonferroni α = 0.05/{n_tests} = {bonf:.4f}.  Cohen's d shown for the requested A↔E1, A↔E2, E1↔E2 contrasts.")
    lines.append("")
    lines.append("| group1 | group2 | mean1 | mean2 | Δ (g2−g1) | t | df | p | sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a, b in pairs:
        xa, xb = abc[a], abc[b]
        ma, mb = st.mean(xa), st.mean(xb)
        t, df, p = _welch_t(xa, xb)
        sig = "**yes**" if p < bonf else "no"
        lines.append(
            f"| {a} | {b} | {ma:+.3f} | {mb:+.3f} | {mb-ma:+.3f} | "
            f"{t:+.2f} | {df:.1f} | {p:.2e} | {sig} |"
        )
    lines.append("")
    lines.append(f"**Cohen's d (ABC) for {model['label']} requested contrasts:**")
    for a, b in [("A", "E1"), ("A", "E2"), ("E1", "E2")]:
        if a in abc and b in abc:
            d = cohens_d(abc[a], abc[b])
            lines.append(f"- {a} vs {b}:  d = {d:+.2f}  "
                         f"(mean_{a}={st.mean(abc[a]):+.3f}, mean_{b}={st.mean(abc[b]):+.3f})")
    lines.append("")


def emit_side_by_side(qwen: dict, gemma: dict, lines: list[str]) -> None:
    lines.append("## Cross-model side-by-side")
    lines.append("")
    lines.append(f"- Qwen mean_pairwise_cos(gold)  = {qwen['mean_pair_cos']:.4f}")
    lines.append(f"- Gemma mean_pairwise_cos(gold) = {gemma['mean_pair_cos']:.4f}")
    lines.append("")
    lines.append("Reading: ABC is the headline number - same vertical axis for both models. Raw MSE shown for traceability but is NOT directly comparable because of the anisotropy gap.")
    lines.append("")

    qabc = per_subgroup(qwen["rows"], "abc")
    gabc = per_subgroup(gemma["rows"], "abc")
    qmse = per_subgroup(qwen["rows"], "reconstruction_mse")
    gmse = per_subgroup(gemma["rows"], "reconstruction_mse")
    qcos = per_subgroup(qwen["rows"], "reconstruction_cos")
    gcos = per_subgroup(gemma["rows"], "reconstruction_cos")

    lines.append("| sub | n | Qwen ABC | Gemma ABC | Δ ABC | Qwen cos | Gemma cos | Qwen MSE | Gemma MSE |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sg in SUBGROUPS:
        if sg not in qabc or sg not in gabc:
            continue
        n = len(qabc[sg])
        qa, ga = st.mean(qabc[sg]), st.mean(gabc[sg])
        lines.append(
            f"| {sg} | {n} | {qa:+.3f} | {ga:+.3f} | {ga-qa:+.3f} | "
            f"{st.mean(qcos[sg]):.3f} | {st.mean(gcos[sg]):.3f} | "
            f"{st.mean(qmse[sg]):.3f} | {st.mean(gmse[sg]):.3f} |"
        )
    lines.append("")


def emit_rank_correlation(qwen: dict, gemma: dict, lines: list[str]) -> None:
    by_idx_q = {r["idx"]: r for r in qwen["rows"]}
    by_idx_g = {r["idx"]: r for r in gemma["rows"]}
    common_idx = sorted(set(by_idx_q) & set(by_idx_g))

    qcos = [by_idx_q[i]["reconstruction_cos"] for i in common_idx]
    gcos = [by_idx_g[i]["reconstruction_cos"] for i in common_idx]
    rho, p = _spearman(qcos, gcos)

    lines.append("## Rank correlation (Spearman ρ on cos(pred, gold))")
    lines.append("")
    lines.append(f"Pairs by idx; ρ measures whether the two models find the same inputs harder/easier to reconstruct, independent of metric scale.")
    lines.append("")
    lines.append(f"- **Overall** (n={len(common_idx)}): ρ = {rho:+.3f}, p = {p:.2e}")
    lines.append("")
    lines.append("| sub | n | Qwen cos mean | Gemma cos mean | Spearman ρ | p |")
    lines.append("|---|---|---|---|---|---|")
    for sg in SUBGROUPS:
        idxs = [i for i in common_idx
                if (by_idx_q[i].get("tier_subgroup") or by_idx_q[i]["tier"]) == sg]
        if len(idxs) < 4:
            continue
        xq = [by_idx_q[i]["reconstruction_cos"] for i in idxs]
        xg = [by_idx_g[i]["reconstruction_cos"] for i in idxs]
        r2, p2 = _spearman(xq, xg)
        lines.append(f"| {sg} | {len(idxs)} | {st.mean(xq):.3f} | {st.mean(xg):.3f} | {r2:+.3f} | {p2:.2e} |")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="Also write the report to this file.")
    args = ap.parse_args()

    qwen = load_model(QWEN_RESULTS, QWEN_ACTS, "Qwen2.5-7B  (layer 20, d=3584)")
    gemma = load_model(GEMMA_RESULTS, GEMMA_ACTS, "Gemma3-12B  (layer 32, d=3840)")

    lines: list[str] = []
    lines.append("# Phase 1 v2 analysis - anisotropy-corrected metrics")
    lines.append("")
    lines.append(f"- Qwen N=500   mean_pairwise_cos(gold) = **{qwen['mean_pair_cos']:.4f}**  ({qwen['n_pairs']:,} pairs)")
    lines.append(f"- Gemma N=500  mean_pairwise_cos(gold) = **{gemma['mean_pair_cos']:.4f}**  ({gemma['n_pairs']:,} pairs)")
    lines.append("")
    lines.append("Methodological note: a raw MSE = 2(1−cos) comparison would credit Gemma for ~0.07 of free MSE just by residual-stream geometry. ABC subtracts that baseline so both models are scored on \"how much beyond random gold-pair similarity\" the AR captures.")
    lines.append("")

    lines.append("## Per-subgroup metrics (lead: ABC)")
    lines.append("")
    emit_per_subgroup_table(qwen, lines)
    emit_per_subgroup_table(gemma, lines)

    lines.append("## Pairwise tests on ABC")
    lines.append("")
    emit_pairwise_welch_abc(qwen, lines)
    emit_pairwise_welch_abc(gemma, lines)

    emit_side_by_side(qwen, gemma, lines)
    emit_rank_correlation(qwen, gemma, lines)

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\n[wrote {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
