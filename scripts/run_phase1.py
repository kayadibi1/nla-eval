from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path("/workspace/nla-research")


class P:
    INPUTS = ROOT / "phase1_inputs.json"
    ACTIVATIONS = ROOT / "phase1_activations.parquet"
    RESULTS_JSONL = ROOT / "phase1_results.jsonl"
    RESULTS_PARQUET = ROOT / "phase1_results.parquet"
    RESULTS_JSON = ROOT / "phase1_results.json"


def set_suffix(suffix: str) -> None:
    P.INPUTS = ROOT / f"phase1_inputs{suffix}.json"
    P.ACTIVATIONS = ROOT / f"phase1_activations{suffix}.parquet"
    P.RESULTS_JSONL = ROOT / f"phase1_results{suffix}.jsonl"
    P.RESULTS_PARQUET = ROOT / f"phase1_results{suffix}.parquet"
    P.RESULTS_JSON = ROOT / f"phase1_results{suffix}.json"


CKPT_ROOT = Path("/workspace/nla-research/natural_language_autoencoders/checkpoints")


MODELS: dict[str, dict] = {
    "qwen": {
        "base_id": "Qwen/Qwen2.5-7B-Instruct",
        "layer": 20,
        "av": str(CKPT_ROOT / "av"),
        "ar": str(CKPT_ROOT / "ar"),
        "is_multimodal": False,
    },
    "gemma": {
        "base_id": "google/gemma-3-12b-it",
        "layer": 32,
        "av": str(CKPT_ROOT / "gemma_av"),
        "ar": str(CKPT_ROOT / "gemma_ar"),


        "is_multimodal": True,
    },
}


MODEL_ID: str = MODELS["qwen"]["base_id"]
LAYER: int = MODELS["qwen"]["layer"]
AV_CHECKPOINT: str = MODELS["qwen"]["av"]
AR_CHECKPOINT: str = MODELS["qwen"]["ar"]


def set_model(name: str) -> None:
    global MODEL_ID, LAYER, AV_CHECKPOINT, AR_CHECKPOINT
    cfg = MODELS[name]
    MODEL_ID = cfg["base_id"]
    LAYER = cfg["layer"]
    AV_CHECKPOINT = cfg["av"]
    AR_CHECKPOINT = cfg["ar"]


def tier_subgroup(tier: str, idx: int) -> str:
    if tier in "ABCD":
        return tier
    if tier == "E":
        if 120 <= idx <= 149:
            return "E1"
        if 430 <= idx <= 499:
            return "E2"
    return "E_other"


def load_inputs() -> list[dict]:
    rows = json.loads(P.INPUTS.read_text())
    assert len(rows) % 5 == 0, f"need a multiple of 5; got {len(rows)}"
    return rows


def round_robin(rows: list[dict]) -> list[dict]:
    by_tier: dict[str, list[dict]] = {t: [] for t in "ABCDE"}
    for r in rows:
        by_tier[r["tier"]].append(r)
    n_per = len(rows) // 5
    assert all(len(by_tier[t]) == n_per for t in "ABCDE"), (
        f"unbalanced tiers: { {t: len(by_tier[t]) for t in 'ABCDE'} }"
    )
    out: list[dict] = []
    for i in range(n_per):
        for t in "ABCDE":
            out.append(by_tier[t][i])
    assert len(out) == len(rows)
    return out


def stage_extract() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_inputs()
    order = round_robin(rows)

    print(f"[extract] loading {MODEL_ID} on cuda (bf16)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"[extract] loaded in {time.time()-t0:.1f}s", flush=True)

    out_idx, out_tier, out_text, out_vec, out_norm, out_seqlen, \
        out_ltid, out_ltstr = [], [], [], [], [], [], [], []

    t1 = time.time()
    for i, r in enumerate(order):
        enc = tok(r["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hs = outputs.hidden_states[LAYER]
        last = hs[0, -1].float().cpu().numpy()
        seq_len = int(enc["input_ids"].shape[1])
        last_id = int(enc["input_ids"][0, -1].item())
        last_str = tok.decode([last_id])
        norm = float(np.linalg.norm(last))

        out_idx.append(r["idx"])
        out_tier.append(r["tier"])
        out_text.append(r["text"])
        out_vec.append(last.tolist())
        out_norm.append(norm)
        out_seqlen.append(seq_len)
        out_ltid.append(last_id)
        out_ltstr.append(last_str)

        if (i + 1) % 30 == 0 or (i + 1) == len(order):
            elapsed = time.time() - t1
            print(f"[extract] {i+1}/{len(order)}  elapsed={elapsed:.1f}s  "
                  f"({elapsed/(i+1)*1000:.0f}ms/input)", flush=True)

    table = pa.table({
        "idx": out_idx,
        "tier": out_tier,
        "tier_subgroup": [tier_subgroup(t, i) for t, i in zip(out_tier, out_idx)],
        "text": out_text,
        "activation_vector": out_vec,
        "activation_norm": out_norm,
        "seq_len": out_seqlen,
        "last_token_id": out_ltid,
        "last_token_str": out_ltstr,
    })


    table = table.sort_by("idx")
    pq.write_table(table, P.ACTIVATIONS)
    print(f"[extract] wrote {len(out_idx)} rows to {P.ACTIVATIONS}", flush=True)
    print(f"[extract] norm summary: min={min(out_norm):.1f} "
          f"max={max(out_norm):.1f} mean={sum(out_norm)/len(out_norm):.1f}",
          flush=True)


def _load_done_idxs() -> set[int]:
    if not P.RESULTS_JSONL.exists():
        return set()
    done = set()
    for line in P.RESULTS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(int(json.loads(line)["idx"]))
        except Exception:
            pass
    return done


def _load_activations() -> dict[int, dict]:
    pf = pq.ParquetFile(P.ACTIVATIONS)
    table = pf.read()
    out: dict[int, dict] = {}
    for r in table.to_pylist():
        out[int(r["idx"])] = r
    return out


def _flush_results() -> None:
    if not P.RESULTS_JSONL.exists():
        return
    rows = [json.loads(l) for l in P.RESULTS_JSONL.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["idx"])
    table = pa.table({
        "idx": [r["idx"] for r in rows],
        "tier": [r["tier"] for r in rows],
        "tier_subgroup": [tier_subgroup(r["tier"], r["idx"]) for r in rows],
        "text": [r["text"] for r in rows],
        "activation_norm": [r["activation_norm"] for r in rows],
        "seq_len": [r["seq_len"] for r in rows],
        "last_token_id": [r["last_token_id"] for r in rows],
        "last_token_str": [r["last_token_str"] for r in rows],
        "explanation": [r["explanation"] for r in rows],
        "reconstructed_norm": [r["reconstructed_norm"] for r in rows],
        "reconstruction_mse": [r["reconstruction_mse"] for r in rows],
        "reconstruction_cos": [r["reconstruction_cos"] for r in rows],
        "av_seconds": [r["av_seconds"] for r in rows],
        "ar_seconds": [r["ar_seconds"] for r in rows],
    })
    pq.write_table(table, P.RESULTS_PARQUET)
    P.RESULTS_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def stage_process(limit: int | None, resume: bool, sglang_url: str,
                  ar_device: str) -> None:
    if not P.ACTIVATIONS.exists():
        sys.exit(f"missing {P.ACTIVATIONS} - run `extract` first")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla_inference import NLAClient, NLACritic


    done = _load_done_idxs() if resume else set()
    if not resume and P.RESULTS_JSONL.exists():

        archived = P.RESULTS_JSONL.with_suffix(".jsonl.bak")
        P.RESULTS_JSONL.rename(archived)
        print(f"[process] archived old {P.RESULTS_JSONL.name} → {archived.name}",
              flush=True)

    activations = _load_activations()
    rows_in_order = round_robin(load_inputs())
    todo = [r for r in rows_in_order if r["idx"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"[process] {len(done)} already done, {len(todo)} to do "
          f"(of 150 total)", flush=True)
    if not todo:
        print("[process] nothing to do.", flush=True)
        _flush_results()
        return

    print(f"[process] connecting to AV server at {sglang_url} ...", flush=True)
    client = NLAClient(AV_CHECKPOINT, sglang_url=sglang_url)

    print(f"[process] loading AR critic on {ar_device} (bf16) ...", flush=True)
    t0 = time.time()
    critic = NLACritic(AR_CHECKPOINT, device=ar_device, dtype=torch.bfloat16)
    print(f"[process] AR loaded in {time.time()-t0:.1f}s", flush=True)

    t_batch_start = time.time()
    new_rows = 0
    slow_warned = False

    with P.RESULTS_JSONL.open("a") as fout:
        for i, meta in enumerate(todo):
            idx = meta["idx"]
            act_row = activations[idx]
            v = np.asarray(act_row["activation_vector"], dtype=np.float32)


            t_av = time.time()
            try:
                explanation = client.generate(
                    v, temperature=0.7, max_new_tokens=200,
                )
            except Exception as e:
                print(f"[process] AV ERROR on idx={idx}: {e!r}", flush=True)
                explanation = f"[ERROR: {e!r}]"
            av_s = time.time() - t_av


            t_ar = time.time()
            try:
                pred = critic.reconstruct(explanation)
                pred_norm = float(pred.norm())
                mse, cos = critic.score(explanation, v)
            except Exception as e:
                print(f"[process] AR ERROR on idx={idx}: {e!r}", flush=True)
                pred_norm = float("nan")
                mse, cos = float("nan"), float("nan")
            ar_s = time.time() - t_ar

            row = {
                "idx": idx,
                "tier": act_row["tier"],
                "tier_subgroup": tier_subgroup(act_row["tier"], idx),
                "text": act_row["text"],
                "activation_norm": float(act_row["activation_norm"]),
                "seq_len": int(act_row["seq_len"]),
                "last_token_id": int(act_row["last_token_id"]),
                "last_token_str": act_row["last_token_str"],
                "explanation": explanation,
                "reconstructed_norm": pred_norm,
                "reconstruction_mse": float(mse),
                "reconstruction_cos": float(cos),
                "av_seconds": av_s,
                "ar_seconds": ar_s,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            new_rows += 1

            elapsed_input = av_s + ar_s
            if elapsed_input > 60 and not slow_warned:
                print(f"[process] WARNING: idx={idx} took {elapsed_input:.1f}s "
                      f"(>60s threshold). Tell the user.", flush=True)
                slow_warned = True

            short_text = meta["text"].replace("\n", " / ")[:60]
            short_expl = explanation.replace("\n", " / ")[:80]
            print(f"[process] [{idx:3d} {act_row['tier']}{act_row['idx']%30:02d}]"
                  f" av={av_s:5.2f}s ar={ar_s:4.2f}s "
                  f"mse={mse:.3f} cos={cos:+.3f}  "
                  f"text={short_text!r:<62}  expl={short_expl!r}",
                  flush=True)

            if new_rows % 30 == 0:
                _flush_results()
                print(f"[process] checkpoint: flushed parquet+json at "
                      f"{new_rows} new rows", flush=True)

    elapsed = time.time() - t_batch_start
    _flush_results()
    print(f"[process] done. wrote {new_rows} rows in {elapsed:.1f}s "
          f"({elapsed/max(new_rows,1):.2f}s/row avg)", flush=True)
    print(f"[process] outputs: {P.RESULTS_PARQUET}, {P.RESULTS_JSON}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="",
                    help="Appended to all phase1_* basenames (e.g. '_v2').")
    ap.add_argument("--model", choices=list(MODELS.keys()), default="qwen",
                    help="Model family: qwen (default) or gemma.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    p = sub.add_parser("process")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many NEW rows.")
    p.add_argument("--resume", action="store_true",
                   help="Skip idxs already in phase1_results*.jsonl.")
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--ar-device", default="cuda")
    args = ap.parse_args()

    set_suffix(args.suffix)
    set_model(args.model)
    print(f"[run_phase1] model={args.model!r}  base={MODEL_ID!r}  layer={LAYER}  "
          f"suffix={args.suffix!r}  inputs={P.INPUTS.name}  "
          f"activations={P.ACTIVATIONS.name}  results={P.RESULTS_PARQUET.name}",
          flush=True)

    if args.cmd == "extract":
        stage_extract()
    elif args.cmd == "process":
        stage_process(args.limit, args.resume, args.sglang_url, args.ar_device)


if __name__ == "__main__":
    main()
