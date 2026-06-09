from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


from nla_inference import NLAClient, NLACritic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--av_checkpoint", required=True)
    ap.add_argument("--ar_checkpoint", required=True)
    ap.add_argument("--sglang_url", default="http://127.0.0.1:30000")
    ap.add_argument("--ar_device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (reproducible). Match phase-1 if you want apples-to-apples.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    table = pq.read_table(args.activations)
    rows_in = table.to_pylist()
    if args.limit:
        rows_in = rows_in[: args.limit]
    print(f"[load] {len(rows_in)} rows from {args.activations}")

    print(f"[load] AV @ {args.av_checkpoint} via SGLang {args.sglang_url}")
    client = NLAClient(args.av_checkpoint, sglang_url=args.sglang_url)

    print(f"[load] AR critic @ {args.ar_checkpoint} on {args.ar_device}")
    t0 = time.time()
    critic = NLACritic(args.ar_checkpoint, device=args.ar_device,
                       dtype=torch.bfloat16)
    print(f"[load] AR loaded in {time.time()-t0:.1f}s")

    results: list[dict] = []
    t_loop = time.time()
    for i, r in enumerate(rows_in):
        v = np.asarray(r["activation_vector"], dtype=np.float32)
        gold = torch.as_tensor(v)


        t0 = time.time()
        try:
            explanation = client.generate(
                v, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=1.0,
            )
        except Exception as e:
            print(f"[AV err] idx={r['idx']}: {e!r}")
            explanation = f"[AV ERROR: {type(e).__name__}: {e}]"
        t_av = time.time() - t0


        t0 = time.time()
        try:
            mse, cos = critic.score(explanation, v)
            pred = critic.reconstruct(explanation)
            pred_norm = float(pred.norm())
        except Exception as e:
            print(f"[AR err] idx={r['idx']}: {e!r}")
            mse = float("nan"); cos = float("nan"); pred_norm = float("nan")
        t_ar = time.time() - t0

        out = {**r}


        out_for_parquet = {**out}
        out_for_parquet.update({
            "explanation": explanation,
            "reconstructed_norm": pred_norm,
            "reconstruction_mse": float(mse),
            "reconstruction_cos": float(cos),
            "av_seconds": round(t_av, 3),
            "ar_seconds": round(t_ar, 3),
        })
        results.append(out_for_parquet)

        if (i + 1) % 5 == 0 or i == 0 or i == len(rows_in) - 1:
            print(
                f"[{i+1:3d}/{len(rows_in)}] "
                f"mcq={r['mcq_id']:02d} {r['condition']:7s} "
                f"correct={r['correct_answer']} biased={r['biased_letter']} "
                f"answer={r['model_answer']} "
                f"mse={mse:.4f} cos={cos:.4f} "
                f"av={t_av:.1f}s ar={t_ar:.2f}s"
            )

    elapsed = time.time() - t_loop
    print(f"[done] {len(results)} rows in {elapsed:.1f}s "
          f"({elapsed/max(len(results),1):.2f}s/row)")


    out_pq = Path(args.out_parquet)
    out_pq.parent.mkdir(parents=True, exist_ok=True)


    columns: dict[str, list] = {col: [] for col in
        list(rows_in[0].keys()) + [
            "explanation", "reconstructed_norm", "reconstruction_mse",
            "reconstruction_cos", "av_seconds", "ar_seconds",
        ]
    }
    for r in results:
        for k in columns:
            columns[k].append(r.get(k))
    pq.write_table(pa.table(columns), out_pq)
    print(f"[wrote] {out_pq}")


    json_rows = [{k: v for k, v in r.items() if k != "activation_vector"}
                 for r in results]
    Path(args.out_json).write_text(
        json.dumps(json_rows, indent=2, ensure_ascii=False))
    print(f"[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
