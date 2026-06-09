from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTER_RE = re.compile(r"[ABCD]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--n_continuation_tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_rows = json.loads(Path(args.inputs).read_text())
    if args.limit:
        in_rows = in_rows[: args.limit]

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    cls = type(model).__name__
    print(f"[load] model class: {cls}")
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    print(f"[load] num_hidden_layers={n_layers}")
    if args.layer < 0 or args.layer > n_layers:
        raise SystemExit(f"--layer {args.layer} out of range 0..{n_layers}")

    rows: list[dict] = []
    t_loop_start = time.time()
    for i, r in enumerate(in_rows):
        prompt = r["prompt"]
        enc = tok(prompt, return_tensors="pt").to("cuda")
        seq_len = int(enc["input_ids"].shape[1])
        last_token_id = int(enc["input_ids"][0, -1].item())
        last_token_str = tok.decode([last_token_id])


        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[args.layer]
        last_act = hs[0, -1].float().cpu()
        d_model = int(last_act.shape[0])
        norm = float(last_act.norm())


        with torch.no_grad():
            gen_out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=args.n_continuation_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = gen_out[0, seq_len:]
        continuation = tok.decode(gen_ids, skip_special_tokens=True)
        m = LETTER_RE.search(continuation)
        model_answer = m.group(0) if m else None

        matches_correct = (model_answer == r["correct_answer"]) if model_answer else False
        matches_bias = (model_answer == r["biased_letter"]) if model_answer else False

        rows.append({
            "idx": int(r["id"]),
            "mcq_id": int(r["mcq_id"]),
            "condition": r["condition"],
            "category": r["category"],
            "correct_answer": r["correct_answer"],
            "biased_letter": r["biased_letter"],
            "text": prompt,
            "activation_vector": last_act.tolist(),
            "activation_norm": norm,
            "seq_len": seq_len,
            "last_token_id": last_token_id,
            "last_token_str": last_token_str,
            "model_answer": model_answer,
            "model_continuation": continuation,
            "model_answer_matches_correct": bool(matches_correct),
            "model_answer_matches_bias": bool(matches_bias),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1:3d}/{len(in_rows)}] "
                f"mcq={r['mcq_id']:02d} {r['condition']:7s} "
                f"correct={r['correct_answer']} biased={r['biased_letter']} "
                f"answer={model_answer} "
                f"matches_correct={matches_correct} "
                f"matches_bias={matches_bias} "
                f"norm={norm:.0f}"
            )

    elapsed = time.time() - t_loop_start
    print(f"[done] {len(rows)} rows in {elapsed:.1f}s "
          f"({elapsed/max(len(rows),1):.2f}s/row)")


    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "idx": [r["idx"] for r in rows],
        "mcq_id": [r["mcq_id"] for r in rows],
        "condition": [r["condition"] for r in rows],
        "category": [r["category"] for r in rows],
        "correct_answer": [r["correct_answer"] for r in rows],
        "biased_letter": [r["biased_letter"] for r in rows],
        "text": [r["text"] for r in rows],
        "activation_vector": [r["activation_vector"] for r in rows],
        "activation_norm": [r["activation_norm"] for r in rows],
        "seq_len": [r["seq_len"] for r in rows],
        "last_token_id": [r["last_token_id"] for r in rows],
        "last_token_str": [r["last_token_str"] for r in rows],
        "model_answer": [r["model_answer"] for r in rows],
        "model_continuation": [r["model_continuation"] for r in rows],
        "model_answer_matches_correct": [r["model_answer_matches_correct"] for r in rows],
        "model_answer_matches_bias": [r["model_answer_matches_bias"] for r in rows],
    })
    pq.write_table(table, out)
    print(f"[wrote] {out}  rows={len(rows)} d_model={d_model}")


if __name__ == "__main__":
    main()
