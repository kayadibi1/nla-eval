import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--text", default="The Eiffel Tower is located in Paris, France.")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--out", default="activation_gemma.parquet")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    print(f"model class:   {type(model).__name__}")
    print(f"text_config layers: {model.config.text_config.num_hidden_layers}")

    enc = tok(args.text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)


    hs = out.hidden_states[args.layer]
    last_token = hs[0, -1].float().cpu()
    d_model = last_token.shape[0]
    norm = float(last_token.norm())
    print(f"text:          {args.text!r}")
    print(f"layer:         {args.layer}")
    print(f"seq_len:       {enc['input_ids'].shape[1]}")
    print(f"d_model:       {d_model}  (expected 3840)")
    print(f"L2 norm:       {norm:.1f}  (Gemma layer-32 typical ~74000 per docs)")
    print(f"first 5:       {last_token[:5].tolist()}")

    table = pa.table({"activation_vector": [last_token.tolist()]})
    out_path = Path(args.out).resolve()
    pq.write_table(table, out_path)
    print(f"wrote:         {out_path}  (1 row, list<float>[{d_model}])")


if __name__ == "__main__":
    main()
