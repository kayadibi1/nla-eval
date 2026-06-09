from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
INJECT_PLACEHOLDER = "<INJECT>"


_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str


    injection_scale: float


def load_nla_config(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: float | None = None,
) -> NLAConfig:
    meta_path = Path(checkpoint_dir) / "nla_meta.yaml"
    assert meta_path.exists(), (
        f"no nla_meta.yaml at {checkpoint_dir!r}. Not an NLA checkpoint - "
        f"the sidecar ships alongside config.json/safetensors. If you "
        f"received a checkpoint without it, ask the provider for the sidecar."
    )
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"
    d_model = meta["d_model"] if kind == "nla_model" else meta["extraction"]["d_model"]


    inj_scale = meta.get("extraction", {}).get("injection_scale")
    if inj_scale is None:
        inj_scale = injection_scale_override
    assert inj_scale is not None, (
        f"nla_meta.yaml at {checkpoint_dir!r} has no extraction.injection_scale "
        f"(kind={kind!r}, role={meta.get('role')!r}). Actor checkpoints always "
        f"have it. If this is a critic sidecar or a dataset sidecar, pass "
        f"injection_scale_override explicitly."
    )

    t = meta["tokens"]
    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        actor_prompt_template=meta["prompt_templates"].get("av")
                              or meta["prompt_templates"]["actor"],
        injection_scale=float(inj_scale),
    )


    live_inj = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} → {live_inj}, sidecar says "
        f"[{cfg.injection_token_id}]. Multi-token = char split = wrong "
        f"tokenizer or vocab changed."
    )
    assert live_inj[0] != tokenizer.unk_token_id, (
        f"{cfg.injection_char!r} maps to UNK"
    )


    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    matches = [i for i, tok in enumerate(ids) if tok == cfg.injection_token_id]
    assert len(matches) == 1, (
        f"injection token appears {len(matches)}× in canonical prompt "
        f"(expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1
    assert ids[p - 1] == cfg.injection_left_neighbor_id, (
        f"left neighbor drift: {ids[p-1]} vs sidecar "
        f"{cfg.injection_left_neighbor_id}"
    )
    assert ids[p + 1] == cfg.injection_right_neighbor_id, (
        f"right neighbor drift: {ids[p+1]} vs sidecar "
        f"{cfg.injection_right_neighbor_id}"
    )

    return cfg


def load_embedding_only(
    checkpoint_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Embedding:
    root = Path(checkpoint_dir)

    def _find_key(keys: list[str], where: str) -> str:
        m = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
        assert len(m) == 1, (
            f"expected exactly one input-embedding key in {where} "
            f"(suffixes {_EMBED_KEY_SUFFIXES!r}), got {m!r}"
        )
        return m[0]

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), f"no model.safetensors or .index.json at {root!r}"
        with safe_open(str(shard), framework="pt") as f:
            key = _find_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d = weight.shape
    embed = torch.nn.Embedding(vocab, d, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed


_SCALED_EMBED_MODEL_TYPES = frozenset({
    "gemma", "gemma2", "gemma3", "gemma3_text", "t5",
})


def resolve_embed_scale(checkpoint_dir: str | Path) -> float:
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    model_type = getattr(text_cfg, "model_type", "") or ""
    if model_type in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_cfg.hidden_size)
    return 1.0


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int, left_id: int, right_id: int,
) -> torch.Tensor:
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == embeddings.shape[:-1]
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1]
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    vec_idx = 0
    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1
    assert vec_idx == vectors.shape[0], (
        f"found {vec_idx} injection sites with correct neighbors, expected "
        f"{vectors.shape[0]}. Template drift, tokenizer mismatch, or prompt "
        f"missing the injection marker."
    )
    return out


class NLAClient:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        sglang_url: str = "http://localhost:30000",
        injection_scale_override: float | None = None,
        device: str = "cpu",
    ):
        checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )


        self.cfg = load_nla_config(
            checkpoint_dir, self.tokenizer,
            injection_scale_override=injection_scale_override,
        )


        self.embed = load_embedding_only(checkpoint_dir, dtype=torch.bfloat16).to(device)
        self.embed_scale = resolve_embed_scale(checkpoint_dir)

        assert self.embed.weight.shape[1] == self.cfg.d_model, (
            f"embedding d={self.embed.weight.shape[1]} != sidecar "
            f"d_model={self.cfg.d_model}. Wrong checkpoint for this sidecar."
        )

        self.sglang_url = sglang_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(120.0))

        print(
            f"[NLAClient] {checkpoint_dir.name}: d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id})"
        )


    def _build_embeds(
        self, v_raw: torch.Tensor, prompt_content: str | None
    ) -> tuple[np.ndarray, int]:
        if prompt_content is None:
            content = self.cfg.actor_prompt_template.format(
                injection_char=self.cfg.injection_char
            )
        else:
            assert INJECT_PLACEHOLDER in prompt_content, (
                f"custom prompt must contain {INJECT_PLACEHOLDER!r}"
            )
            content = prompt_content.replace(
                INJECT_PLACEHOLDER, self.cfg.injection_char
            )


        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )
        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():


            embeds = (self.embed(ids_t.to(self.embed.weight.device))
                      * self.embed_scale).float()

        assert torch.isfinite(v_raw).all(), "activation has NaN/Inf"
        v_scaled = normalize_activation(
            v_raw.float().view(1, -1), self.cfg.injection_scale
        )

        injected = inject_at_marked_positions(
            ids_t, embeds.cpu(), v_scaled,
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )

        return injected[0].contiguous().numpy(), len(input_ids)

    def _sglang_generate(
        self, embeds_np: np.ndarray, **sampling: object
    ) -> dict[str, Any]:


        sp = {"temperature": 1.0, "max_new_tokens": 200,
              "skip_special_tokens": False}
        sp.update(sampling)
        body = orjson.dumps(
            {"input_embeds": embeds_np, "sampling_params": sp},
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        resp = self._http.post(
            f"{self.sglang_url}/generate",
            content=body, headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        out = resp.json()
        return out[0] if isinstance(out, list) else out


    def generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> str:
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        )

        embeds_np, _ = self._build_embeds(v, prompt)
        out = self._sglang_generate(embeds_np, **sampling)
        text = out["text"]

        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        if m is None:


            print(f"[NLAClient] WARNING: no <explanation> tags. "
                  f"Raw[:200]={text[:200]!r}")
            return text
        return m.group(1).strip()

    def generate_batch(
        self,
        activations: Iterable[Iterable[float] | np.ndarray | torch.Tensor],
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> list[str]:
        return [self.generate(v, prompt=prompt,
                              extract_explanation=extract_explanation,
                              **sampling)
                for v in activations]


_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


class NLACritic:

    def __init__(self, checkpoint_dir: str | Path, *,
                 device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        checkpoint_dir = Path(checkpoint_dir)
        meta = yaml.safe_load((checkpoint_dir / "nla_meta.yaml").read_text())
        assert meta["role"] in ("critic", "ar"), (
            f"sidecar role={meta['role']!r}, expected 'critic' or 'ar'. "
            f"Point NLACritic at the AR (reconstructor) checkpoint, not the AV."
        )
        ms = meta["extraction"]["mse_scale"]
        assert ms is not None, (
            f"sidecar mse_scale is None (raw-MSE mode). NLACritic.score() is "
            f"direction-only (2(1-cos)) and requires a numeric mse_scale; this "
            f"checkpoint was trained without normalization and is not supported here."
        )
        self.mse_scale: float = float(ms)
        self.template: str = (meta["prompt_templates"].get("ar")
                              or meta["prompt_templates"]["critic"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )


        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        assert bos is None or probe[0] == bos, (
            f"tokenizer has bos_token_id={bos} but add_special_tokens=True "
            f"produced first token {probe[0]}. Critic was trained with BOS "
            f"prefix - reconstruct() must match."
        )


        backbone = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True,
        )


        backbone.lm_head = torch.nn.Identity()
        inner = backbone.model
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"no final-LN attribute on {type(inner).__name__} - tried "
                f"{_FINAL_LN_ATTRS!r}. Add the arch's attr name to that list."
            )

        d = backbone.config.hidden_size
        self.value_head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        head_path = checkpoint_dir / "value_head.safetensors"
        assert head_path.exists(), (
            f"no value_head.safetensors at {checkpoint_dir!r}. NLA critic "
            f"checkpoints ship this alongside config.json - it's the trained "
            f"reconstruction head, not derivable from the backbone."
        )
        self.value_head.load_state_dict(load_file(str(head_path)))

        self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device
        print(f"[NLACritic] {backbone.config.num_hidden_layers} layers  "
              f"d_model={d}  mse_scale={self.mse_scale:.2f}")

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        prompt = self.template.format(explanation=explanation)


        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=True)["input_ids"].to(self.device)
        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]
        return self.value_head(h).float().cpu()

    def score(self, explanation: str,
              original: np.ndarray | torch.Tensor) -> tuple[float, float]:
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        pred_n = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
        return mse, cos


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="HF-format NLA actor dir (with nla_meta.yaml)")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--parquet", default=None,
                    help="Parquet with activation_vector column. Default: "
                         "smoke-test with one random vector.")
    ap.add_argument("--n", type=int, default=3, help="rows to sample from parquet")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--injection-scale", type=float, default=None,
                    help="Override sidecar value (OOD - only if sidecar is "
                         "wrong/missing)")
    ap.add_argument("--prompt", default=None,
                    help="Custom user content with <INJECT> marker. Default: "
                         "sidecar's actor template (recommended).")
    ap.add_argument("--raw", action="store_true",
                    help="Print raw output (no tag extraction)")
    args = ap.parse_args()

    client = NLAClient(
        args.checkpoint,
        sglang_url=args.sglang_url,
        injection_scale_override=args.injection_scale,
    )

    if args.parquet is None:
        print("[smoke] No parquet - generating for one random unit vector.")
        v = np.random.randn(client.cfg.d_model).astype(np.float32)
        out = client.generate(
            v, prompt=args.prompt,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )
        print(f"\n{out}\n")
        return

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.n, columns=["activation_vector"]))

    flat = batch.column("activation_vector").flatten().to_numpy(
        zero_copy_only=False).astype(np.float32)
    vecs = flat.reshape(len(batch), -1)

    for i, v in enumerate(vecs):
        out = client.generate(
            v, prompt=args.prompt,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )
        print(f"─── [{i}]  ||v||={np.linalg.norm(v):.1f} ─────────────────────")
        print(out)
        print()


if __name__ == "__main__":
    _main()
