"""LongBench-EN evaluation with Block-GTQ KV-cache compression.

Public port of the research LongBench runner: uses
``blockgtq.unaccelerated.patch_model_kv`` (pure-torch Block-GTQ on K +
TurboQuant-MSE on V, applied as quantize-dequantize on the pre-RoPE
projections) plus the vendored LongBench config + scorer
(``bench/longbench_config``, ``bench/_longbench_scorer``, both from
THUDM/LongBench @ 2e00731, MIT).

The LongBench **data** is not bundled. Download ``data.zip`` from
huggingface.co/datasets/THUDM/LongBench, extract the per-subtask ``.jsonl``
files, and point ``$LONGBENCH_DATA_DIR`` at the directory.

Usage::

    LONGBENCH_DATA_DIR=/path/to/longbench/data CC=gcc \\
    python bench/bench_longbench.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --calib-cache calib_caches/calib_Llama-3.1-8B.pt \\
        --k-bits 3 --v-bits 3 \\
        --subtasks qasper hotpotqa multifieldqa_en passage_retrieval_en

Calibration uses the pre-RoPE Q/K activations saved by
``examples/build_calib_cache.py`` (the patch quantizes the pre-RoPE k_proj
output, so the calibrate point matches the apply point). Pass ``--fp16`` to
score the uncompressed baseline.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `python bench/bench_longbench.py ...` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.longbench_config import (
    SUBTASKS, MAX_NEW_TOKENS, PROMPT_TEMPLATES, METRIC_NAMES,
    DEFAULT_INPUT_CAP, LONGBENCH_PIN,
)
from bench._longbench_scorer import score_subtask
from blockgtq.unaccelerated import (
    build_unaccelerated_quantizers, patch_model_kv, unpatch_model_kv,
)

_LB_DATA_DIR = Path(os.environ.get(
    "LONGBENCH_DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "longbench_data")))
RESULTS_DIR = Path(os.environ.get(
    "BLOCKGTQ_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "longbench_results")))
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Subtasks where upstream pred.py skips the chat-template wrapper.
_SKIP_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def load_subtask(subtask, max_samples=None, offset=0):
    fpath = _LB_DATA_DIR / f"{subtask}.jsonl"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Missing {fpath}. Set $LONGBENCH_DATA_DIR to the directory of "
            f"<subtask>.jsonl files extracted from THUDM/LongBench data.zip.")
    out = []
    with fpath.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if max_samples is not None and len(out) >= max_samples:
                break
            out.append(json.loads(line))
    return out


def is_instruct_model(name):
    n = name.lower()
    return ("instruct" in n) or ("r1-distill" in n) or ("-chat" in n)


def truncate_middle(prompt, tokenizer, max_tokens):
    """Keep first half + last half of tokens when too long (LongBench rule)."""
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return prompt
    half = max_tokens // 2
    return (tokenizer.decode(toks[:half], skip_special_tokens=True)
            + tokenizer.decode(toks[-half:], skip_special_tokens=True))


def build_prompt(ex, tokenizer, subtask, max_input, instruct_model):
    raw = PROMPT_TEMPLATES[subtask].format(context=ex["context"], input=ex["input"])
    prompt = truncate_middle(raw, tokenizer, max_input)
    if instruct_model and subtask not in _SKIP_CHAT:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    return prompt


@torch.no_grad()
def run_subtask(model, tokenizer, device, subtask, max_new, max_input,
                instruct_model, max_samples=None):
    ds = load_subtask(subtask, max_samples=max_samples)
    add_special = not (instruct_model and subtask not in _SKIP_CHAT)
    pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)
    preds, refs = [], []
    all_classes = None
    t0 = time.time()
    for ex in ds:
        prompt = build_prompt(ex, tokenizer, subtask, max_input, instruct_model)
        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=add_special).to(device)
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)
        preds.append(gen)
        refs.append(list(ex["answers"]))
        if all_classes is None and ex.get("all_classes"):
            all_classes = list(ex["all_classes"])
    sc = score_subtask(subtask, preds, refs, all_classes=all_classes)
    sc["wall_time_s"] = round(time.time() - t0, 1)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--calib-cache", default=None,
                    help="Path to the .pt from examples/build_calib_cache.py "
                         "(required unless --fp16).")
    ap.add_argument("--k-bits", type=float, default=3.0,
                    help="Average K bits/dim (allocated non-uniformly).")
    ap.add_argument("--v-bits", type=int, default=3)
    ap.add_argument("--subtasks", nargs="+", default=list(SUBTASKS),
                    choices=sorted(PROMPT_TEMPLATES))
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap samples per subtask (smoke / quick validation).")
    ap.add_argument("--max-input", type=int, default=DEFAULT_INPUT_CAP)
    ap.add_argument("--fp16", action="store_true",
                    help="Score the uncompressed fp16 baseline (no patching).")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    if not args.fp16 and not args.calib_cache:
        ap.error("--calib-cache is required unless --fp16")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="flash_attention_2").to(args.device).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    instruct_model = is_instruct_model(args.model)

    method = "fp16"
    hooks = None
    if not args.fp16:
        # Build Block-GTQ (K) + TurboQuant-MSE (V) per (layer, head) from the
        # calib cache, then patch. calib_q holds all GQA query heads as samples;
        # calib_k is pre-RoPE — exactly what the unaccelerated patch expects.
        ck = torch.load(args.calib_cache, map_location="cpu", weights_only=False)
        ck_k, ck_q = ck["calib_k"], ck["calib_q"]
        if ck_k.shape[1] != nkv or ck_k.shape[3] != hd:
            raise ValueError(
                f"calib cache n_kv/hd {tuple(ck_k.shape)} != model nkv={nkv} hd={hd}")
        layer_data = {li: {"q": ck_q[li], "k": ck_k[li]} for li in range(n_layers)}
        kq, vq = build_unaccelerated_quantizers(
            layer_data, n_layers, nkv, hd,
            k_avg_bits=args.k_bits, v_bits=args.v_bits, device=args.device)
        hooks = patch_model_kv(model, kq, vq, hd, nkv)
        method = f"Block-GTQ-K{int(args.k_bits)}V{args.v_bits}"

    print(f"[longbench] model={args.model} method={method} "
          f"subtasks={args.subtasks} max_input={args.max_input} "
          f"max_samples={args.max_samples} pin={LONGBENCH_PIN}")
    try:
        subtask_scores = {}
        for st in args.subtasks:
            sc = run_subtask(model, tokenizer, args.device, st,
                             MAX_NEW_TOKENS[st], args.max_input,
                             instruct_model, max_samples=args.max_samples)
            print(f"  [{st}] {sc['score']} ({sc['metric']}, n={sc['n']}) "
                  f"{sc['wall_time_s']}s")
            subtask_scores[st] = sc
        scores = [s["score"] for s in subtask_scores.values()
                  if s["n"] > 0 and s["score"] == s["score"]]
        avg = round(sum(scores) / max(len(scores), 1), 2) if scores else float("nan")
        print(f"  -> avg_normalized = {avg}")
    finally:
        if hooks is not None:
            unpatch_model_kv(hooks)

    ts = args.out_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = RESULTS_DIR / f"longbench_{ts}.json"
    with open(out_json, "w") as f:
        json.dump({
            "model": args.model, "method": method,
            "k_bits": args.k_bits, "v_bits": args.v_bits,
            "max_input": args.max_input, "max_samples": args.max_samples,
            "longbench_pin": LONGBENCH_PIN,
            "subtasks": subtask_scores, "avg_normalized": avg,
        }, f, indent=2, default=str)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
