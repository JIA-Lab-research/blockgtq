#!/usr/bin/env python3
"""AIME 2024/2025 evaluation under Block-GTQ / TQ-MSE KV-cache compression.

Reasoning-task fidelity benchmark. DeepSeek-R1-distilled models emit long
chain-of-thought (up to 32K tokens), so the KV cache grows during decoding —
a stress test for quality preservation under K/V quantization.

The quantize path quantizes-then-dequantizes every K/V projection on the fly
(no fp16 recent-key buffer kept), via the ``blockgtq.unaccelerated`` module
(``patch_model_kv`` / ``unpatch_model_kv``).

Methods (external baselines such as KIVI / PM-KVQ are out of scope here):

  * ``Block-GTQ`` — RoPE-aware non-uniform allocation: per-(layer, KV-head)
    ``BlockGTQPipeline(min_bits=1, max_bits=8)`` calibrated on the energy
    score. Built via ``build_unaccelerated_quantizers``.
  * ``TQ-MSE``    — uniform baseline: per-head **full-head** ``TurboQuantMSE``
    (NOT a block-uniform ``BlockGTQPipeline``; see ``build_kv_quantizers``).

V is identical for both methods (per-head ``TurboQuantMSE`` at ``v_bits``), so
the comparison isolates the K-side allocation.

Calibration: the model is forwarded once on the first ``--n-calib-tokens``
(default 2048) of ``--calib-prompt-file`` (default the bundled
``calib_data/wikitext2_calib_2k.txt``), pre-RoPE
(``collect_qk_activations(post_rope=False)``).

Usage:
    python bench/bench_aime.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --device cuda:0 --datasets aime24 aime25 \\
        --k-bits 3 --v-bits 2 --methods Block-GTQ TQ-MSE \\
        --seeds 0
"""
import sys
import os
import time
import json
import re
import gc
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockgtq import collect_qk_activations
from blockgtq.unaccelerated import (
    build_unaccelerated_quantizers, patch_model_kv, unpatch_model_kv,
    patch_model_kv_buffered,
)
from blockgtq.tq import TurboQuantMSE

RESULTS_DIR = Path(os.environ.get(
    "BLOCKGTQ_RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_CALIB = (Path(__file__).resolve().parent.parent
                 / "calib_data" / "wikitext2_calib_2k.txt")


# ============================================================================
# AIME data loading
# ============================================================================

def load_aime_dataset(dataset_name, max_problems=None):
    """Load an AIME dataset by name. Returns list of {problem, answer, id}."""
    from datasets import load_dataset

    if dataset_name == 'aime24':
        ds = load_dataset('HuggingFaceH4/aime_2024', split='train')
        problems = [{
            'problem': item['problem'],
            'answer': str(item['answer']).strip(),
            'id': item.get('id', f'aime24_{i}'),
        } for i, item in enumerate(ds)]
    elif dataset_name == 'aime25':
        try:
            ds = load_dataset('opencompass/AIME2025', 'AIME2025-I', split='test')
            problems_i = [{
                'problem': item['question'],
                'answer': str(item['answer']).strip(),
                'id': f'aime25_I_{i}',
            } for i, item in enumerate(ds)]
        except Exception:
            problems_i = []
        try:
            ds = load_dataset('opencompass/AIME2025', 'AIME2025-II', split='test')
            problems_ii = [{
                'problem': item['question'],
                'answer': str(item['answer']).strip(),
                'id': f'aime25_II_{i}',
            } for i, item in enumerate(ds)]
        except Exception:
            problems_ii = []
        problems = problems_i + problems_ii
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_problems is not None:
        problems = problems[:max_problems]
    return problems


# ============================================================================
# Answer extraction
# ============================================================================

def _find_all_boxed(text):
    r"""Return contents inside every \boxed{...} with proper brace matching."""
    out = []
    i = 0
    needle = '\\boxed{'
    while True:
        idx = text.find(needle, i)
        if idx == -1:
            break
        start = idx + len(needle)
        depth = 1
        end = start
        while end < len(text) and depth > 0:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        if depth == 0:
            out.append(text[start:end - 1])
        i = end
    return out


def extract_aime_answer(text):
    r"""Extract the final integer answer from a chain-of-thought.

    AIME answers are integers in [0, 999], placed in \boxed{...}. We walk all
    \boxed{...} (brace-matched, last wins), then fall back to "answer is N" and
    finally the last integer in the text.
    """
    if not text:
        return None

    boxes = _find_all_boxed(text)
    if boxes:
        m = re.search(r'-?\d+', boxes[-1])
        if m:
            return m.group(0)

    m = re.search(r'(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?\\?-?(\d+)',
                  text, re.IGNORECASE)
    if m:
        return m.group(1)

    nums = re.findall(r'-?\d+', text)
    if nums:
        return nums[-1]

    return None


def is_correct(predicted, gold):
    """Compare predicted vs gold AIME answer numerically (digits only)."""
    if predicted is None:
        return False
    pred_digits = re.sub(r'\D', '', str(predicted))
    gold_digits = re.sub(r'\D', '', str(gold))
    if not pred_digits or not gold_digits:
        return False
    try:
        return int(pred_digits) == int(gold_digits)
    except (ValueError, TypeError):
        return False


def _finalize_record(rec):
    """Add per-problem aggregate fields over the ``responses`` list."""
    responses = rec.get('responses', [])
    n = len(responses)
    if n == 0:
        return {**rec, 'avg_correct': 0.0, 'avg_gen_tokens': 0.0}
    return {
        **rec,
        'avg_correct': sum(int(r.get('correct', 0)) for r in responses) / n,
        'avg_gen_tokens': sum(int(r.get('n_tokens', 0)) for r in responses) / n,
    }


# ============================================================================
# Prompt + generation
# ============================================================================

def build_aime_prompt(problem, tokenizer, model_family=None):
    r"""Chat-templated AIME prompt, model-family aware (DSR1 / Qwen3 / GPT-OSS)."""
    user_msg = (
        f"{problem}\n\n"
        f"Please reason step by step, and put your final answer within "
        f"\\boxed{{}}."
    )
    family = (model_family or getattr(tokenizer, 'name_or_path', '')).lower()

    if 'qwen3' in family and 'distill' not in family:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)

    if 'gpt-oss' in family or 'gptoss' in family:
        cot_msg = (
            "Solve the following AIME problem. Think carefully and show "
            "all reasoning steps before giving the final answer.\n\n"
            f"{problem}\n\n"
            "Format: first write your reasoning, then end with "
            r"`Final answer: \boxed{<integer>}`."
        )
        messages = [{"role": "user", "content": cot_msg}]
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"User: {cot_msg}\nAssistant:"

    # DSR1 family (default).
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": user_msg}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            if not prompt_text.rstrip().endswith("<think>"):
                prompt_text = prompt_text + "<think>\n"
            return prompt_text
        except Exception:
            pass
    return f"User: {user_msg}\nAssistant: <think>\n"


@torch.no_grad()
def generate_aime_batch(model, tokenizer, prompts, device, max_new=32768,
                        temperature=0.6, top_p=0.95, seed=0, n_responses=1):
    """Batched generation. Returns a list of
    ``(problem_idx, response_idx, gen_text, n_gen_tokens, prompt_len)``.

    ``n_responses>1`` routes through ``num_return_sequences`` (one generate()
    call emits N trajectories that share the prompt cache). Left-padding is set
    for batched causal decode. RNG seeded once per call for reproducibility.
    """
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)

    saved_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=False,
        ).to(device)
        prompt_lens = inputs['attention_mask'].sum(dim=1).tolist()
        padded_len = inputs['input_ids'].shape[1]

        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_responses,
            pad_token_id=tokenizer.eos_token_id,
        )

        results = []
        for i in range(out.shape[0]):
            problem_i = i // n_responses
            response_r = i % n_responses
            gen_ids = out[i, padded_len:]
            mask = gen_ids != tokenizer.eos_token_id
            last_real = int(mask.nonzero()[-1].item()) + 1 if mask.any() else 0
            gen_ids = gen_ids[:last_real].cpu()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append((
                problem_i, response_r, gen_text,
                int(gen_ids.shape[0]), int(prompt_lens[problem_i]),
            ))
        return results
    finally:
        tokenizer.padding_side = saved_side


# ============================================================================
# Quantizer construction
# ============================================================================

def build_kv_quantizers(method, layer_data, n_layers, nkv, hd, kb, vb, device):
    """Build ``[li][hi]`` K and V quantizer lists for one method.

    * ``Block-GTQ`` → ``build_unaccelerated_quantizers`` (K =
      ``BlockGTQPipeline(min_bits=1, max_bits=8)`` calibrated on ``layer_data``;
      V = ``TurboQuantMSE(vb, seed=1000+h)``).
    * ``TQ-MSE`` → K = **direct full-head** ``TurboQuantMSE(kb, seed=42+h)``
      (NOT a block-uniform ``BlockGTQPipeline``); V identical to above.

    V is constructed identically for both methods, so the only difference is
    the K-side bit allocation. Both quantizer classes expose
    ``compress_decompress`` and are applied via the public ``patch_model_kv``.
    """
    if method == "Block-GTQ":
        return build_unaccelerated_quantizers(
            layer_data, n_layers, nkv, hd,
            k_avg_bits=float(kb), v_bits=int(vb),
            min_bits=1, max_bits=8, v_seed_base=1000, device=device)

    if method == "TQ-MSE":
        dev = torch.device(device)
        kq = [[TurboQuantMSE(d=hd, bit_width=int(kb), seed=42 + h, device=dev)
               for h in range(nkv)] for _ in range(n_layers)]
        vq = [[TurboQuantMSE(d=hd, bit_width=int(vb), seed=1000 + h, device=dev)
               for h in range(nkv)] for _ in range(n_layers)]
        return kq, vq

    raise ValueError(f"Unknown method: {method!r} (use Block-GTQ or TQ-MSE)")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=["aime24"],
                    help="aime24 and/or aime25")
    ap.add_argument("--k-bits", type=int, default=3, help="K avg bits per dim")
    ap.add_argument("--v-bits", type=int, default=2, help="V bits per dim")
    ap.add_argument("--methods", nargs="+", default=["fp16", "Block-GTQ", "TQ-MSE"],
                    help="Subset of: fp16 Block-GTQ TQ-MSE")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-responses", type=int, default=1,
                    help="Trajectories per problem via num_return_sequences. "
                         ">1 forces batch-size=1.")
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--problem-start", type=int, default=None,
                    help="Per-dataset start index (inclusive) for sharding.")
    ap.add_argument("--problem-end", type=int, default=None,
                    help="Per-dataset end index (exclusive) for sharding.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Problems per generate() call (forced to 1 when "
                         "n_responses>1).")
    ap.add_argument("--calib-prompt-file", default=str(DEFAULT_CALIB),
                    help="Held-out text for quantizer calibration (paper used "
                         "wikitext2_calib_2k.txt).")
    ap.add_argument("--n-calib-tokens", type=int, default=2048)
    ap.add_argument("--setting", choices=["nobuffer", "withbuffer"],
                    default="nobuffer",
                    help="nobuffer: every token quantized (stress). "
                         "withbuffer: sink + recent-fp16-window + quantized "
                         "middle (PM-KVQ-aligned).")
    ap.add_argument("--recent-window", type=int, default=128,
                    help="withbuffer: # most-recent tokens kept fp16.")
    ap.add_argument("--n-sink", type=int, default=4,
                    help="withbuffer: # sink (prefix) tokens kept fp16.")
    ap.add_argument("--block-size", type=int, default=1,
                    help="withbuffer: # tokens flushed per block.")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    valid = {"fp16", "Block-GTQ", "TQ-MSE"}
    bad = [m for m in args.methods if m not in valid]
    if bad:
        raise ValueError(f"--methods must be in {valid}, got {bad}")

    if args.n_responses > 1 and args.batch_size != 1:
        print(f"[warn] n_responses={args.n_responses}>1; forcing batch_size=1.")
        args.batch_size = 1

    kb, vb = args.k_bits, args.v_bits
    ts = args.out_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    ms_label = args.model.replace('/', '_')
    log_file = RESULTS_DIR / f"aime_{ms_label}_K{kb}V{vb}_{args.setting}_{ts}.md"
    json_file = RESULTS_DIR / f"aime_{ms_label}_K{kb}V{vb}_{args.setting}_{ts}.json"

    lines = [
        f"# AIME (Block-GTQ / TQ-MSE, K+V compressed) — {ts}", "",
        f"**Model:** {args.model}",
        f"**Datasets:** {args.datasets}",
        f"**(K,V) bits:** ({kb}, {vb})",
        f"**Methods:** {args.methods}",
        f"**Seeds:** {args.seeds}  **Responses/problem:** {args.n_responses}",
        f"**Generation:** temperature={args.temperature}, top_p={args.top_p}, "
        f"max_new={args.max_new}",
        f"**Calib:** {args.calib_prompt_file} (first {args.n_calib_tokens} "
        f"tokens, pre-RoPE).",
        f"**Setting:** {args.setting}"
        + (f" (sink={args.n_sink}, window={args.recent_window}, "
           f"block={args.block_size})" if args.setting == "withbuffer"
           else " (no buffer — every token quantized)"),
        f"**V quantizer:** TurboQuant-MSE for both methods (K-side isolates "
        f"the allocation).", "",
    ]

    def log(msg):
        print(msg)
        lines.append(msg)
        Path(log_file).write_text("\n".join(lines))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(args.device).eval()
    except (ImportError, ValueError):
        # flash-attn unavailable (not installed / no CUDA): fall back to the
        # default attention implementation.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(args.device).eval()

    hd = getattr(model.config, 'head_dim',
                 model.config.hidden_size // model.config.num_attention_heads)
    nkv = getattr(model.config, 'num_key_value_heads',
                  model.config.num_attention_heads)
    n_layers = model.config.num_hidden_layers
    model_family = args.model.lower()
    log(f"  hd={hd}, n_kv={nkv}, n_layers={n_layers}")

    # ── Offline calibration (once per run): pre-RoPE Q+K on the calib prompt,
    #    wrapped through build_aime_prompt for tokenization parity. Only needed
    #    for Block-GTQ; TQ-MSE is data-free.
    layer_data = None
    if "Block-GTQ" in args.methods:
        calib_text = Path(args.calib_prompt_file).read_text()
        calib_prompt = build_aime_prompt(calib_text, tokenizer, model_family)
        calib_ids = tokenizer(calib_prompt, return_tensors='pt')[
            'input_ids'][:, :args.n_calib_tokens].to(args.device)
        log(f"Calibrating Block-GTQ on {calib_ids.shape[1]} tokens "
            f"(pre-RoPE, GQA-averaged Q)...")
        layer_data = collect_qk_activations(
            model, calib_ids, args.device,
            n_calib_tokens=calib_ids.shape[1], post_rope=False)

    method_order = [m for m in ("fp16", "Block-GTQ", "TQ-MSE")
                    if m in args.methods]
    all_results = []

    for dataset_name in args.datasets:
        log(f"\n## Dataset: {dataset_name}\n")
        problems = load_aime_dataset(dataset_name, args.max_problems)
        if args.problem_start is not None or args.problem_end is not None:
            s = args.problem_start or 0
            e = args.problem_end if args.problem_end is not None else len(problems)
            problems = problems[s:e]
            log(f"Sliced {dataset_name} to [{s}:{e}] = {len(problems)} problems")
        else:
            log(f"Loaded {len(problems)} problems")
        log("")
        log("| Method | Acc | (n_correct / n_total) | Avg gen tokens |")
        log("|--------|-----|----------------------|----------------|")

        prompts = [build_aime_prompt(p['problem'], tokenizer, model_family)
                   for p in problems]
        golds = [p['answer'] for p in problems]
        ids = [p['id'] for p in problems]
        n_problems = len(problems)

        for method in method_order:
            handles = None
            buf_patch = None
            if method != "fp16":
                kq, vq = build_kv_quantizers(
                    method, layer_data, n_layers, nkv, hd, kb, vb, args.device)
                if args.setting == "withbuffer":
                    buf_patch = patch_model_kv_buffered(
                        model, kq, vq, hd, nkv, n_layers,
                        block_size=args.block_size,
                        recent_window=args.recent_window,
                        n_sink_token=args.n_sink)
                else:
                    handles = patch_model_kv(model, kq, vq, hd, nkv)

            records = [{'id': ids[i], 'gold': golds[i], 'responses': []}
                       for i in range(n_problems)]

            # Resume: pre-populate from a prior partial JSON for this cell.
            if Path(json_file).exists():
                try:
                    old = json.load(open(json_file))
                    for cell in old.get('results', []):
                        if (cell.get('method') == method
                                and cell.get('dataset') == dataset_name
                                and int(cell.get('n_responses', 0)) == args.n_responses):
                            old_by_id = {r['id']: r for r in cell.get('records', [])}
                            resumed = 0
                            for i in range(n_problems):
                                old_r = old_by_id.get(ids[i], {}).get('responses', [])
                                if len(old_r) >= args.n_responses:
                                    records[i]['responses'] = old_r
                                    resumed += 1
                            if resumed:
                                log(f"  [resume] {method}/{dataset_name}: "
                                    f"{resumed}/{n_problems} problems reused")
                            break
                except Exception as e:
                    log(f"  [resume] could not load {json_file}: {e}")

            method_t0 = time.time()
            for seed in args.seeds:
                for cs in range(0, n_problems, args.batch_size):
                    ce = min(cs + args.batch_size, n_problems)
                    if all(len(records[i]['responses']) >= args.n_responses
                           for i in range(cs, ce)):
                        continue
                    if buf_patch is not None:
                        buf_patch.reset()
                    t0 = time.time()
                    try:
                        batch = generate_aime_batch(
                            model, tokenizer, prompts[cs:ce], args.device,
                            max_new=args.max_new, temperature=args.temperature,
                            top_p=args.top_p, seed=seed,
                            n_responses=args.n_responses)
                    except torch.cuda.OutOfMemoryError as e:
                        log(f"  OOM {method} seed={seed} [{cs}:{ce}]: {e}")
                        torch.cuda.empty_cache()
                        continue
                    elapsed = time.time() - t0

                    for plocal, _r, gen_text, n_gen, plen in batch:
                        gi = cs + plocal
                        pred = extract_aime_answer(gen_text)
                        records[gi]['responses'].append({
                            'response_idx': len(records[gi]['responses']),
                            'seed': seed,
                            'gen_text_tail': gen_text[-2000:],
                            'n_tokens': n_gen, 'prompt_len': plen,
                            'pred': pred,
                            'correct': int(is_correct(pred, golds[gi])),
                        })

                    # Running aggregate.
                    nc, nt = 0.0, 0
                    for r in records:
                        rs = r['responses']
                        if rs:
                            nc += sum(int(x['correct']) for x in rs) / len(rs)
                            nt += 1
                    log(f"  {method} seed={seed} [{cs}:{ce}] {elapsed:.0f}s "
                        f"running avg@N={nc:.1f}/{nt} ({nc/max(1,nt):.1%})")

                    # Checkpoint after each chunk.
                    cell = _result_cell(dataset_name, method, kb, vb,
                                        args.n_responses, records)
                    cell['partial'] = True
                    _write_json(json_file, args, n_layers, nkv, hd,
                                _merge_cell(all_results, cell))

            if handles is not None:
                unpatch_model_kv(handles)
            if buf_patch is not None:
                buf_patch.remove()
            torch.cuda.empty_cache()
            gc.collect()

            cell = _result_cell(dataset_name, method, kb, vb,
                                args.n_responses, records)
            cell['method_wall_sec'] = round(time.time() - method_t0, 1)
            log(f"| {method} | {cell['accuracy']:.1%} | "
                f"({cell['n_correct']:.1f}/{cell['n_total']}) | "
                f"{cell['avg_gen_tokens']:.0f} |")
            all_results = _merge_cell(all_results, cell)
            _write_json(json_file, args, n_layers, nkv, hd, all_results)

    log(f"\nSaved log to {log_file}")
    log(f"Saved JSON to {json_file}")


def _result_cell(dataset_name, method, kb, vb, n_responses, records):
    nc, nt, tg = 0.0, 0, 0.0
    for r in records:
        rs = r['responses']
        if rs:
            nc += sum(int(x['correct']) for x in rs) / len(rs)
            tg += sum(int(x['n_tokens']) for x in rs) / len(rs)
            nt += 1
    return {
        'dataset': dataset_name, 'method': method,
        'k_bits': kb, 'v_bits': vb, 'n_responses': n_responses,
        'accuracy': nc / max(1, nt), 'n_correct': nc, 'n_total': nt,
        'avg_gen_tokens': tg / max(1, nt),
        'records': [_finalize_record(r) for r in records],
    }


def _merge_cell(all_results, cell):
    out = [r for r in all_results
           if not (r.get('method') == cell['method']
                   and r.get('dataset') == cell['dataset'])]
    out.append(cell)
    return out


def _write_json(json_file, args, n_layers, nkv, hd, results):
    with open(json_file, 'w') as f:
        json.dump({
            'model': args.model, 'kv_compressed': True,
            'temperature': args.temperature, 'top_p': args.top_p,
            'max_new_tokens': args.max_new, 'seeds': args.seeds,
            'n_responses': args.n_responses, 'batch_size': args.batch_size,
            'n_layers': n_layers, 'n_kv_heads': nkv, 'head_dim': hd,
            'calib_prompt_file': args.calib_prompt_file,
            'n_calib_tokens': args.n_calib_tokens,
            'setting': args.setting, 'recent_window': args.recent_window,
            'n_sink': args.n_sink, 'block_size': args.block_size,
            'results': results,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
