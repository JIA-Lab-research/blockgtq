"""Vendored LongBench scorer.

Source: THUDM/LongBench @ 2e00731f8d0bff23dc4325161044d0ed8af94c1e
        (LongBench/metrics.py + LongBench/eval.py)
Pinned date: 2025-01-15.

The metric functions are byte-equivalent to upstream `metrics.py`. The wrapper
``score_subtask`` matches the call signature in the spec:

    score_subtask(subtask, predictions, references, all_classes=None)
        -> {"score": float, "metric": str, "n": int}

Score is reported on the [0, 100] scale (LongBench convention) so the headline
``avg_normalized`` averages cleanly across heterogeneous metrics.
"""
import re
import string
from collections import Counter
from typing import List, Optional, Sequence

from fuzzywuzzy import fuzz
from rouge import Rouge

from bench.longbench_config import METRIC_NAMES, SUBTASKS


# ============================================================================
# Vendored normalize/score primitives — verbatim from upstream metrics.py
# ============================================================================

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction_tokens, ground_truth_tokens) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    p_tok = _normalize_answer(prediction).split()
    g_tok = _normalize_answer(ground_truth).split()
    if not p_tok or not g_tok:
        return 0.0
    return _f1_score(p_tok, g_tok)


def _rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def _retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    right = sum(1 for n in nums if str(n) == str(gt_id))
    return right / len(nums)


def _code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    # Take first non-comment line — upstream behavior.
    pred = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            pred = line
            break
    return fuzz.ratio(pred, ground_truth) / 100.0


def _classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_classes = kwargs.get("all_classes") or []
    em = [c for c in all_classes if c in prediction]
    # Drop classes that are substrings of the ground truth but not equal.
    em = [c for c in em if not (c in ground_truth and c != ground_truth)]
    if ground_truth in em:
        return 1.0 / len(em)
    return 0.0


# ============================================================================
# Subtask → metric routing (matches LongBench/eval.py:dataset2metric)
# ============================================================================

_DATASET2METRIC = {
    "qasper":               _qa_f1_score,
    "multifieldqa_en":      _qa_f1_score,
    "hotpotqa":             _qa_f1_score,
    "2wikimqa":             _qa_f1_score,
    "gov_report":           _rouge_score,
    "trec":                 _classification_score,
    "passage_retrieval_en": _retrieval_score,
    "lcc":                  _code_sim_score,
    # KIVI-aligned LongBench-8 (NeurIPS'24 KIVI paper) additions:
    "qmsum":                _rouge_score,
    "multi_news":           _rouge_score,
    "triviaqa":             _qa_f1_score,
    "samsum":               _rouge_score,
    "repobench-p":          _code_sim_score,
}

# Subtasks where upstream scorer strips first newline + line of prediction.
_FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}


def _per_example_score(subtask: str, prediction: str,
                       ground_truths: Sequence[str],
                       all_classes: Optional[Sequence[str]]) -> float:
    """Match LongBench/eval.py:scorer per-example logic."""
    pred = prediction
    if subtask in _FIRST_LINE_ONLY:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = _DATASET2METRIC[subtask]
    best = 0.0
    for gt in ground_truths:
        s = fn(pred, gt, all_classes=all_classes)
        if s > best:
            best = s
    return best


# ============================================================================
# Public API
# ============================================================================

def score_subtask(subtask: str,
                  predictions: List[str],
                  references: List[List[str]],
                  all_classes: Optional[List[str]] = None) -> dict:
    """Score one subtask following the LongBench protocol.

    Args:
        subtask: name in SUBTASKS.
        predictions: list[str], one prediction per example.
        references: list[list[str]] — multiple acceptable answers per example.
            Strings are also accepted and wrapped to one-element lists.
        all_classes: only used by classification (`trec`) scorer.

    Returns:
        {"score": float in [0, 100], "metric": metric_name, "n": int}
    """
    if subtask not in _DATASET2METRIC:
        raise ValueError(f"Unsupported subtask {subtask!r}. "
                         f"Known: {sorted(SUBTASKS)}")
    if len(predictions) != len(references):
        raise ValueError(f"len(predictions)={len(predictions)} != "
                         f"len(references)={len(references)}")
    n = len(predictions)
    if n == 0:
        return {"score": float("nan"), "metric": METRIC_NAMES[subtask], "n": 0}

    total = 0.0
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        total += _per_example_score(subtask, pred, refs, all_classes)
    score = round(100.0 * total / n, 2)
    return {"score": score, "metric": METRIC_NAMES[subtask], "n": n}


__all__ = ["score_subtask", "SUBTASKS", "METRIC_NAMES"]
