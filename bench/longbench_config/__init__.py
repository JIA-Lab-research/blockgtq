"""LongBench-EN subset configuration for the K+V cache compression campaign.

The 8 chosen subtasks span all 6
LongBench categories, biased toward retrieval-style and multi-doc tasks
where KV-cache compression is most likely to break.

Pinned to LongBench commit 2e00731f8d0bff23dc4325161044d0ed8af94c1e
(THUDM/LongBench main, 2025-01-15). Files dataset2prompt.json and
dataset2maxlen.json are byte-equivalent slices of the upstream config files
restricted to the 8 chosen subtasks.

Use:
    from experiments.longbench_config import (
        SUBTASKS, MAX_NEW_TOKENS, PROMPT_TEMPLATES, METRIC_NAMES,
    )
"""
import json
from pathlib import Path


_HERE = Path(__file__).parent

with (_HERE / "dataset2prompt.json").open() as _f:
    PROMPT_TEMPLATES: dict[str, str] = json.load(_f)
with (_HERE / "dataset2maxlen.json").open() as _f:
    MAX_NEW_TOKENS: dict[str, int] = json.load(_f)

# Order matters for the headline `avg_normalized`: mean across the 8
# subtasks. Keep alphabetic to match LongBench leaderboard ordering.
SUBTASKS: tuple[str, ...] = (
    "2wikimqa",
    "gov_report",
    "hotpotqa",
    "lcc",
    "multifieldqa_en",
    "passage_retrieval_en",
    "qasper",
    "trec",
)

# KIVI paper (NeurIPS 2024) reports a different 8-task subset of LongBench-EN
# (Qasper, QMSum, MultiNews, TREC, TriviaQA, SAMSum, LCC, RepoBench-P).
# Used for the KIVI-aligned main table — direct comparison with their numbers.
SUBTASKS_KIVI: tuple[str, ...] = (
    "lcc",
    "multi_news",
    "qasper",
    "qmsum",
    "repobench-p",
    "samsum",
    "trec",
    "triviaqa",
)

# Per LongBench/eval.py upstream — these names are what we report next to the
# numerical score in the JSON output.
METRIC_NAMES: dict[str, str] = {
    "qasper":               "qa_f1",
    "multifieldqa_en":      "qa_f1",
    "hotpotqa":             "qa_f1",
    "2wikimqa":             "qa_f1",
    "gov_report":           "rouge_l",
    "trec":                 "classification",
    "passage_retrieval_en": "retrieval",
    "lcc":                  "code_sim",
    # KIVI-aligned additions:
    "qmsum":                "rouge_l",
    "multi_news":           "rouge_l",
    "triviaqa":             "qa_f1",
    "samsum":               "rouge_l",
    "repobench-p":          "code_sim",
}

# Upstream commit pinned in vendored config + scorer.
LONGBENCH_PIN = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"

# 31.5K input cap matching the LongBench protocol default. Models with
# narrower context windows (e.g. Qwen2.5-3B at 32K) override this in
# exp_longbench_kv.py.
DEFAULT_INPUT_CAP = 31500


def assert_consistent() -> None:
    """Sanity check: prompt/maxlen/metric tables cover both 8-task panels."""
    panel = set(SUBTASKS) | set(SUBTASKS_KIVI)
    assert panel.issubset(PROMPT_TEMPLATES), \
        f"prompts missing {panel - set(PROMPT_TEMPLATES)}"
    assert panel.issubset(MAX_NEW_TOKENS), \
        f"maxlen missing {panel - set(MAX_NEW_TOKENS)}"
    assert panel.issubset(METRIC_NAMES), \
        f"metrics missing {panel - set(METRIC_NAMES)}"


assert_consistent()
