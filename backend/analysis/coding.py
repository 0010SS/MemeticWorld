"""Blind Sonnet coding of free text (OBSERVER ONLY; ontology v3 §5.10). STUB (MVP scope cut).

Planned codebooks (new prompts code_implication_v1.txt, code_record_v1.txt, code_function_v1.txt): implication of
N/P free text, record-entry recommended action (the keyword-map fallback in record_lineage), pragmatic function
of X/W/Y usages. Blind (condition, day, regime, author stripped), shuffled across runs, 10 items per call, on an
observer client with scope `code:`; validation by a paraphrased second pass, Haiku as second coder and the
120-item human kappa sample (`scripts/export_validation_sample.py`, also deferred). Writes `coding.jsonl`.
Config read when implemented: analysis.coder.{backend, model, second, blind, human_sample}.
"""
from __future__ import annotations

CODEBOOKS = ("implication", "record", "function")


def code_items(items, codebook: str, cfg: dict | None = None):
    raise NotImplementedError("Sonnet coding is deferred (v3 MVP scope cut)")


def kappa(codes_a, codes_b):
    raise NotImplementedError("coder agreement / human kappa sample is deferred (v3 MVP scope cut)")
