"""Support-vector and attention-feature selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import CodeSample
from .logging_utils import write_json
from .surrogate import attention_top_tokens, tokenize_code


@dataclass(frozen=True)
class FeatureBundle:
    sample_idx: int
    key_tokens: list[str]
    support_vector_indices: list[int]
    token_scores: list[tuple[str, float]]


def select_non_vulnerable_support_vectors(
    samples: list[CodeSample],
    kernel: str = "rbf",
    max_support_vectors: int = 200,
) -> list[int]:
    non_vulnerable = [sample.idx for sample in samples if sample.label == 0]
    if len({sample.label for sample in samples}) < 2:
        return non_vulnerable[:max_support_vectors]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import SVC

        vectorizer = TfidfVectorizer(tokenizer=tokenize_code, token_pattern=None, max_features=5000)
        matrix = vectorizer.fit_transform([sample.func for sample in samples])
        labels = [sample.label for sample in samples]
        svm = SVC(kernel=kernel)
        svm.fit(matrix, labels)
        support_ids = [samples[index].idx for index in svm.support_ if samples[index].label == 0]
        return support_ids[:max_support_vectors]
    except Exception:
        # Keep the pipeline runnable in minimal environments; full runs should install sklearn.
        return non_vulnerable[:max_support_vectors]


def build_feature_bundle(
    sample: CodeSample,
    support_vector_indices: list[int],
    surrogate_model: Any,
    vocab: Any,
    top_k: int,
    output_path: Path | None = None,
) -> FeatureBundle:
    token_scores = attention_top_tokens(surrogate_model, vocab, sample.func, top_k=top_k)
    bundle = FeatureBundle(
        sample_idx=sample.idx,
        key_tokens=[token for token, _score in token_scores],
        support_vector_indices=support_vector_indices,
        token_scores=token_scores,
    )
    if output_path is not None:
        write_json(
            output_path,
            {
                "sample_idx": bundle.sample_idx,
                "key_tokens": bundle.key_tokens,
                "support_vector_indices": bundle.support_vector_indices,
                "token_scores": bundle.token_scores,
            },
        )
    return bundle
