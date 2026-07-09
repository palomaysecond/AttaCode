"""FGA-style adversarial snippet selection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .data import AttackSample
from .insertion import InsertionSite, insert_snippet
from .target_model import Prediction, TargetModelAdapter


@dataclass(frozen=True)
class CandidateSnippet:
    snippet_id: str
    snippet: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AttackResult:
    idx: int
    status: str
    queries: int
    original_label: int
    original_prob_vulnerable: float
    final_prediction: int
    final_prob_vulnerable: float
    selected_snippets: list[str]
    adversarial_code: str | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "status": self.status,
            "queries": self.queries,
            "original_label": self.original_label,
            "original_prob_vulnerable": self.original_prob_vulnerable,
            "final_prediction": self.final_prediction,
            "final_prob_vulnerable": self.final_prob_vulnerable,
            "selected_snippets": self.selected_snippets,
            "adversarial_code": self.adversarial_code,
            "reason": self.reason,
        }


class FGASelector:
    def __init__(self, cfg: dict[str, Any], rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or random.Random(42)

    def attack(
        self,
        sample: AttackSample,
        candidates: list[CandidateSnippet],
        sites: list[InsertionSite],
        target: TargetModelAdapter,
    ) -> AttackResult:
        if not candidates:
            return self._failed(sample, 0, sample.original_pred, sample.original_prob_vulnerable, "empty attack pool")
        if not sites:
            return self._failed(sample, 0, sample.original_pred, sample.original_prob_vulnerable, "no insertion site")

        start_queries = target.query_count
        population = self._initial_population(candidates, sites)
        best: tuple[float, list[CandidateSnippet], InsertionSite, Prediction, str] | None = None

        for _generation in range(int(self.cfg.get("max_generations", 5))):
            scored = self._score_population(sample, population, target)
            scored.sort(key=lambda item: item[0], reverse=True)
            if best is None or scored[0][0] > best[0]:
                best = scored[0]
            if bool(self.cfg.get("early_stop_on_success", True)) and scored[0][3].label != sample.label:
                return self._success(sample, target.query_count - start_queries, scored[0])
            population = self._next_population(scored, candidates, sites)

        if best is None:
            return self._failed(sample, target.query_count - start_queries, sample.original_pred, sample.original_prob_vulnerable, "no evaluated candidate")
        if best[3].label != sample.label:
            return self._success(sample, target.query_count - start_queries, best)
        return AttackResult(
            idx=sample.idx,
            status="failed",
            queries=target.query_count - start_queries,
            original_label=sample.label,
            original_prob_vulnerable=sample.original_prob_vulnerable,
            final_prediction=best[3].label,
            final_prob_vulnerable=best[3].prob_vulnerable,
            selected_snippets=[snippet.snippet_id for snippet in best[1]],
            adversarial_code=best[4],
            reason="search exhausted",
        )

    def _initial_population(
        self,
        candidates: list[CandidateSnippet],
        sites: list[InsertionSite],
    ) -> list[tuple[list[CandidateSnippet], InsertionSite]]:
        population_size = int(self.cfg.get("population_size", 20))
        max_inserted = int(self.cfg.get("max_inserted_snippets", 4))
        population: list[tuple[list[CandidateSnippet], InsertionSite]] = []
        for _idx in range(population_size):
            size = self.rng.randint(1, min(max_inserted, len(candidates)))
            snippets = self.rng.sample(candidates, size)
            site = self.rng.choice(sites)
            population.append((snippets, site))
        return population

    def _score(
        self,
        sample: AttackSample,
        individual: tuple[list[CandidateSnippet], InsertionSite],
        target: TargetModelAdapter,
    ) -> tuple[float, list[CandidateSnippet], InsertionSite, Prediction, str]:
        snippets, site = individual
        combined = "\n".join(snippet.snippet for snippet in snippets)
        modified = insert_snippet(sample.func, combined, site)
        prediction = target.predict_one(modified)
        length_penalty = float(self.cfg.get("length_penalty_lambda", 0.1)) * len(combined.splitlines())
        success_bonus = 1.0 if prediction.label != sample.label else 0.0
        score = success_bonus + (1.0 - prediction.prob_vulnerable) - length_penalty
        return score, snippets, site, prediction, modified

    def _score_population(
        self,
        sample: AttackSample,
        population: list[tuple[list[CandidateSnippet], InsertionSite]],
        target: TargetModelAdapter,
    ) -> list[tuple[float, list[CandidateSnippet], InsertionSite, Prediction, str]]:
        modified_codes: list[str] = []
        metadata: list[tuple[list[CandidateSnippet], InsertionSite, str]] = []
        for snippets, site in population:
            combined = "\n".join(snippet.snippet for snippet in snippets)
            modified = insert_snippet(sample.func, combined, site)
            modified_codes.append(modified)
            metadata.append((snippets, site, combined))

        predictions = target.predict_many(modified_codes)
        scored: list[tuple[float, list[CandidateSnippet], InsertionSite, Prediction, str]] = []
        for (snippets, site, combined), prediction, modified in zip(metadata, predictions, modified_codes):
            length_penalty = float(self.cfg.get("length_penalty_lambda", 0.1)) * len(combined.splitlines())
            success_bonus = 1.0 if prediction.label != sample.label else 0.0
            score = success_bonus + (1.0 - prediction.prob_vulnerable) - length_penalty
            scored.append((score, snippets, site, prediction, modified))
        return scored

    def _next_population(
        self,
        scored: list[tuple[float, list[CandidateSnippet], InsertionSite, Prediction, str]],
        candidates: list[CandidateSnippet],
        sites: list[InsertionSite],
    ) -> list[tuple[list[CandidateSnippet], InsertionSite]]:
        survivors = scored[: max(2, len(scored) // 2)]
        next_population: list[tuple[list[CandidateSnippet], InsertionSite]] = [
            (snippets, site) for _score, snippets, site, _pred, _code in survivors
        ]
        while len(next_population) < int(self.cfg.get("population_size", 20)):
            parent_a = self.rng.choice(survivors)
            parent_b = self.rng.choice(survivors)
            child_snippets = self._crossover(parent_a[1], parent_b[1], candidates)
            child_site = parent_a[2] if self.rng.random() > 0.5 else parent_b[2]
            if self.rng.random() < float(self.cfg.get("mutation_rate", 0.2)):
                child_snippets = self._mutate(child_snippets, candidates)
            if self.rng.random() < float(self.cfg.get("mutation_rate", 0.2)):
                child_site = self.rng.choice(sites)
            next_population.append((child_snippets, child_site))
        return next_population

    def _crossover(
        self,
        left: list[CandidateSnippet],
        right: list[CandidateSnippet],
        candidates: list[CandidateSnippet],
    ) -> list[CandidateSnippet]:
        if self.rng.random() > float(self.cfg.get("crossover_rate", 0.8)):
            return list(left)
        merged = list({snippet.snippet_id: snippet for snippet in left + right}.values())
        max_inserted = int(self.cfg.get("max_inserted_snippets", 4))
        if not merged:
            return [self.rng.choice(candidates)]
        return merged[:max_inserted]

    def _mutate(
        self,
        snippets: list[CandidateSnippet],
        candidates: list[CandidateSnippet],
    ) -> list[CandidateSnippet]:
        if not snippets:
            return [self.rng.choice(candidates)]
        mutated = list(snippets)
        mutated[self.rng.randrange(len(mutated))] = self.rng.choice(candidates)
        return list({snippet.snippet_id: snippet for snippet in mutated}.values())

    @staticmethod
    def _success(
        sample: AttackSample,
        queries: int,
        scored: tuple[float, list[CandidateSnippet], InsertionSite, Prediction, str],
    ) -> AttackResult:
        _score, snippets, _site, prediction, modified = scored
        return AttackResult(
            idx=sample.idx,
            status="success",
            queries=queries,
            original_label=sample.label,
            original_prob_vulnerable=sample.original_prob_vulnerable,
            final_prediction=prediction.label,
            final_prob_vulnerable=prediction.prob_vulnerable,
            selected_snippets=[snippet.snippet_id for snippet in snippets],
            adversarial_code=modified,
            reason="misclassified",
        )

    @staticmethod
    def _failed(
        sample: AttackSample,
        queries: int,
        final_prediction: int,
        final_prob: float,
        reason: str,
    ) -> AttackResult:
        return AttackResult(
            idx=sample.idx,
            status="failed",
            queries=queries,
            original_label=sample.label,
            original_prob_vulnerable=sample.original_prob_vulnerable,
            final_prediction=final_prediction,
            final_prob_vulnerable=final_prob,
            selected_snippets=[],
            adversarial_code=None,
            reason=reason,
        )
