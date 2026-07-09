"""Target detector adapters."""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import CodeSample

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: int
    prob_vulnerable: float
    raw: Any = None


class TargetModelAdapter:
    def __init__(self) -> None:
        self.query_count = 0

    def predict_one(self, code: str) -> Prediction:
        return self.predict_many([code])[0]

    def predict_many(self, codes: list[str]) -> list[Prediction]:
        raise NotImplementedError

    def reset_queries(self) -> None:
        self.query_count = 0


class MockTargetModelAdapter(TargetModelAdapter):
    """Simple deterministic detector used for lightweight debugging."""

    def predict_many(self, codes: list[str]) -> list[Prediction]:
        self.query_count += len(codes)
        predictions: list[Prediction] = []
        for code in codes:
            risk = 0.2
            if any(token in code for token in ["strcpy", "unchecked", "arr[idx]", "memcpy"]):
                risk += 0.65
            if "eatvul_noise" in code:
                risk -= 0.45
            risk = float(np.clip(risk, 0.01, 0.99))
            predictions.append(Prediction(label=int(risk > 0.5), prob_vulnerable=risk, raw={"mock": True}))
        return predictions


class AttaCodeTargetModelAdapter(TargetModelAdapter):
    """Loads CodeBERT, UniXcoder, or CodeT5 from the AttaCode repository."""

    def __init__(self, cfg: ExperimentConfig, dataset: str, target_model: str, device: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.target_model = target_model
        self.device = device or self._default_device()
        self.model_cfg = cfg.target_model_cfg(target_model)
        self.repo_root = cfg.resolve_path(cfg.raw["project"]["attacode_repo"])
        self.checkpoint_path = cfg.checkpoint_path(dataset, target_model)
        self._load_model()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def predict_many(self, codes: list[str]) -> list[Prediction]:
        if not codes:
            return []
        predictions = self._predict_many_batched(codes)
        self.query_count += len(codes)
        return predictions

    def _load_model(self) -> None:
        hf_model_id = self._hf_model_id()
        if hf_model_id:
            self._load_hf_sequence_classifier(hf_model_id)
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing target checkpoint: {self.checkpoint_path}")
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Missing AttaCode repo: {self.repo_root}")

        import torch
        from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

        model_name = self.target_model.lower()
        model_dir = self.repo_root / self.target_model
        sys.path.insert(0, str(self.repo_root))
        sys.path.insert(0, str(model_dir))

        args = SimpleNamespace(
            block_size=int(self.model_cfg.get("block_size", 512)),
            eval_batch_size=int(self.model_cfg.get("eval_batch_size", 4)),
            device=self.device,
            cache_dir="",
            model_name=model_name,
            model_type="roberta",
            output_dir=str(self.cfg.output_dir / "target_cache" / self.dataset / self.target_model),
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        if self.target_model == "CodeT5":
            from transformers import T5Config, T5ForConditionalGeneration

            run_mod = importlib.import_module("CodeT5.run")
            model_mod = importlib.import_module("CodeT5.model")
            tokenizer = RobertaTokenizer.from_pretrained(self.model_cfg["tokenizer_name"])
            config = T5Config.from_pretrained(self.model_cfg["model_name_or_path"])
            encoder = T5ForConditionalGeneration.from_pretrained(self.model_cfg["model_name_or_path"])
            model = model_mod.CodeT5Model(encoder, config, tokenizer, args)
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device), strict=False)
            self._run_mod = run_mod
            self._detect_fn = run_mod.vulnerability_detect
        else:
            if self.target_model == "CodeBERT":
                run_mod = importlib.import_module("CodeBERT.run")
                model_mod = importlib.import_module("CodeBERT.model")
                wrapper_cls = model_mod.CodeBERTModel
                config = RobertaConfig.from_pretrained(self.model_cfg["model_name_or_path"])
                config.num_labels = 1
            elif self.target_model == "UniXcoder":
                run_mod = importlib.import_module("UniXcoder.run")
                model_mod = importlib.import_module("UniXcoder.model")
                wrapper_cls = model_mod.UniXcoderModel
                config = RobertaConfig.from_pretrained(self.model_cfg["model_name_or_path"])
                config.num_labels = 2
            else:
                raise ValueError(f"Unsupported target model: {self.target_model}")

            tokenizer = RobertaTokenizer.from_pretrained(self.model_cfg["tokenizer_name"])
            encoder = RobertaForSequenceClassification.from_pretrained(
                self.model_cfg["model_name_or_path"],
                config=config,
            )
            if self.target_model == "CodeBERT":
                model = wrapper_cls(encoder, config, tokenizer, args)
            else:
                model = wrapper_cls(encoder, config, args)
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device), strict=False)
            self._run_mod = run_mod
            self._detect_fn = getattr(run_mod, "vulnerability_detect", None)

        self.args = args
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        LOGGER.info("Loaded %s target for %s", self.target_model, self.dataset)

    def _hf_model_id(self) -> str | None:
        hf_ids = self.model_cfg.get("hf_model_ids") or {}
        value = hf_ids.get(self.dataset)
        if value in {None, "", "null"}:
            return None
        return str(value)

    def _load_hf_sequence_classifier(self, hf_model_id: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        args = SimpleNamespace(
            block_size=int(self.model_cfg.get("block_size", 512)),
            eval_batch_size=int(self.model_cfg.get("eval_batch_size", 4)),
            device=self.device,
            cache_dir="",
            model_name=self.target_model.lower(),
            model_type="hf_sequence_classifier",
            output_dir=str(self.cfg.output_dir / "target_cache" / self.dataset / self.target_model),
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        self.args = args
        tokenizer_name = str(self.model_cfg.get("tokenizer_name", hf_model_id))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_id).to(self.device)
        self.model.eval()
        self._run_mod = None
        self._detect_fn = None
        self._hf_sequence_classifier = True
        LOGGER.info("Loaded HF sequence classifier %s for %s/%s", hf_model_id, self.dataset, self.target_model)

    def _predict_with_attacode(self, code: str) -> Prediction:
        import torch

        if getattr(self, "_hf_sequence_classifier", False):
            encoded = self.tokenizer(
                code,
                truncation=True,
                max_length=int(self.model_cfg.get("block_size", 512)),
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    prob_vuln = torch.sigmoid(logits)[0, 0].item()
                else:
                    prob_vuln = torch.softmax(logits, dim=-1)[0, 1].item()
            return Prediction(label=int(prob_vuln > 0.5), prob_vulnerable=float(prob_vuln), raw=logits)

        if self._detect_fn is not None:
            with torch.no_grad():
                output = self._detect_fn(code, self.model, self.tokenizer, self.args)
            return self._normalize_detect_output(output)

        sample = CodeSample(idx=0, label=0, func=code, meta={})
        record = sample.to_record()
        feature_fn = getattr(self._run_mod, "unixcoder_convert_examples_to_features")
        feature = feature_fn(record, self.tokenizer, self.args)
        input_ids = torch.tensor([feature.input_ids]).to(self.device)
        with torch.no_grad():
            prob, logits = self.model(input_ids=input_ids)
        prob_vuln = float(prob[0][1].detach().cpu().item())
        return Prediction(label=int(prob_vuln > 0.5), prob_vulnerable=prob_vuln, raw=logits)

    def _predict_many_batched(self, codes: list[str]) -> list[Prediction]:
        import torch

        if getattr(self, "_hf_sequence_classifier", False):
            encoded = self.tokenizer(
                codes,
                truncation=True,
                max_length=int(self.model_cfg.get("block_size", 512)),
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            return self._predictions_from_logits(logits)

        feature_fn = self._feature_converter()
        if feature_fn is None:
            return [self._predict_with_attacode(code) for code in codes]

        records = [{"func": code, "idx": idx, "target": 0} for idx, code in enumerate(codes)]
        features = [feature_fn(record, self.tokenizer, self.args) for record in records]
        batch_size = int(self.model_cfg.get("eval_batch_size", 4))
        predictions: list[Prediction] = []
        self.model.eval()
        for start in range(0, len(features), batch_size):
            batch_features = features[start : start + batch_size]
            input_ids = torch.tensor([feature.input_ids for feature in batch_features], dtype=torch.long).to(self.device)
            with torch.no_grad():
                output = self.model(input_ids=input_ids)
            if isinstance(output, tuple):
                if len(output) >= 2:
                    _prob, logits = output[0], output[1]
                else:
                    logits = output[0]
            else:
                logits = output
            predictions.extend(self._predictions_from_logits(logits))
        return predictions

    def _feature_converter(self) -> Any | None:
        if self._run_mod is None:
            return None
        if self.target_model == "CodeBERT":
            return getattr(self._run_mod, "codebert_convert_examples_to_features")
        if self.target_model == "UniXcoder":
            if bool(self.model_cfg.get("use_encoder_only_prefix", False)):
                return self._unixcoder_encoder_only_features
            return getattr(self._run_mod, "unixcoder_convert_examples_to_features")
        if self.target_model == "CodeT5":
            return getattr(self._run_mod, "codet5_convert_examples_to_features")
        return None

    def _unixcoder_encoder_only_features(self, record: dict[str, Any], tokenizer: Any, args: Any) -> Any:
        code = " ".join(str(record["func"]).split())
        code_tokens = tokenizer.tokenize(code)[: int(args.block_size) - 4]
        tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [tokenizer.pad_token_id] * (int(args.block_size) - len(input_ids))
        return SimpleNamespace(input_tokens=tokens, input_ids=input_ids, index=record["idx"], label=int(record["target"]))

    @staticmethod
    def _predictions_from_logits(logits: Any) -> list[Prediction]:
        import torch

        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        logits_cpu = logits.detach().cpu()
        if logits_cpu.ndim == 1:
            logits_cpu = logits_cpu.unsqueeze(-1)
        if logits_cpu.shape[-1] == 1:
            probs = torch.sigmoid(logits_cpu).view(-1)
        else:
            probs = torch.softmax(logits_cpu, dim=-1)[:, 1]
        return [
            Prediction(label=int(float(prob.item()) > 0.5), prob_vulnerable=float(prob.item()), raw=logit)
            for prob, logit in zip(probs, logits_cpu)
        ]

    @staticmethod
    def _normalize_detect_output(output: Any) -> Prediction:
        if isinstance(output, tuple) and len(output) == 3:
            is_vuln, _logit, prob = output
            prob_tensor = prob.detach().cpu().view(-1) if hasattr(prob, "detach") else None
            logit_tensor = _logit.detach().cpu().view(-1) if hasattr(_logit, "detach") else None
            if prob_tensor is not None and prob_tensor.numel() >= 2:
                prob_value = float(prob_tensor[1].item())
            elif logit_tensor is not None and logit_tensor.numel() == 1:
                import torch

                prob_value = float(torch.sigmoid(logit_tensor)[0].item())
            elif prob_tensor is not None and prob_tensor.numel() == 1:
                prob_value = float(prob_tensor[0].item())
            else:
                prob_value = float(prob)
            label = int(bool(is_vuln.detach().cpu().view(-1)[0].item()) if hasattr(is_vuln, "detach") else bool(is_vuln))
            return Prediction(label=label, prob_vulnerable=prob_value, raw=output)
        if isinstance(output, tuple) and len(output) == 2:
            is_vuln, confidence = output
            return Prediction(label=int(bool(is_vuln)), prob_vulnerable=float(confidence), raw=output)
        raise ValueError(f"Unsupported AttaCode detection output: {type(output)}")


def build_target_adapter(
    cfg: ExperimentConfig,
    dataset: str,
    target_model: str,
    mock: bool = False,
    device: str | None = None,
) -> TargetModelAdapter:
    if mock:
        return MockTargetModelAdapter()
    return AttaCodeTargetModelAdapter(cfg, dataset=dataset, target_model=target_model, device=device)
