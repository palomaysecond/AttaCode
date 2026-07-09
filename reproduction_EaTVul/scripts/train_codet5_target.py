from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import ExperimentConfig, load_config
from eatvul_repro.data import CodeSample, load_samples
from eatvul_repro.logging_utils import set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    samples: int


class CodeT5FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[CodeSample], tokenizer: Any, args: SimpleNamespace, converter: Any) -> None:
        self.features = [converter(sample.to_record(), tokenizer, args) for sample in samples]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[index]
        return (
            torch.tensor(feature.input_ids, dtype=torch.long),
            torch.tensor(int(feature.label), dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AttaCode-compatible CodeT5 target detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    if args.log_file is not None:
        _add_file_handler(Path(args.log_file))
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    set_seed(seed)

    if args.dataset not in cfg.raw["datasets"]:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.target_model_cfg("CodeT5")
    output_dir = _resolve_output_dir(cfg, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint-best-acc").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint-best-f1").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint-latest").mkdir(parents=True, exist_ok=True)

    train_samples = _limit_samples(load_samples(cfg.dataset_path(args.dataset, "train")), args.max_train_samples)
    valid_samples = _limit_samples(load_samples(cfg.dataset_path(args.dataset, "valid")), args.max_valid_samples)

    run_mod, model_mod = _load_attacode_codet5(cfg)
    train_args = SimpleNamespace(
        block_size=int(model_cfg.get("block_size", 512)),
        eval_batch_size=int(args.eval_batch_size),
        train_batch_size=int(args.train_batch_size),
        device=str(device),
        cache_dir="",
        model_name="codet5",
        model_type="codet5",
        output_dir=str(output_dir),
    )

    LOGGER.info("Loading CodeT5 base model %s on %s", model_cfg["model_name_or_path"], device)
    tokenizer = RobertaTokenizer.from_pretrained(model_cfg["tokenizer_name"])
    config = T5Config.from_pretrained(model_cfg["model_name_or_path"])
    encoder = T5ForConditionalGeneration.from_pretrained(model_cfg["model_name_or_path"])
    model = model_mod.CodeT5Model(encoder=encoder, config=config, tokenizer=tokenizer, args=train_args).to(device)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        LOGGER.info("Encoder is frozen; training classifier head only.")

    converter = getattr(run_mod, "codet5_convert_examples_to_features")
    train_dataset = CodeT5FeatureDataset(train_samples, tokenizer, train_args, converter)
    valid_dataset = CodeT5FeatureDataset(valid_samples, tokenizer, train_args, converter)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = _build_optimizer(model, args)
    update_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.gradient_accumulation_steps))
    warmup_steps = int(update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=update_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.fp16 and device.type == "cuda"))

    LOGGER.info(
        "Training %s/CodeT5 target: train=%d valid=%d epochs=%d batch=%d grad_accum=%d",
        args.dataset,
        len(train_dataset),
        len(valid_dataset),
        args.epochs,
        args.train_batch_size,
        args.gradient_accumulation_steps,
    )
    started_at = time.time()
    best_accuracy = -1.0
    best_f1 = -1.0
    best_accuracy_epoch = 0
    best_f1_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args, epoch)
        metrics = _evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics.__dict__})
        LOGGER.info(
            "epoch=%d train_loss=%.5f valid_loss=%.5f acc=%.4f f1=%.4f precision=%.4f recall=%.4f",
            epoch,
            train_loss,
            metrics.loss,
            metrics.accuracy,
            metrics.f1,
            metrics.precision,
            metrics.recall,
        )
        _save_state_dict(model, output_dir / "checkpoint-latest" / "model.bin")
        if metrics.accuracy >= best_accuracy:
            best_accuracy = metrics.accuracy
            best_accuracy_epoch = epoch
            _save_state_dict(model, output_dir / "checkpoint-best-acc" / "model.bin")
        if metrics.f1 >= best_f1:
            best_f1 = metrics.f1
            best_f1_epoch = epoch
            _save_state_dict(model, output_dir / "checkpoint-best-f1" / "model.bin")

    metadata = {
        "dataset": args.dataset,
        "target_model": "CodeT5",
        "base_model": model_cfg["model_name_or_path"],
        "tokenizer": model_cfg["tokenizer_name"],
        "seed": seed,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "block_size": train_args.block_size,
        "freeze_encoder": bool(args.freeze_encoder),
        "fp16": bool(args.fp16),
        "epochs": args.epochs,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "history": history,
        "best_accuracy": best_accuracy,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_f1": best_f1,
        "best_f1_epoch": best_f1_epoch,
        "best_checkpoint": str((output_dir / "checkpoint-best-f1" / "model.bin").resolve()),
        "best_accuracy_checkpoint": str((output_dir / "checkpoint-best-acc" / "model.bin").resolve()),
    }
    _write_json(output_dir / "training_metrics.json", metadata)
    LOGGER.info("Saved CodeT5 target checkpoint to %s", metadata["best_checkpoint"])
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "best_checkpoint": metadata["best_checkpoint"],
    }, ensure_ascii=False))


def _resolve_output_dir(cfg: ExperimentConfig, args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return cfg.resolve_path(args.output_dir)
    checkpoint = cfg.checkpoint_path(args.dataset, "CodeT5")
    return checkpoint.parent.parent


def _limit_samples(samples: list[CodeSample], limit: int | None) -> list[CodeSample]:
    if limit is None:
        return samples
    return samples[: max(1, int(limit))]


def _load_attacode_codet5(cfg: ExperimentConfig) -> tuple[Any, Any]:
    repo_root = cfg.resolve_path(cfg.raw["project"]["attacode_repo"])
    model_dir = repo_root / "CodeT5"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing AttaCode CodeT5 directory: {model_dir}")
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(model_dir))
    return importlib.import_module("CodeT5.run"), importlib.import_module("CodeT5.model")


def _build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    encoder_params = []
    classifier_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_params.append(parameter)
        else:
            encoder_params.append(parameter)
    groups: list[dict[str, Any]] = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.learning_rate, "weight_decay": args.weight_decay})
    if classifier_params:
        groups.append({
            "params": classifier_params,
            "lr": args.classifier_learning_rate,
            "weight_decay": args.weight_decay,
        })
    return AdamW(groups, lr=args.learning_rate, eps=1e-8)


def _train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.amp.autocast(device_type=device.type, enabled=bool(args.fp16 and device.type == "cuda")):
            loss, _prob = model(input_ids=input_ids, labels=labels)
            loss = loss / max(1, args.gradient_accumulation_steps)
        scaler.scale(loss).backward()
        losses.append(float(loss.detach().cpu().item()) * max(1, args.gradient_accumulation_steps))
        if step % max(1, args.gradient_accumulation_steps) == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if args.log_every_steps > 0 and step % args.log_every_steps == 0:
            recent = losses[-min(len(losses), args.log_every_steps) :]
            LOGGER.info(
                "epoch=%d step=%d/%d recent_train_loss=%.5f",
                epoch,
                step,
                len(train_loader),
                sum(recent) / max(1, len(recent)),
            )
    return sum(losses) / max(1, len(losses))


def _evaluate(model: torch.nn.Module, valid_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> EvalMetrics:
    model.eval()
    losses: list[float] = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in valid_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, prob = model(input_ids=input_ids, labels=labels)
            preds = torch.argmax(prob, dim=-1)
            losses.append(float(loss.detach().cpu().item()))
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            true_positive += int(((preds == 1) & (labels == 1)).sum().item())
            false_positive += int(((preds == 1) & (labels == 0)).sum().item())
            false_negative += int(((preds == 0) & (labels == 1)).sum().item())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return EvalMetrics(
        loss=sum(losses) / max(1, len(losses)),
        accuracy=correct / max(1, total),
        precision=precision,
        recall=recall,
        f1=f1,
        samples=total,
    )


def _save_state_dict(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _add_file_handler(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(handler)


if __name__ == "__main__":
    main()
