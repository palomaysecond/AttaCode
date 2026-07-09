"""BiLSTM-attention surrogate model used by EaTVul."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CodeSample

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|&&|\|\||[{}()\[\];,.*&+\-/<>=%]")


def tokenize_code(code: str) -> list[str]:
    return TOKEN_PATTERN.findall(code)


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    max_len: int

    @staticmethod
    def build(samples: list[CodeSample], max_len: int, min_freq: int = 1) -> "Vocab":
        counts: dict[str, int] = {}
        for sample in samples:
            for token in tokenize_code(sample.func):
                counts[token] = counts.get(token, 0) + 1
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, count in sorted(counts.items()):
            if count >= min_freq:
                token_to_id[token] = len(token_to_id)
        return Vocab(token_to_id=token_to_id, max_len=max_len)

    def encode(self, code: str) -> tuple[list[int], list[str]]:
        tokens = tokenize_code(code)[: self.max_len]
        ids = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens]
        ids += [self.token_to_id["<pad>"]] * (self.max_len - len(ids))
        return ids, tokens

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump({"token_to_id": self.token_to_id, "max_len": self.max_len}, handle, indent=2)

    @staticmethod
    def load(path: str | Path) -> "Vocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return Vocab(token_to_id={str(k): int(v) for k, v in data["token_to_id"].items()}, max_len=int(data["max_len"]))


class SurrogateDataset(Dataset):
    def __init__(
        self,
        samples: list[CodeSample],
        vocab: Vocab,
        soft_targets: dict[int, float] | None = None,
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.soft_targets = soft_targets or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        ids, _tokens = self.vocab.encode(sample.func)
        label = float(sample.label)
        soft = float(self.soft_targets.get(sample.idx, label))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(soft, dtype=torch.float32),
            torch.tensor(sample.idx, dtype=torch.long),
        )


class BiLSTMAttentionSurrogate(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = input_ids.ne(0)
        emb = self.dropout(self.embedding(input_ids))
        hidden, _state = self.lstm(emb)
        attention_logits = self.attention(hidden).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        pooled = torch.sum(hidden * attention_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs, attention_weights

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        _logits, _probs, attention = self.forward(input_ids)
        emb = self.embedding(input_ids)
        return torch.sum(emb * attention.unsqueeze(-1), dim=1)


@dataclass(frozen=True)
class SurrogateArtifacts:
    model_path: Path
    vocab_path: Path
    metrics_path: Path


class SurrogateTrainer:
    def __init__(self, cfg: dict[str, Any], device: str | None = None) -> None:
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        train_samples: list[CodeSample],
        valid_samples: list[CodeSample],
        soft_targets: dict[int, float],
        output_dir: Path,
    ) -> SurrogateArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        vocab = Vocab.build(
            train_samples,
            max_len=int(self.cfg.get("max_len", 200)),
            min_freq=int(self.cfg.get("min_token_freq", 1)),
        )
        model = BiLSTMAttentionSurrogate(
            vocab_size=len(vocab.token_to_id),
            embedding_dim=int(self.cfg.get("embedding_dim", 100)),
            hidden_dim=int(self.cfg.get("hidden_dim", 128)),
            dropout=float(self.cfg.get("dropout", 0.2)),
        ).to(self.device)

        train_loader = DataLoader(
            SurrogateDataset(train_samples, vocab, soft_targets),
            batch_size=int(self.cfg.get("batch_size", 32)),
            shuffle=True,
        )
        valid_loader = DataLoader(
            SurrogateDataset(valid_samples, vocab, soft_targets),
            batch_size=int(self.cfg.get("batch_size", 32)),
            shuffle=False,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.cfg.get("learning_rate", 0.001)))
        best_loss = float("inf")
        patience = int(self.cfg.get("patience", 40))
        stale_epochs = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(int(self.cfg.get("epochs", 100))):
            model.train()
            for input_ids, labels, soft, _idx in train_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                soft = soft.to(self.device)
                logits, _probs, _attention = model(input_ids)
                loss = self._distillation_loss(logits, labels, soft)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_loss = self._evaluate_loss(model, valid_loader)
            if valid_loss < best_loss:
                best_loss = valid_loss
                stale_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model_path = output_dir / "surrogate.pt"
        vocab_path = output_dir / "vocab.json"
        metrics_path = output_dir / "metrics.json"
        torch.save(model.state_dict(), model_path)
        vocab.save(vocab_path)
        metrics_path.write_text(
            json.dumps({"best_valid_loss": best_loss}, indent=2),
            encoding="utf-8",
        )
        return SurrogateArtifacts(model_path=model_path, vocab_path=vocab_path, metrics_path=metrics_path)

    def _distillation_loss(self, logits: torch.Tensor, labels: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        hard_loss = F.binary_cross_entropy_with_logits(logits, labels)
        kd_loss = F.binary_cross_entropy_with_logits(logits, soft)
        alpha = float(self.cfg.get("kd_alpha", 0.9))
        beta = float(self.cfg.get("kd_beta", 0.1))
        return alpha * kd_loss + beta * hard_loss

    def _evaluate_loss(self, model: BiLSTMAttentionSurrogate, loader: DataLoader) -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for input_ids, labels, soft, _idx in loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                soft = soft.to(self.device)
                logits, _probs, _attention = model(input_ids)
                losses.append(float(self._distillation_loss(logits, labels, soft).cpu().item()))
        return float(np.mean(losses)) if losses else float("inf")


def load_surrogate(
    artifacts_dir: Path,
    cfg: dict[str, Any],
    device: str | None = None,
) -> tuple[BiLSTMAttentionSurrogate, Vocab]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(artifacts_dir / "vocab.json")
    model = BiLSTMAttentionSurrogate(
        vocab_size=len(vocab.token_to_id),
        embedding_dim=int(cfg.get("embedding_dim", 100)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(torch.load(artifacts_dir / "surrogate.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, vocab


def attention_top_tokens(
    model: BiLSTMAttentionSurrogate,
    vocab: Vocab,
    code: str,
    top_k: int,
    device: str | None = None,
) -> list[tuple[str, float]]:
    device = device or next(model.parameters()).device
    ids, tokens = vocab.encode(code)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        _logits, _probs, attention = model(input_ids)
    scores = attention[0][: len(tokens)].detach().cpu().numpy()
    ranked = sorted(zip(tokens, scores), key=lambda item: float(item[1]), reverse=True)
    return [(token, float(score)) for token, score in ranked[:top_k]]

