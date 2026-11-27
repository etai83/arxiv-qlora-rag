"""Configure, run, log, and package MLX QLoRA training for Llama-3.2-3B-Instruct."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src import write_jsonl

DEFAULT_MLX_PYTHON = Path.home() / ".local/share/uv/tools/mlx-lm/bin/python"
DEFAULT_LOCAL_VENV_PYTHON = Path(".venv-qlora/bin/python")
DEFAULT_BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

_TRAIN_LINE_RE = re.compile(
    r"^Iter\s+(?P<iter>\d+):\s*Train loss\s+(?P<train_loss>[-+]?\d*\.?\d+)"
    r"(?:,\s*Learning Rate\s+(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"(?:,\s*It/sec\s+(?P<it_sec>[-+]?\d*\.?\d+))?"
    r"(?:,\s*Tokens/sec\s+(?P<tokens_sec>[-+]?\d*\.?\d+))?"
    r"(?:,\s*Trained [Tt]okens\s+(?P<trained_tokens>\d+))?"
    r"(?:,\s*Peak mem\s+(?P<peak_mem_gb>[-+]?\d*\.?\d+)\s*GB)?"
)
_VAL_LINE_RE = re.compile(
    r"^Iter\s+(?P<iter>\d+):\s*Val loss\s+(?P<val_loss>[-+]?\d*\.?\d+)"
    r"(?:,\s*Val took\s+(?P<val_took_s>[-+]?\d*\.?\d+)s)?"
)
_TABLE_TRAIN_LINE_RE = re.compile(
    r"^(?P<iter>\d+)\s+(?P<train_loss>[-+]?\d*\.?\d+)(?:\s+[▲▼─•\-])?"
    r"\s+(?P<tokens_sec>[-+]?\d*\.?\d+)\s+(?P<trained_tokens_raw>[\d.]+[kKmM]?)$"
)
_TABLE_VAL_LINE_RE = re.compile(
    r"^(?P<iter>\d+)\s+val\s+(?P<val_loss>[-+]?\d*\.?\d+)"
    r"(?:\s+(?P<val_took_s>[-+]?\d*\.?\d+)s)?$"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TRAINABLE_RE = re.compile(
    r"Trainable parameters:\s*(?P<pct>[-+]?\d*\.?\d+)%\s*\((?P<trainable_m>[-+]?\d*\.?\d+)M/(?P<total_m>[-+]?\d*\.?\d+)M\)"
)


def write_lora_config(
    out_path: str | Path = "data/qlora/lora_config.yaml",
    *,
    model: str = DEFAULT_BASE_MODEL,
    data_dir: str | Path = "data/qlora",
    adapter_path: str | Path = "models/llama32_3b_qlora_adapters",
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_layers: int = 28,
    rank: int = 16,
    scale: float = 2.0,
    dropout: float = 0.05,
    max_seq_length: int = 2048,
    seed: int = 42,
) -> Path:
    """Write a clean YAML configuration file for `mlx_lm lora --config ...`."""
    warmup_steps = min(40, iters)
    config: dict[str, Any] = {
        "model": str(model),
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "seed": int(seed),
        "num_layers": int(num_layers),
        "batch_size": int(batch_size),
        "iters": int(iters),
        "val_batches": 25,
        "learning_rate": float(learning_rate),
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 200,
        "adapter_path": str(adapter_path),
        "max_seq_length": int(max_seq_length),
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {
            "rank": int(rank),
            "scale": float(scale),
            "dropout": float(dropout),
        },
        "lr_schedule": {
            "name": "cosine_decay",
            "warmup": int(warmup_steps),
            "arguments": [float(learning_rate), int(iters), 1e-5],
        },
    }
    config_file = Path(out_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return config_file


def _parse_token_count(raw: str) -> int:
    """Parse compact token counts like `2.0k` or `1.5M` into integer counts."""
    text = raw.strip().lower()
    if text.endswith("k"):
        return int(round(float(text[:-1]) * 1_000))
    if text.endswith("m"):
        return int(round(float(text[:-1]) * 1_000_000))
    return int(float(text))


def parse_mlx_training_log(log_text: str) -> list[dict[str, Any]]:
    """Parse MLX LoRA training and validation stdout lines into structured dicts."""
    events: list[dict[str, Any]] = []
    for raw_line in log_text.splitlines():
        line = _ANSI_RE.sub("", raw_line).strip()
        if not line:
            continue

        train_match = _TRAIN_LINE_RE.match(line)
        if train_match:
            groups = train_match.groupdict()
            iter_num = int(groups["iter"])
            entry: dict[str, Any] = {
                "event": "train",
                "iter": iter_num,
                "step": iter_num,
                "train_loss": float(groups["train_loss"]),
            }
            if groups["lr"] is not None:
                entry["learning_rate"] = float(groups["lr"])
            if groups["it_sec"] is not None:
                entry["it_sec"] = float(groups["it_sec"])
            if groups["tokens_sec"] is not None:
                entry["tokens_sec"] = float(groups["tokens_sec"])
            if groups["trained_tokens"] is not None:
                entry["trained_tokens"] = int(groups["trained_tokens"])
            if groups["peak_mem_gb"] is not None:
                entry["peak_mem_gb"] = float(groups["peak_mem_gb"])
            events.append(entry)
            continue

        table_train_match = _TABLE_TRAIN_LINE_RE.match(line)
        if table_train_match:
            groups = table_train_match.groupdict()
            iter_num = int(groups["iter"])
            events.append(
                {
                    "event": "train",
                    "iter": iter_num,
                    "step": iter_num,
                    "train_loss": float(groups["train_loss"]),
                    "tokens_sec": float(groups["tokens_sec"]),
                    "trained_tokens": _parse_token_count(groups["trained_tokens_raw"]),
                }
            )
            continue

        val_match = _VAL_LINE_RE.match(line) or _TABLE_VAL_LINE_RE.match(line)
        if val_match:
            groups = val_match.groupdict()
            iter_num = int(groups["iter"])
            val_entry: dict[str, Any] = {
                "event": "val",
                "iter": iter_num,
                "step": iter_num,
                "val_loss": float(groups["val_loss"]),
            }
            if groups["val_took_s"] is not None:
                val_entry["val_took_s"] = float(groups["val_took_s"])
            events.append(val_entry)

    return events


def _extract_trainable_params(log_text: str) -> dict[str, float] | None:
    """Extract trainable parameter percentage and counts from MLX startup logs."""
    match = _TRAINABLE_RE.search(log_text)
    if not match:
        return None
    return {
        "percent": float(match.group("pct")),
        "trainable_millions": float(match.group("trainable_m")),
        "total_millions": float(match.group("total_m")),
    }


def _directory_size_mb(path: str | Path) -> float:
    """Return total file size inside a directory (or single file) in MiB."""
    root = Path(path)
    if not root.exists():
        return 0.0
    if root.is_file():
        return round(root.stat().st_size / (1024 * 1024), 3)
    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return round(total_bytes / (1024 * 1024), 3)


def write_train_manifest(
    *,
    out_path: str | Path = "data/qlora/train_manifest.json",
    config: dict[str, Any],
    events: list[dict[str, Any]],
    adapter_path: str | Path = "models/llama32_3b_qlora_adapters",
    wall_time_s: float = 0.0,
    trainable_params: dict[str, float] | None = None,
) -> Path:
    """Write `train_manifest.json` summarizing training config, losses, memory, and artifacts."""
    train_events = [e for e in events if e.get("event") == "train"]
    val_events = [e for e in events if e.get("event") == "val"]

    train_losses = [float(e["train_loss"]) for e in train_events]
    val_losses = [float(e["val_loss"]) for e in val_events]
    peak_mems = [float(e["peak_mem_gb"]) for e in train_events if "peak_mem_gb" in e]

    best_val_event = min(val_events, key=lambda e: float(e["val_loss"])) if val_events else None
    best_train_event = min(train_events, key=lambda e: float(e["train_loss"])) if train_events else None
    best_iter = (
        int(best_val_event["iter"])
        if best_val_event is not None
        else (int(best_train_event["iter"]) if best_train_event is not None else None)
    )

    manifest = {
        "config": config,
        "initial_train_loss": train_losses[0] if train_losses else None,
        "best_train_loss": min(train_losses) if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "initial_val_loss": val_losses[0] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_iter": best_iter,
        "peak_mem_gb": max(peak_mems) if peak_mems else None,
        "wall_time_s": round(float(wall_time_s), 2),
        "adapter_size_mb": _directory_size_mb(adapter_path),
        "trainable_params": trainable_params,
    }
    manifest_file = Path(out_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_file


def build_modelfile(
    fused_model_dir: str | Path = "models/llama32_3b_qlora_fused",
    out_path: str | Path = "Modelfile.qlora",
) -> Path:
    """Write an Ollama Modelfile using the exact Llama-3.2-3B-Instruct chat template."""
    content = (
        f"FROM {fused_model_dir}\n\n"
        'TEMPLATE """{{- if .System }}<|start_header_id|>system<|end_header_id|>\n\n'
        "{{ .System }}<|eot_id|>\n"
        "{{- end }}\n"
        "{{- range $i, $_ := .Messages }}\n"
        "{{- $last := eq (len (slice $.Messages $i)) 1 }}\n"
        '{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n'
        "{{ .Content }}<|eot_id|>\n"
        "{{- if $last }}<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{{ end }}\n"
        '{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>\n\n'
        "{{ .Content }}{{ if not $last }}<|eot_id|>{{ end }}\n"
        "{{- end }}\n"
        '{{- end }}"""\n\n'
        'PARAMETER stop "<|start_header_id|>"\n'
        'PARAMETER stop "<|end_header_id|>"\n'
        'PARAMETER stop "<|eot_id|>"\n'
    )
    modelfile_path = Path(out_path)
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def _resolve_python_bin(python_bin: str | Path | None) -> str:
    """Resolve the Python executable that has `mlx_lm` installed."""
    if python_bin is not None:
        candidate = Path(python_bin)
        if candidate.exists():
            return str(candidate)
        resolved = shutil.which(str(python_bin))
        if resolved:
            return resolved
        raise FileNotFoundError(f"Configured python_bin not found: {python_bin}")
    for candidate in (DEFAULT_LOCAL_VENV_PYTHON, DEFAULT_MLX_PYTHON):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_training(
    *,
    config_path: str | Path = "data/qlora/lora_config.yaml",
    model: str = DEFAULT_BASE_MODEL,
    data_dir: str | Path = "data/qlora",
    adapter_path: str | Path = "models/llama32_3b_qlora_adapters",
    fused_model_dir: str | Path = "models/llama32_3b_qlora_fused",
    modelfile_path: str | Path = "Modelfile.qlora",
    python_bin: str | Path | None = None,
    iters: int = 600,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_layers: int = 28,
    rank: int = 16,
    scale: float = 2.0,
    dropout: float = 0.05,
    max_seq_length: int = 2048,
    seed: int = 42,
    fuse: bool = True,
    overwrite_config: bool = False,
) -> dict[str, Any]:
    """Load or write LoRA config, execute MLX QLoRA training, log metrics, and optionally fuse."""
    cfg_path = Path(config_path)
    if overwrite_config or not cfg_path.exists():
        cfg_path = write_lora_config(
            out_path=config_path,
            model=model,
            data_dir=data_dir,
            adapter_path=adapter_path,
            iters=iters,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_layers=num_layers,
            rank=rank,
            scale=scale,
            dropout=dropout,
            max_seq_length=max_seq_length,
            seed=seed,
        )
    config_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model = str(config_data.get("model", model))
    data_dir = config_data.get("data", data_dir)
    adapter_path = config_data.get("adapter_path", adapter_path)
    py_exe = _resolve_python_bin(python_bin)

    cmd = [py_exe, "-m", "mlx_lm", "lora", "--config", str(cfg_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_time_s = time.perf_counter() - t0
    log_text = proc.stdout or ""

    out_root = Path(data_dir)
    raw_log_path = out_root / "training_stdout.log"
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.write_text(log_text, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            f"mlx_lm lora failed with exit code {proc.returncode}. See {raw_log_path}:\n{log_text[-2000:]}"
        )

    events = parse_mlx_training_log(log_text)
    log_jsonl_path = write_jsonl(out_root / "training_log.jsonl", events)
    manifest_path = write_train_manifest(
        out_path=out_root / "train_manifest.json",
        config=config_data,
        events=events,
        adapter_path=adapter_path,
        wall_time_s=wall_time_s,
        trainable_params=_extract_trainable_params(log_text),
    )

    fused_path: str | None = None
    built_modelfile: str | None = None
    if fuse:
        fuse_cmd = [
            py_exe,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(model),
            "--adapter-path",
            str(adapter_path),
            "--save-path",
            str(fused_model_dir),
            "--dequantize",
        ]
        fuse_proc = subprocess.run(
            fuse_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if fuse_proc.returncode != 0:
            raise RuntimeError(
                f"mlx_lm fuse failed with exit code {fuse_proc.returncode}:\n{fuse_proc.stdout}"
            )
        fused_path = str(fused_model_dir)
        built_modelfile = str(build_modelfile(fused_model_dir=fused_model_dir, out_path=modelfile_path))

    return {
        "config_path": str(cfg_path),
        "training_log_path": str(log_jsonl_path),
        "train_manifest_path": str(manifest_path),
        "fused_model_dir": fused_path,
        "modelfile_path": built_modelfile,
        "events_count": len(events),
    }


def main() -> None:
    """CLI entrypoint for QLoRA config generation and MLX training."""
    parser = argparse.ArgumentParser(description="Configure and run MLX QLoRA fine-tuning.")
    parser.add_argument("--config-path", default="data/qlora/lora_config.yaml")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", default="data/qlora")
    parser.add_argument("--adapter-path", default="models/llama32_3b_qlora_adapters")
    parser.add_argument("--fused-model-dir", default="models/llama32_3b_qlora_fused")
    parser.add_argument("--modelfile-path", default="Modelfile.qlora")
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fuse", action="store_true")
    parser.add_argument("--write-config-only", action="store_true")
    args = parser.parse_args()

    if args.write_config_only:
        cfg = write_lora_config(
            out_path=args.config_path,
            model=args.model,
            data_dir=args.data_dir,
            adapter_path=args.adapter_path,
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_layers=args.num_layers,
            rank=args.rank,
            scale=args.scale,
            dropout=args.dropout,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )
        mf = build_modelfile(fused_model_dir=args.fused_model_dir, out_path=args.modelfile_path)
        print(json.dumps({"config_path": str(cfg), "modelfile_path": str(mf)}, indent=2))
        return

    summary = run_training(
        config_path=args.config_path,
        model=args.model,
        data_dir=args.data_dir,
        adapter_path=args.adapter_path,
        fused_model_dir=args.fused_model_dir,
        modelfile_path=args.modelfile_path,
        python_bin=args.python_bin,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        rank=args.rank,
        scale=args.scale,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        fuse=not args.no_fuse,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
