from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from brian_sphere_llm.utils.logging import JsonlLogger, write_json

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


_DASHBOARD_METRIC_KEYS = {
    "loss",
    "lm_loss",
    "validation_loss",
    "perplexity",
    "learning_rate",
    "tokens_per_second",
    "train_step_time_seconds",
    "cuda_memory_allocated_mb",
    "cuda_max_memory_allocated_mb",
    "block_load_entropy_normalized",
    "route_entropy",
    "average_route_steps",
    "route_path_diversity",
    "recur_ratio",
    "random_route_probability",
    "route_logit_noise_std",
    "bdre_key_weight_entropy",
    "bdre_value_weight_entropy",
    "bdre_cache_memory_mb",
}


@dataclass(frozen=True)
class TerminalDashboardConfig:
    enabled: bool = False
    interval: int = 1
    position_interval: int = 100
    sample_index: int = 0
    output_dir: str = "terminal_dashboard"

    @classmethod
    def from_train_config(cls, config: Mapping[str, Any]) -> "TerminalDashboardConfig":
        raw = config.get("terminal_dashboard", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("terminal_dashboard must be a mapping.")
        return cls(
            enabled=_bool_value(raw.get("enabled", False), "terminal_dashboard.enabled"),
            interval=_int_value(raw.get("interval", 1), "terminal_dashboard.interval", minimum=1),
            position_interval=_int_value(
                raw.get("position_interval", 100),
                "terminal_dashboard.position_interval",
                minimum=1,
            ),
            sample_index=_int_value(
                raw.get("sample_index", 0),
                "terminal_dashboard.sample_index",
                minimum=0,
            ),
            output_dir=_nonempty_string(
                raw.get("output_dir", "terminal_dashboard"),
                "terminal_dashboard.output_dir",
            ),
        )

    def train_due(self, step: int, max_steps: int) -> bool:
        return self.enabled and (step % self.interval == 0 or step == max_steps)

    def positions_due(self, step: int, max_steps: int) -> bool:
        return self.enabled and (step % self.position_interval == 0 or step == 1 or step == max_steps)


class LiveTelemetryWriter:
    """Append-only, best-effort telemetry for the detached terminal dashboard."""

    schema_version = 1

    def __init__(
        self,
        run_dir: str | Path,
        config: TerminalDashboardConfig,
        *,
        run_name: str,
        max_steps: int,
        start_step: int,
        num_blocks: int,
        model_name: str,
        world_size: int,
        device_name: str,
        device_memory_mb: float | None,
    ) -> None:
        self.config = config
        self.directory = Path(run_dir) / config.output_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.errors_path = self.directory / "errors.jsonl"
        self._events = JsonlLogger(self.events_path)
        self._errors = JsonlLogger(self.errors_path)
        self._failed = False
        write_json(
            {
                "schema_version": self.schema_version,
                "run_name": run_name,
                "model_name": model_name,
                "max_steps": max_steps,
                "start_step": start_step,
                "num_blocks": num_blocks,
                "world_size": world_size,
                "device_name": device_name,
                "device_memory_mb": device_memory_mb,
            },
            self.directory / "metadata.json",
        )
        self.write_status("running", step=start_step)

    @property
    def healthy(self) -> bool:
        return not self._failed

    def write_train(
        self,
        row: Mapping[str, Any],
        *,
        step: int,
        max_steps: int,
        route: list[int],
        token_index: int,
        positions: list[list[float]] | None,
    ) -> None:
        payload: dict[str, Any] = {
            "kind": "train",
            "schema_version": self.schema_version,
            "step": step,
            "max_steps": max_steps,
            "token_index": token_index,
            "route": route,
            "metrics": _numeric_metrics(row),
        }
        if positions is not None:
            payload["positions"] = positions
        self._write(payload)

    def write_eval(self, row: Mapping[str, Any], *, step: int) -> None:
        self._write(
            {
                "kind": "eval",
                "schema_version": self.schema_version,
                "step": step,
                "metrics": _numeric_metrics(row),
            }
        )

    def write_status(self, status: str, *, step: int) -> None:
        self._write(
            {
                "kind": "status",
                "schema_version": self.schema_version,
                "step": step,
                "status": status,
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        if self._failed:
            return
        try:
            self._events.write(payload)
        except Exception as exc:  # pragma: no cover - telemetry must never stop training.
            self._failed = True
            try:
                self._errors.write(
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "event_kind": payload.get("kind"),
                        "event_step": payload.get("step"),
                    }
                )
            except Exception:
                pass


def extract_route_trace(
    outputs: Mapping[str, Any],
    *,
    num_internal_blocks: int,
    sample_index: int,
) -> list[int]:
    """Copy one real route with a single device transfer after the forward is complete."""

    if torch is None:
        return []
    route_info = outputs.get("route_info")
    if not isinstance(route_info, Mapping):
        return []
    selected = route_info.get("selected_actions")
    if not isinstance(selected, list) or not selected:
        return []
    samples: list[torch.Tensor] = []
    for action in selected:
        if not isinstance(action, torch.Tensor) or action.numel() == 0:
            continue
        if action.ndim == 0:
            if sample_index != 0:
                return []
            sample = action.reshape(1)
        else:
            if sample_index >= action.shape[0]:
                return []
            sample = action[sample_index].reshape(-1)[-1:]
        samples.append(sample.to(dtype=torch.long))
    if not samples:
        return []
    actions = torch.cat(samples).detach().cpu().tolist()
    route: list[int] = []
    for value in actions:
        action = int(value)
        if action >= num_internal_blocks:
            break
        if action >= 0:
            route.append(action)
    return route


def internal_position_snapshot(model: Any, *, num_internal_blocks: int) -> list[list[float]] | None:
    if torch is None or F is None:
        return None
    table = getattr(model, "position_table", None)
    embeddings = getattr(table, "embeddings", None)
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        return None
    if embeddings.shape[0] < num_internal_blocks:
        return None
    values = F.normalize(embeddings[:num_internal_blocks], dim=-1).detach().float().cpu().tolist()
    return [[round(float(value), 6) for value in row] for row in values]


def cuda_device_metadata(device: Any) -> tuple[str, float | None]:
    if torch is None or getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return str(device), None
    properties = torch.cuda.get_device_properties(device)
    return properties.name, properties.total_memory / (1024.0 * 1024.0)


def _numeric_metrics(row: Mapping[str, Any]) -> dict[str, int | float | bool]:
    result: dict[str, int | float | bool] = {}
    for key, value in row.items():
        if key not in _DASHBOARD_METRIC_KEYS:
            continue
        if isinstance(value, bool):
            result[str(key)] = value
        elif isinstance(value, int):
            result[str(key)] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[str(key)] = value
    return result


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _int_value(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()
