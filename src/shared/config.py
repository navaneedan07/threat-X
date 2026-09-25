"""Config loading, so no stage hard-codes a parameter.

Every stage reads its own file from ``configs/``:

===============  ===========================================================
File             Covers
===============  ===========================================================
``data.yaml``    sources, variables, grid, time, ensemble
``model.yaml``   detector, graph/GNN, transition, downscaling
``tracking.yaml``association thresholds, lifecycle, severity bands
``validation.yaml``metric sets, PASS/DEGRADE/SUPPRESS gates
===============  ===========================================================

Two rules this module exists to enforce:

1. **A parameter comes from config, not from source.** ``get_value`` returns
   ``None`` for an unresolved value rather than a fallback guess.
2. **An unresolved value stays unresolved.** Unset keys are deliberately
   ``null`` in these files. Callers must handle ``None`` by disabling the
   behaviour and reporting it, never by substituting a number. See
   ``docs/experiments.md`` for why gate thresholds are null.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class ConfigError(RuntimeError):
    """A config file exists but could not be read."""


@cache
def load_config(name: str, configs_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML file from ``configs/``.

    ``name`` is a bare filename (``"model.yaml"``). A missing file returns an
    empty dict — that is not an error, because a stage must still be importable
    and testable when the config tree is unavailable. A file that exists but
    cannot be parsed *is* an error, so a typo never silently becomes defaults.
    """
    path = Path(configs_dir) / name if configs_dir else CONFIGS_DIR / name
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is in requirements
        raise ConfigError(f"PyYAML is required to read {path}") from exc
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    return loaded or {}


def get_value(
    config: dict[str, Any],
    dotted_path: str,
    default: Any = None,
) -> Any:
    """Read ``"downscaling.interpolation.order"`` from a loaded config.

    Returns ``default`` when the path is absent **or when the value is
    explicitly null**, because those two cases mean the same thing to a caller:
    this parameter has not been resolved. Handling them the same way is what
    stops a ``null`` in config from quietly becoming ``0`` in code.
    """
    node: Any = config
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def require_value(config: dict[str, Any], dotted_path: str) -> Any:
    """Like :func:`get_value`, but raise when the value is unresolved.

    Use this for a parameter the caller genuinely cannot proceed without, so the
    failure names the config key to fill in instead of surfacing as a
    downstream ``TypeError``.
    """
    value = get_value(config, dotted_path)
    if value is None:
        raise ConfigError(
            f"config value {dotted_path!r} is unresolved (null or absent). "
            "Fill it in the relevant configs/ file — do not substitute a guess."
        )
    return value


def stage_config(name: str, configs_dir: str | Path | None = None) -> dict[str, Any]:
    """Convenience: ``stage_config("model.yaml")`` → parsed dict."""
    return load_config(name, configs_dir)
