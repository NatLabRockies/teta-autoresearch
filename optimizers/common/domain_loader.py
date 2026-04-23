"""Resolve a domain's optimizer hooks by name.

Domains ship a `search/` subpackage with three required callables and a few
optional exports (WARM_START_CONFIGS, ALL_FAMILIES, default_data_path).
See EXTENDING.md for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Callable

REQUIRED_ATTRS = ("sample_config", "load_and_engineer", "run_trial")


@dataclass
class DomainHooks:
    name: str
    module: ModuleType

    @property
    def sample_config(self) -> Callable[..., dict]:
        return self.module.sample_config

    @property
    def load_and_engineer(self) -> Callable[[str], Any]:
        return self.module.load_and_engineer

    @property
    def run_trial(self) -> Callable[..., float]:
        return self.module.run_trial

    def optional(self, name: str, default: Any = None) -> Any:
        return getattr(self.module, name, default)


def load(domain: str) -> DomainHooks:
    try:
        module = import_module(f"domains.{domain}.search")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"domain {domain!r}: cannot import `domains.{domain}.search` — "
            f"does `domains/{domain}/search/__init__.py` exist and re-export the hooks?"
        ) from exc

    missing = [a for a in REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        raise AttributeError(
            f"domain {domain!r} is missing required exports in "
            f"`domains.{domain}.search`: {missing}"
        )
    return DomainHooks(name=domain, module=module)
