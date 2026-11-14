"""Prometheus observability for ProfileScorer's Eq. 4b learning state."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, Optional

try:
    from prometheus_client import CollectorRegistry, Gauge, Histogram, REGISTRY, start_http_server
except ImportError as exc:  # pragma: no cover - covered by isolated import test
    raise ImportError(
        "Prometheus metrics require the optional dependency. Install with "
        "`pip install graph-attention-engine[prometheus]`."
    ) from exc

from gae.profile_scorer import ProfileScorer


class RGIMetrics:
    """Export a ``ProfileScorer``'s verified-outcome state to Prometheus.

    Gauges reflect the scorer's Eq. 4b centroid-learning outcomes only when
    :meth:`update` is called. Use :meth:`score_timer` around ``scorer.score()``
    to record request latency without changing score semantics.

    Parameters
    ----------
    scorer:
        Scorer whose public observability methods are exported.
    registry:
        Prometheus registry. Defaults to the process-wide registry; inject a
        dedicated registry for tests or isolated embedding applications.

    Reference: blog Eq. 4b-final (verified centroid pull/push learning).
    """

    _CONSERVATION_VALUES = {
        "RED": 0.0,
        "AMBER": 1.0,
        "YELLOW": 1.0,
        "GREEN": 2.0,
    }
    _EVOLVER_PHASES = {
        "Shadow": 0.0,
        "Promoted": 1.0,
        "RolledBack": 2.0,
    }

    def __init__(
        self,
        scorer: ProfileScorer,
        registry: Optional[CollectorRegistry] = None,
    ) -> None:
        self.scorer = scorer
        self._registry = REGISTRY if registry is None else registry
        self._conservation_status = Gauge(
            "rgi_conservation_status",
            "Conservation status: 0=RED, 1=YELLOW/AMBER, 2=GREEN.",
            ["category"],
            registry=self._registry,
        )
        self._verified_count = Gauge(
            "rgi_verified_count",
            "Verified decisions by category.",
            ["category"],
            registry=self._registry,
        )
        self._accuracy = Gauge(
            "rgi_accuracy",
            "Verified-outcome accuracy by category, from zero through one.",
            ["category"],
            registry=self._registry,
        )
        self._centroid_drift = Gauge(
            "rgi_centroid_drift",
            "Latest Eq. 4b centroid-update magnitude by category.",
            ["category"],
            registry=self._registry,
        )
        self._score_latency = Histogram(
            "rgi_score_latency_seconds",
            "Elapsed time spent in ProfileScorer.score().",
            registry=self._registry,
        )
        self._agentevolver_phase = Gauge(
            "rgi_agentevolver_phase",
            "AgentEvolver phase: 0=Shadow, 1=Promoted, 2=RolledBack.",
            ["variant"],
            registry=self._registry,
        )

    def _category_label(self, category_index: int) -> str:
        """Return a stable Prometheus label for one scorer category."""
        if self.scorer.categories is None:
            return str(category_index)
        return self.scorer.categories[category_index]

    def update(self) -> None:
        """Refresh category gauges from the current scorer state (Eq. 4b)."""
        conservation_value = self._CONSERVATION_VALUES.get(
            self.scorer.conservation_status.upper(),
            0.0,
        )
        for category_index in range(self.scorer.n_categories):
            category = self._category_label(category_index)
            self._conservation_status.labels(category=category).set(conservation_value)
            self._verified_count.labels(category=category).set(
                self.scorer.verified_count(category_index)
            )
            self._accuracy.labels(category=category).set(
                self.scorer.accuracy(category_index)
            )
            self._centroid_drift.labels(category=category).set(
                self.scorer.centroid_drift(category_index)
            )

    def set_agentevolver_phase(self, variant: str, phase: str = "Shadow") -> None:
        """Set the placeholder AgentEvolver lifecycle gauge for one variant."""
        try:
            phase_value = self._EVOLVER_PHASES[phase]
        except KeyError as exc:
            supported = ", ".join(self._EVOLVER_PHASES)
            raise ValueError(f"unknown AgentEvolver phase {phase!r}; use {supported}") from exc
        self._agentevolver_phase.labels(variant=variant).set(phase_value)

    def start(self, port: int = 9090) -> None:
        """Start the Prometheus HTTP exporter on ``port``."""
        start_http_server(port, registry=self._registry)

    @contextmanager
    def score_timer(self) -> Iterator[None]:
        """Record elapsed ``ProfileScorer.score()`` time in the histogram."""
        started = perf_counter()
        try:
            yield
        finally:
            self._score_latency.observe(perf_counter() - started)
