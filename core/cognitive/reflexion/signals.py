"""Health signal collection for Reflexion phase.

Gathers objective metrics, baseline comparisons, and phenomenological
signals to build a complete picture of cognitive health.

This module provides the HealthSignalCollector class that aggregates:
- Prediction confirmation rates (rolling window from Prediction nodes)
- Integration success rates (from CognitiveStateSnapshot nodes)
- Phase timing anomalies
- Error frequencies
- Phenomenological signals (thought diversity, SAE feature diversity)
- Rolling baselines (24-hour averages from ReflexionEntry nodes)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.cognitive.state import CognitiveState
    from core.cognitive.telemetry_evaluator import TelemetryEvaluator


class HealthSignalCollector:
    """Collects health signals from various sources.

    Aggregates metrics from:
    - Prediction nodes: confirmation rates over rolling window
    - CognitiveStateSnapshot nodes: integration success rates
    - Phase timing data: duration anomalies
    - Error logs: frequency by type
    - CognitiveState: phenomenological signals (diversity, resonance)
    - ReflexionEntry nodes: 24-hour rolling baselines
    - TelemetrySummary: biofeedback signals (logit dynamics, residual slopes)

    All async methods query Psyche for historical data. The synchronous
    collect_phenomenological_signals method extracts signals from the
    current CognitiveState.

    Attributes:
        _psyche: Client for querying historical data from Psyche graph
        _window_size: Number of cycles to include in rolling window calculations
        _telemetry_evaluator: Optional evaluator for biofeedback signals
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        window_size: int = 20,
        telemetry_evaluator: "TelemetryEvaluator | None" = None,
    ):
        """Initialize collector.

        Args:
            psyche: Client for querying historical data from Psyche graph
            window_size: Number of cycles to include in rolling window (default: 20)
            telemetry_evaluator: Optional evaluator for biofeedback telemetry signals
        """
        self._psyche = psyche
        self._window_size = window_size
        self._telemetry_evaluator = telemetry_evaluator

    async def collect_prediction_metrics(self, cycle: int) -> dict[str, float]:
        """Collect prediction confirmation metrics.

        Queries Prediction nodes that have been resolved (status = verified/falsified)
        within the rolling window and calculates the confirmation rate.

        Args:
            cycle: Current cycle number

        Returns:
            Dict with:
                - confirmation_rate: Proportion of verified predictions (0.0-1.0)
                - sample_size: Number of predictions in sample
        """
        query = """
        MATCH (p:Prediction)
        WHERE p.status IN ['verified', 'falsified']
          AND p.cycle_created >= $min_cycle
        RETURN (p.status = 'verified') AS verified, p.cycle_created AS cycle
        ORDER BY p.cycle_created DESC
        LIMIT $limit
        """
        min_cycle = max(0, cycle - self._window_size)

        results = await self._psyche.query(
            query,
            {"min_cycle": min_cycle, "limit": self._window_size}
        )

        if not results:
            return {"confirmation_rate": 0.0, "sample_size": 0}

        verified_count = sum(1 for r in results if r.get("verified"))
        total = len(results)

        return {
            "confirmation_rate": verified_count / total if total > 0 else 0.0,
            "sample_size": total,
        }

    async def collect_integration_metrics(self, cycle: int) -> dict[str, float]:
        """Collect integration success metrics.

        Queries CognitiveStateSnapshot nodes within the rolling window
        and calculates integration success rate.

        Args:
            cycle: Current cycle number

        Returns:
            Dict with:
                - success_rate: Proportion of successful integrations (0.0-1.0)
                - error_count: Number of failed integrations
        """
        query = """
        MATCH (s:CognitiveStateSnapshot)
        WHERE s.cycle >= $min_cycle
        RETURN s.integration_success AS success
        ORDER BY s.cycle DESC
        LIMIT $limit
        """
        min_cycle = max(0, cycle - self._window_size)

        results = await self._psyche.query(
            query,
            {"min_cycle": min_cycle, "limit": self._window_size}
        )

        if not results:
            return {"success_rate": 1.0, "error_count": 0}

        # Default to True if success field is missing (assume success)
        success_count = sum(1 for r in results if r.get("success", True))
        total = len(results)

        return {
            "success_rate": success_count / total if total > 0 else 1.0,
            "error_count": total - success_count,
        }

    async def collect_phase_timing(self, cycle: int) -> dict[str, dict[str, float]]:
        """Collect phase timing metrics.

        Queries CognitiveStateSnapshot nodes with phase_timing data
        and aggregates statistics by phase name.

        Args:
            cycle: Current cycle number

        Returns:
            Dict mapping phase names to timing stats:
                - average: Mean duration across window
                - min: Minimum duration
                - max: Maximum duration
                - latest: Most recent duration
        """
        query = """
        MATCH (s:CognitiveStateSnapshot)
        WHERE s.cycle >= $min_cycle AND s.phase_timing IS NOT NULL
        RETURN s.phase_timing AS timing
        ORDER BY s.cycle DESC
        LIMIT $limit
        """
        min_cycle = max(0, cycle - self._window_size)

        results = await self._psyche.query(
            query,
            {"min_cycle": min_cycle, "limit": self._window_size}
        )

        # Aggregate timings by phase
        phase_timings: dict[str, list[float]] = {}
        for r in results:
            timing = r.get("timing", {})
            if isinstance(timing, dict):
                for phase, duration in timing.items():
                    if phase not in phase_timings:
                        phase_timings[phase] = []
                    phase_timings[phase].append(float(duration))

        # Compute stats for each phase
        stats: dict[str, dict[str, float]] = {}
        for phase, durations in phase_timings.items():
            if durations:
                avg = sum(durations) / len(durations)
                stats[phase] = {
                    "average": avg,
                    "min": min(durations),
                    "max": max(durations),
                    "latest": durations[0] if durations else 0.0,
                }

        return stats

    async def collect_error_frequency(self, cycle: int) -> dict[str, int]:
        """Collect error frequency by type.

        Placeholder for future implementation. Would query error log
        nodes and aggregate by error type.

        Args:
            cycle: Current cycle number

        Returns:
            Dict mapping error types to counts (currently empty)
        """
        # This would query error logs - for now return empty
        # In production, this could query a dedicated error log table
        return {}

    def collect_phenomenological_signals(
        self,
        state: "CognitiveState"
    ) -> dict[str, float]:
        """Collect phenomenological signals from current state.

        Computes diversity metrics from the CognitiveState:
        - Thought diversity: ratio of unique concepts to total concepts
          (Jaccard-inspired measure of repetition)
        - SAE diversity: ratio of unique SAE feature indices to total features
          (measures feature redundancy)

        This is a synchronous method since it only reads from the state object.

        Args:
            state: Current cognitive state

        Returns:
            Dict with:
                - thought_diversity: Unique/total concept ratio (0.0-1.0)
                - sae_diversity: Unique/total SAE feature ratio (0.0-1.0)
                - concept_count: Total number of recent concepts
        """
        # Thought diversity: ratio of unique to total concepts
        concepts = state.recent_concepts if hasattr(state, "recent_concepts") else []
        if len(concepts) >= 2:
            unique = len(set(concepts))
            diversity = unique / len(concepts)
        else:
            diversity = 1.0

        # SAE feature diversity: ratio of unique to total feature indices
        sae_features = state.sae_features if hasattr(state, "sae_features") else []
        if sae_features:
            feature_indices = [f[0] for f in sae_features]
            feature_diversity = len(set(feature_indices)) / len(feature_indices)
        else:
            feature_diversity = 1.0

        return {
            "thought_diversity": diversity,
            "sae_diversity": feature_diversity,
            "concept_count": len(concepts),
        }

    def collect_telemetry_signals(
        self,
        state: "CognitiveState"
    ) -> dict[str, Any]:
        """Collect biofeedback telemetry signals from current state.

        Evaluates the TelemetrySummary (if available) using the TelemetryEvaluator
        to compute z-scores, trigger states, and normalized scores for health
        assessment.

        This runs in log-only mode initially for baseline calibration. The
        TelemetryEvaluator tracks rolling baselines internally and returns
        whether conditions would trigger if active.

        Args:
            state: Current cognitive state (must have telemetry_summary field)

        Returns:
            Dict with:
                - available: Whether telemetry was available for evaluation
                - confidence_score: Normalized confidence (0.0-1.0), -1 if unavailable
                - strain_score: Normalized strain (0.0-1.0), -1 if unavailable
                - margin_z: Z-score for top1-top2 margin
                - entropy_z: Z-score for logit entropy
                - slope_z: Z-score for residual slope
                - confidence_high: Boolean trigger predicate
                - strain_high: Boolean trigger predicate
                - should_verify: Whether full biofeedback verification triggered
                - would_trigger_if_active: Whether would trigger if not in shadow mode
                - baseline_samples: Number of samples in baseline (warmup indicator)
                - triggered_reason: Human-readable trigger reason if any
        """
        if not self._telemetry_evaluator:
            return {"available": False, "reason": "no_evaluator"}

        telemetry = getattr(state, "telemetry_summary", None)
        if telemetry is None:
            return {"available": False, "reason": "no_telemetry"}

        # Evaluate telemetry against baselines
        trigger_state = self._telemetry_evaluator.evaluate(telemetry)

        return {
            "available": True,
            "confidence_score": telemetry.confidence_score,
            "strain_score": telemetry.strain_score,
            "margin_z": trigger_state.margin_z,
            "entropy_z": trigger_state.entropy_z,
            "slope_z": trigger_state.slope_z,
            "confidence_high": trigger_state.confidence_high,
            "strain_high": trigger_state.strain_high,
            "should_verify": trigger_state.should_verify,
            "would_trigger_if_active": trigger_state.would_trigger_if_active,
            "baseline_samples": trigger_state.baseline_samples,
            "triggered_reason": trigger_state.triggered_reason,
        }

    async def collect_baselines(self) -> dict[str, float]:
        """Collect 24-hour rolling baselines.

        Queries ReflexionEntry nodes from the last 24 hours and computes
        average metric values for baseline comparison.

        Returns:
            Dict mapping metric names to baseline values:
                - prediction: Average prediction confirmation rate
                - integration: Average integration success rate
                - coherence: Average thought coherence
        """
        # Calculate cutoff timestamp (24 hours ago) as ISO 8601 string
        # FalkorDB doesn't support datetime() function, so we use string comparison
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        query = """
        MATCH (e:ReflexionEntry)
        WHERE e.timestamp > $cutoff
        RETURN
            avg(e.metrics_snapshot.pred_accuracy) AS pred_baseline,
            avg(e.metrics_snapshot.integration_rate) AS integration_baseline,
            avg(e.overall_coherence) AS coherence_baseline
        """

        results = await self._psyche.query(query, {"cutoff": cutoff})

        if results and results[0]:
            r = results[0]
            return {
                "prediction": r.get("pred_baseline") or 0.35,
                "integration": r.get("integration_baseline") or 0.92,
                "coherence": r.get("coherence_baseline") or 0.70,
            }

        # Default baselines if no history
        return {
            "prediction": 0.35,   # 35% confirmation rate baseline
            "integration": 0.92,  # 92% integration success baseline
            "coherence": 0.70,    # 70% thought coherence baseline
        }

    async def collect_all(self, state: "CognitiveState") -> dict[str, Any]:
        """Collect all health signals into a complete snapshot.

        Calls all collection methods and aggregates results into a
        single dictionary suitable for health assessment.

        Args:
            state: Current cognitive state

        Returns:
            Complete snapshot with keys:
                - prediction: Prediction metrics dict
                - integration: Integration metrics dict
                - timing: Phase timing stats dict
                - errors: Error frequency dict
                - phenomenological: Phenomenological signals dict
                - telemetry: Biofeedback telemetry signals dict
                - baselines: Rolling baselines dict
                - cycle: Current cycle number
        """
        cycle = state.cycle_count if hasattr(state, "cycle_count") else 0

        prediction = await self.collect_prediction_metrics(cycle)
        integration = await self.collect_integration_metrics(cycle)
        timing = await self.collect_phase_timing(cycle)
        errors = await self.collect_error_frequency(cycle)
        phenomenological = self.collect_phenomenological_signals(state)
        telemetry = self.collect_telemetry_signals(state)
        baselines = await self.collect_baselines()

        return {
            "prediction": prediction,
            "integration": integration,
            "timing": timing,
            "errors": errors,
            "phenomenological": phenomenological,
            "telemetry": telemetry,
            "baselines": baselines,
            "cycle": cycle,
        }
