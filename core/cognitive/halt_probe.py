"""HALT Probe: Epistemic uncertainty detection from hidden states.

Based on arXiv:2601.14210 - Hallucination Assessment via Latent Testing.
Reads hallucination risk from intermediate layers where epistemic signals
are preserved before decoding attenuation.

Key insight: Epistemic signals are encoded in intermediate layers (~70% depth)
but attenuated by final decoding layers. Probing question tokens before
generation enables zero-latency routing decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Handle optional torch dependency
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class HALTProbeConfig:
    """Configuration for HALT epistemic probe.

    Attributes:
        hidden_dim: MLP hidden layer dimension
        num_layers: Number of MLP layers (including output)
        probe_layer: Target transformer layer to probe (~70% depth)
        aggregation: Sequence aggregation method (mean, max, last, attention)
        dropout: Dropout probability for regularization
    """

    hidden_dim: int = 256
    num_layers: int = 4
    probe_layer: int = 20  # ~70% depth for Qwen3-8B (28 layers)
    aggregation: str = "mean"  # mean, max, last
    dropout: float = 0.1


@dataclass
class HALTProbeResult:
    """Result from HALT epistemic probe.

    Attributes:
        epistemic_confidence: Probability of reliable answer, p in [0, 1]
        probe_layer: Which transformer layer was probed
        aggregation_method: How sequence positions were aggregated
        latency_ms: Probe execution time in milliseconds
    """

    epistemic_confidence: float
    probe_layer: int
    aggregation_method: str
    latency_ms: float


class HALTProbe:
    """Lightweight MLP probe for epistemic uncertainty detection.

    The probe reads hidden states from an intermediate transformer layer
    and predicts the probability that the model's response will be reliable
    (not a hallucination).

    Architecture:
        Input: [d_model] hidden state vector (aggregated across sequence)
        MLP: hidden_dim -> hidden_dim -> ... -> 1
        Output: Sigmoid probability in [0, 1]

    Usage:
        probe = HALTProbe(config, d_model=4096)
        result = await probe.probe_hidden_states(hidden_states)
        if result.epistemic_confidence < 0.4:
            # Trigger simulation for rigorous hypothesis testing
            ...

    Note:
        The probe requires training on (question, hidden_states, correctness_label)
        data. Initial deployment uses random initialization (~0.5 output) with
        calibration over time via Bayesian updates.
    """

    def __init__(
        self,
        config: HALTProbeConfig,
        d_model: int = 4096,
        weights_path: Optional[Path] = None,
    ):
        """Initialize HALT probe.

        Args:
            config: Probe configuration
            d_model: Model hidden dimension (4096 for Qwen3-8B)
            weights_path: Optional path to pre-trained probe weights
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for HALTProbe")

        self.config = config
        self.d_model = d_model
        self.probe = self._build_probe(d_model, config)
        self._loaded = False

        if weights_path and weights_path.exists():
            self._load_weights(weights_path)

    def _build_probe(self, d_model: int, config: HALTProbeConfig) -> "nn.Module":
        """Build MLP probe architecture.

        Args:
            d_model: Input dimension (model hidden size)
            config: Probe configuration

        Returns:
            Sequential MLP module
        """
        layers = []
        in_dim = d_model

        # Hidden layers with ReLU and Dropout
        for i in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.hidden_dim

        # Output layer with Sigmoid
        layers.append(nn.Linear(config.hidden_dim, 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _load_weights(self, weights_path: Path) -> None:
        """Load pre-trained probe weights.

        Args:
            weights_path: Path to saved weights file
        """
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.probe.load_state_dict(state_dict)
            self._loaded = True
            logger.info(f"Loaded HALT probe weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load HALT probe weights: {e}")
            self._loaded = False

    def save_weights(self, weights_path: Path) -> None:
        """Save probe weights for later reuse.

        Args:
            weights_path: Path to save weights
        """
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.probe.state_dict(), weights_path)
        logger.info(f"Saved HALT probe weights to {weights_path}")

    def to(self, device: str) -> "HALTProbe":
        """Move probe to specified device.

        Args:
            device: Target device (cuda, cpu)

        Returns:
            Self for chaining
        """
        self.probe = self.probe.to(device)
        return self

    def _process_hidden_states(
        self,
        hidden_states: "torch.Tensor",
        question_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Process hidden states: handle batch dimension, apply mask, aggregate.

        This helper consolidates the common logic shared between probe_hidden_states
        and probe_hidden_states_sync.

        Args:
            hidden_states: Activations from probe_layer, shape [seq_len, d_model]
                or [batch, seq_len, d_model]
            question_mask: Optional boolean mask for question tokens only.
                If provided, only masked positions are used for aggregation.

        Returns:
            Aggregated tensor of shape [d_model]
        """
        # Handle batch dimension
        if hidden_states.dim() == 3:
            # [batch, seq, d_model] -> use first batch item
            hidden_states = hidden_states[0]

        # Apply question mask if provided
        if question_mask is not None:
            states = hidden_states[question_mask]
        else:
            states = hidden_states

        # Aggregate across sequence dimension
        if self.config.aggregation == "mean":
            aggregated = states.mean(dim=0)
        elif self.config.aggregation == "max":
            aggregated = states.max(dim=0).values
        elif self.config.aggregation == "last":
            aggregated = states[-1]
        else:
            # Default to mean
            aggregated = states.mean(dim=0)

        return aggregated

    async def probe_hidden_states(
        self,
        hidden_states: "torch.Tensor",
        question_mask: Optional["torch.Tensor"] = None,
    ) -> HALTProbeResult:
        """Run probe on hidden states from target layer.

        The probe aggregates hidden states across the sequence dimension
        using the configured method, then passes through the MLP to
        produce an epistemic confidence score.

        Args:
            hidden_states: Activations from probe_layer, shape [seq_len, d_model]
                or [batch, seq_len, d_model]
            question_mask: Optional boolean mask for question tokens only.
                If provided, only masked positions are used for aggregation.

        Returns:
            HALTProbeResult with epistemic_confidence in [0, 1]
        """
        start = time.perf_counter()

        # Process hidden states: batch handling, masking, aggregation
        aggregated = self._process_hidden_states(hidden_states, question_mask)

        # Run probe (no gradients needed for inference)
        with torch.no_grad():
            confidence = self.probe(aggregated).item()

        latency = (time.perf_counter() - start) * 1000

        return HALTProbeResult(
            epistemic_confidence=confidence,
            probe_layer=self.config.probe_layer,
            aggregation_method=self.config.aggregation,
            latency_ms=latency,
        )

    def probe_hidden_states_sync(
        self,
        hidden_states: "torch.Tensor",
        question_mask: Optional["torch.Tensor"] = None,
    ) -> HALTProbeResult:
        """Synchronous version of probe_hidden_states for non-async contexts.

        Args:
            hidden_states: Activations from probe_layer
            question_mask: Optional boolean mask for question tokens

        Returns:
            HALTProbeResult with epistemic_confidence
        """
        start = time.perf_counter()

        # Process hidden states: batch handling, masking, aggregation
        aggregated = self._process_hidden_states(hidden_states, question_mask)

        # Run probe
        with torch.no_grad():
            confidence = self.probe(aggregated).item()

        latency = (time.perf_counter() - start) * 1000

        return HALTProbeResult(
            epistemic_confidence=confidence,
            probe_layer=self.config.probe_layer,
            aggregation_method=self.config.aggregation,
            latency_ms=latency,
        )

    @property
    def is_trained(self) -> bool:
        """Check if probe has trained weights loaded."""
        return self._loaded
