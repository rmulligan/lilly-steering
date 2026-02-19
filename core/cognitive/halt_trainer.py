"""HALT Probe Trainer: Train epistemic uncertainty probe from collected examples.

Based on arXiv:2601.14210 - Hallucination Assessment via Latent Testing.
Trains the HALT probe's MLP on collected hidden states with ground-truth labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.cognitive.halt_probe import HALTProbe
    from core.cognitive.halt_collector import HALTTrainingCollector, HALTTrainingExample

# Handle optional torch dependency
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore


class HALTTrainer:
    """Trains HALT epistemic probe from collected training examples.

    The trainer retrieves labeled examples from the collector and trains
    the probe's MLP using confidence-weighted BCE loss. Training is triggered
    when sufficient unused examples have accumulated.

    Training algorithm:
        1. Retrieve high-confidence examples (predictions, faithfulness)
        2. Fall back to lower-confidence examples if insufficient
        3. Prepare tensors: X (hidden states), y (labels), weights (confidences)
        4. Train with AdamW optimizer and weighted BCE loss
        5. Mark examples as used after training

    Usage:
        trainer = HALTTrainer(probe, collector, settings)
        if trainer.should_train():
            result = await trainer.train()
            if result["status"] == "success":
                probe.save_weights(path)

    Attributes:
        TRAINING_THRESHOLD: Minimum unused examples to trigger training (100)
        BATCH_SIZE: Training batch size (32)
        EPOCHS: Number of training epochs (3)
        LEARNING_RATE: AdamW learning rate (1e-4)
    """

    # Class constants for training configuration
    TRAINING_THRESHOLD: int = 100
    BATCH_SIZE: int = 32
    EPOCHS: int = 3
    LEARNING_RATE: float = 1e-4

    def __init__(
        self,
        probe: "HALTProbe",
        collector: "HALTTrainingCollector",
        settings: "Settings",
    ) -> None:
        """Initialize HALT trainer.

        Args:
            probe: The HALT probe to train.
            collector: Source of training data with labeled examples.
            settings: Application settings.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for HALTTrainer")

        self._probe = probe
        self._collector = collector
        self._settings = settings

    async def should_train(self) -> bool:
        """Check if training should be triggered.

        Training is triggered when the number of unused examples
        reaches or exceeds TRAINING_THRESHOLD.

        Returns:
            True if training should be triggered, False otherwise.
        """
        stats = await self._collector.get_stats()
        unused = stats["total"] - stats["used"]
        return unused >= self.TRAINING_THRESHOLD

    async def train(self) -> dict:
        """Train the probe on collected examples.

        Training flow:
            1. Get high-confidence examples first (min_confidence=0.7)
            2. Fall back to lower confidence (0.4) if < 50 examples
            3. Return insufficient_data if < 20 examples
            4. Prepare tensors and train with weighted BCE loss
            5. Mark examples as used after training

        Returns:
            Dictionary with training results:
            - status: "success" or "insufficient_data"
            - count: Number of examples (if insufficient)
            - examples_used: Number of examples used (if success)
            - epochs: Number of training epochs (if success)
            - final_loss: Final training loss (if success)
        """
        # Get high-confidence examples first (predictions and faithfulness)
        examples = await self._collector.get_training_batch(
            min_confidence=0.7, limit=500
        )

        # Fall back to lower confidence if insufficient
        if len(examples) < 50:
            logger.info(
                "Only %d high-confidence examples, falling back to lower confidence",
                len(examples),
            )
            examples = await self._collector.get_training_batch(
                min_confidence=0.4, limit=500
            )

        # Check minimum threshold
        if len(examples) < 20:
            logger.warning(
                "Insufficient training data: %d examples (need >= 20)",
                len(examples),
            )
            return {"status": "insufficient_data", "count": len(examples)}

        logger.info("Training HALT probe with %d examples", len(examples))

        # Prepare training tensors
        X, y, weights = self._prepare_tensors(examples)

        # Train the probe
        final_loss = self._train_loop(X, y, weights)

        # Mark examples as used
        example_ids = [ex.example_id for ex in examples]
        await self._collector.mark_examples_used(example_ids)

        # Mark probe as trained
        self._probe._loaded = True

        logger.info(
            "HALT probe training complete: %d examples, final loss %.4f",
            len(examples),
            final_loss,
        )

        return {
            "status": "success",
            "examples_used": len(examples),
            "epochs": self.EPOCHS,
            "final_loss": final_loss,
        }

    def _prepare_tensors(
        self, examples: "list[HALTTrainingExample]"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Prepare training tensors from examples.

        Args:
            examples: List of HALTTrainingExample instances.

        Returns:
            Tuple of (X, y, weights):
            - X: Hidden states tensor [N, d_model]
            - y: Labels tensor [N, 1]
            - weights: Confidence weights tensor [N, 1]
        """
        # Extract hidden states, labels, and confidences
        hidden_states = [ex.hidden_states_aggregated for ex in examples]
        labels = [ex.label for ex in examples]
        confidences = [ex.label_confidence for ex in examples]

        # Convert to tensors
        X = torch.tensor(hidden_states, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(confidences, dtype=torch.float32).unsqueeze(1)

        return X, y, weights

    def _train_loop(
        self,
        X: "torch.Tensor",
        y: "torch.Tensor",
        weights: "torch.Tensor",
    ) -> float:
        """Run training loop with weighted BCE loss.

        Args:
            X: Hidden states tensor [N, d_model].
            y: Labels tensor [N, 1].
            weights: Confidence weights tensor [N, 1].

        Returns:
            Final training loss value.
        """
        # Get device from probe
        device = next(self._probe.probe.parameters()).device

        # Move tensors to device
        X = X.to(device)
        y = y.to(device)
        weights = weights.to(device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y, weights)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
        )

        # Setup optimizer and loss function
        optimizer = torch.optim.AdamW(
            self._probe.probe.parameters(),
            lr=self.LEARNING_RATE,
        )
        criterion = nn.BCELoss(reduction="none")

        # Training loop
        self._probe.probe.train()
        final_loss = 0.0

        for epoch in range(self.EPOCHS):
            epoch_loss = 0.0
            batch_count = 0

            for batch_X, batch_y, batch_weights in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self._probe.probe(batch_X)

                # Compute weighted loss
                loss = criterion(outputs, batch_y)
                weighted_loss = (loss * batch_weights).mean()

                # Backward pass
                weighted_loss.backward()
                optimizer.step()

                epoch_loss += weighted_loss.item()
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            logger.debug(
                "HALT training epoch %d/%d, loss: %.4f",
                epoch + 1,
                self.EPOCHS,
                avg_epoch_loss,
            )
            final_loss = avg_epoch_loss

        # Set back to evaluation mode
        self._probe.probe.train(False)

        return final_loss
