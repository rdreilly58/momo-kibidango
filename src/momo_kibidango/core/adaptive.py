"""Adaptive threshold controller for speculative decoding.

Tunes acceptance thresholds at runtime based on observed acceptance rates
using an exponential moving average (EMA).  When the EMA drifts above the
target range the threshold is raised (too lenient, wasting compute); when it
drifts below, the threshold is lowered (too strict, rejecting too much).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ThresholdState:
    """Internal state for a single stage threshold."""
    threshold: float
    ema: float
    update_count: int = 0


class AdaptiveThreshold:
    """Adaptive threshold controller using exponential moving average.

    Parameters
    ----------
    initial_stage1:
        Starting threshold for stage-1 (qualifier) verification.
    initial_stage2:
        Starting threshold for stage-2 (target) verification.
    target_acceptance_rate:
        Desired long-run acceptance rate.
    ema_alpha:
        Smoothing factor for the EMA (higher = more weight on recent).
    adjustment_step:
        Amount to raise or lower the threshold per update.
    min_threshold:
        Floor for any threshold value.
    max_threshold:
        Ceiling for any threshold value.
    warmup_iterations:
        Number of updates before adjustments begin.
    """

    def __init__(
        self,
        initial_stage1: float = 0.10,
        initial_stage2: float = 0.03,
        target_acceptance_rate: float = 0.70,
        ema_alpha: float = 0.05,
        adjustment_step: float = 0.005,
        min_threshold: float = 0.01,
        max_threshold: float = 0.50,
        warmup_iterations: int = 20,
    ) -> None:
        self._initial_stage1 = initial_stage1
        self._initial_stage2 = initial_stage2
        self._target = target_acceptance_rate
        self._alpha = ema_alpha
        self._step = adjustment_step
        self._min = min_threshold
        self._max = max_threshold
        self._warmup = warmup_iterations

        self._states: dict[str, ThresholdState] = {
            "stage1": ThresholdState(threshold=initial_stage1, ema=target_acceptance_rate),
            "stage2": ThresholdState(threshold=initial_stage2, ema=target_acceptance_rate),
        }

    # -- Public properties ---------------------------------------------------

    @property
    def stage1_threshold(self) -> float:
        """Current stage-1 (qualifier) threshold."""
        return self._states["stage1"].threshold

    @property
    def stage2_threshold(self) -> float:
        """Current stage-2 (target) threshold."""
        return self._states["stage2"].threshold

    # -- Core logic ----------------------------------------------------------

    def update(self, stage: str, accepted: int, total: int) -> None:
        """Update the EMA and, after warmup, adjust the threshold.

        Parameters
        ----------
        stage:
            ``"stage1"`` or ``"stage2"``.
        accepted:
            Number of tokens accepted in this batch.
        total:
            Total number of tokens evaluated in this batch.
        """
        if stage not in self._states:
            raise ValueError(f"Unknown stage '{stage}'; expected 'stage1' or 'stage2'")

        state = self._states[stage]

        # Guard against division by zero
        if total <= 0:
            return

        batch_rate = accepted / total
        state.ema = self._alpha * batch_rate + (1 - self._alpha) * state.ema
        state.update_count += 1

        # Skip adjustment during warmup
        if state.update_count < self._warmup:
            logger.debug(
                "Adaptive %s warmup %d/%d  ema=%.4f",
                stage, state.update_count, self._warmup, state.ema,
            )
            return

        # Adjust threshold based on EMA vs target
        if state.ema > self._target + 0.05:
            # Too lenient — raise threshold to save compute
            state.threshold = min(state.threshold + self._step, self._max)
            logger.debug(
                "Adaptive %s: ema=%.4f > target+0.05 => raised threshold to %.4f",
                stage, state.ema, state.threshold,
            )
        elif state.ema < self._target - 0.05:
            # Too strict — lower threshold to accept more
            state.threshold = max(state.threshold - self._step, self._min)
            logger.debug(
                "Adaptive %s: ema=%.4f < target-0.05 => lowered threshold to %.4f",
                stage, state.ema, state.threshold,
            )

    def snapshot(self) -> dict:
        """Return current state as a dict suitable for logging or serialization."""
        return {
            "target_acceptance_rate": self._target,
            "ema_alpha": self._alpha,
            "adjustment_step": self._step,
            "warmup_iterations": self._warmup,
            "stage1": {
                "threshold": self._states["stage1"].threshold,
                "ema": self._states["stage1"].ema,
                "update_count": self._states["stage1"].update_count,
            },
            "stage2": {
                "threshold": self._states["stage2"].threshold,
                "ema": self._states["stage2"].ema,
                "update_count": self._states["stage2"].update_count,
            },
        }

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._states["stage1"] = ThresholdState(
            threshold=self._initial_stage1, ema=self._target,
        )
        self._states["stage2"] = ThresholdState(
            threshold=self._initial_stage2, ema=self._target,
        )
        logger.info("Adaptive thresholds reset to initial values")
