"""Unit tests for AdaptiveThreshold controller."""

import pytest
from momo_kibidango.core.adaptive import AdaptiveThreshold


class TestAdaptiveThresholdInit:
    """Test initial state of AdaptiveThreshold."""

    def test_initial_thresholds(self, adaptive_threshold):
        assert adaptive_threshold.stage1_threshold == 0.10
        assert adaptive_threshold.stage2_threshold == 0.03

    def test_initial_thresholds_custom(self):
        at = AdaptiveThreshold(initial_stage1=0.20, initial_stage2=0.05)
        assert at.stage1_threshold == 0.20
        assert at.stage2_threshold == 0.05


class TestAdaptiveThresholdUpdate:
    """Test threshold adjustments via update()."""

    def test_update_raises_stage1(self):
        """When acceptance is too high (above target + 0.05), threshold increases."""
        at = AdaptiveThreshold(
            initial_stage1=0.10,
            target_acceptance_rate=0.70,
            ema_alpha=1.0,  # EMA follows batch rate exactly
            adjustment_step=0.01,
            warmup_iterations=0,  # No warmup
        )
        # 100% acceptance -> EMA = 1.0, well above 0.75 -> raise threshold
        at.update("stage1", accepted=10, total=10)
        assert at.stage1_threshold == pytest.approx(0.11)

    def test_update_lowers_stage1(self):
        """When acceptance is too low (below target - 0.05), threshold decreases."""
        at = AdaptiveThreshold(
            initial_stage1=0.10,
            target_acceptance_rate=0.70,
            ema_alpha=1.0,
            adjustment_step=0.01,
            warmup_iterations=0,
        )
        # 0% acceptance -> EMA = 0.0, well below 0.65 -> lower threshold
        at.update("stage1", accepted=0, total=10)
        assert at.stage1_threshold == pytest.approx(0.09)

    def test_warmup_no_adjustment(self, adaptive_threshold):
        """During warmup period, thresholds should not change."""
        original_stage1 = adaptive_threshold.stage1_threshold
        # Default warmup is 20; do 19 updates
        for _ in range(19):
            adaptive_threshold.update("stage1", accepted=10, total=10)
        assert adaptive_threshold.stage1_threshold == original_stage1

    def test_adjustment_after_warmup(self):
        """After warmup completes, adjustments should occur."""
        at = AdaptiveThreshold(
            initial_stage1=0.10,
            target_acceptance_rate=0.70,
            ema_alpha=1.0,
            adjustment_step=0.01,
            warmup_iterations=5,
        )
        # Do 5 warmup updates with high acceptance
        for _ in range(5):
            at.update("stage1", accepted=10, total=10)
        original = at.stage1_threshold
        # The 6th call should trigger an adjustment
        at.update("stage1", accepted=10, total=10)
        assert at.stage1_threshold > original

    def test_clamp_min(self):
        """Threshold should not go below min_threshold."""
        at = AdaptiveThreshold(
            initial_stage1=0.02,
            min_threshold=0.01,
            ema_alpha=1.0,
            adjustment_step=0.05,  # Large step
            warmup_iterations=0,
        )
        # Low acceptance -> try to lower
        at.update("stage1", accepted=0, total=10)
        assert at.stage1_threshold >= 0.01

    def test_clamp_max(self):
        """Threshold should not go above max_threshold."""
        at = AdaptiveThreshold(
            initial_stage1=0.49,
            max_threshold=0.50,
            ema_alpha=1.0,
            adjustment_step=0.05,
            warmup_iterations=0,
        )
        # High acceptance -> try to raise
        at.update("stage1", accepted=10, total=10)
        assert at.stage1_threshold <= 0.50

    def test_zero_total_ignored(self, adaptive_threshold):
        """Update with total=0 should be a no-op."""
        original = adaptive_threshold.stage1_threshold
        adaptive_threshold.update("stage1", accepted=0, total=0)
        assert adaptive_threshold.stage1_threshold == original

    def test_invalid_stage_raises(self, adaptive_threshold):
        with pytest.raises(ValueError, match="Unknown stage"):
            adaptive_threshold.update("stage3", accepted=1, total=1)


class TestAdaptiveThresholdStageIndependence:
    """Verify stage1 and stage2 are independent."""

    def test_stage2_independent(self):
        at = AdaptiveThreshold(
            initial_stage1=0.10,
            initial_stage2=0.03,
            ema_alpha=1.0,
            adjustment_step=0.01,
            warmup_iterations=0,
        )
        original_stage1 = at.stage1_threshold
        # Only update stage2 with high acceptance
        at.update("stage2", accepted=10, total=10)
        # stage1 should be unaffected
        assert at.stage1_threshold == original_stage1
        # stage2 should have changed
        assert at.stage2_threshold != 0.03


class TestAdaptiveThresholdReset:
    """Test reset functionality."""

    def test_reset(self):
        at = AdaptiveThreshold(
            initial_stage1=0.10,
            initial_stage2=0.03,
            ema_alpha=1.0,
            warmup_iterations=0,
        )
        # Modify thresholds
        at.update("stage1", accepted=10, total=10)
        at.update("stage2", accepted=10, total=10)
        assert at.stage1_threshold != 0.10 or at.stage2_threshold != 0.03

        at.reset()
        assert at.stage1_threshold == 0.10
        assert at.stage2_threshold == 0.03


class TestAdaptiveThresholdSnapshot:
    """Test snapshot output."""

    def test_snapshot(self, adaptive_threshold):
        snap = adaptive_threshold.snapshot()
        assert "target_acceptance_rate" in snap
        assert "ema_alpha" in snap
        assert "adjustment_step" in snap
        assert "warmup_iterations" in snap
        assert "stage1" in snap
        assert "stage2" in snap
        assert "threshold" in snap["stage1"]
        assert "ema" in snap["stage1"]
        assert "update_count" in snap["stage1"]
        assert snap["stage1"]["threshold"] == 0.10
        assert snap["stage2"]["threshold"] == 0.03

    def test_snapshot_reflects_updates(self):
        at = AdaptiveThreshold(warmup_iterations=0, ema_alpha=1.0, adjustment_step=0.01)
        at.update("stage1", accepted=10, total=10)
        snap = at.snapshot()
        assert snap["stage1"]["update_count"] == 1
        assert snap["stage1"]["ema"] == pytest.approx(1.0)
