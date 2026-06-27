"""sa3_control.mert_reward — Part 5 of the longform-development system.

Trained MERT continuation reward / re-ranker. Replaces the per-window
Audiobox+MERT subprocess (slow, N decodes per window) with a small learned
model that predicts the Part-4 ``composite_reward`` directly from MERT features
of (prev window, candidate).

This pass is a SCAFFOLD ONLY (no GPU / no training run):

- :func:`reward.composite_reward` / :class:`reward.RewardSpec` — the pure reward
  math (single source of truth, shared with the Part-4 selector / ``mert_score``).
- :class:`model.MERTRewardModel` — tiny MLP over MERT features (CPU forward-shape
  tested).
- :func:`dataset.build_reward_dataset` — dataset builder stub consuming Part-4
  ``bestof_log`` entries.
- :func:`train.main` / :func:`train.train` — train() entrypoint stub with the loss.
"""

from sa3_control.mert_reward.reward import (
    RewardSpec,
    composite_reward,
    composite_reward_spec,
)

__all__ = [
    "RewardSpec",
    "composite_reward",
    "composite_reward_spec",
]

__version__ = "0.0.1"
