from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class DirectionMagMLPLowdimPolicy(BaseLowdimPolicy):
    """
    MLP baseline for Push-T lowdim.

    The policy predicts an action chunk as direction + magnitude in normalized
    action space:
      - direction: 32-way categorical bin per action step
      - magnitude: continuous non-negative step length per action step

    The first delta is from the current agent position to the first action;
    later deltas are between consecutive action targets. Current agent position
    is read from the last two dims of the raw lowdim observation and normalized
    with the action normalizer, not the obs normalizer.
    """

    def __init__(
        self,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_dir_bins: int = 32,
        hidden_dim: int = 512,
        depth: int = 4,
        activation: str = "mish",
        layer_norm: bool = True,
        dir_loss_weight: float = 1.0,
        mag_loss_weight: float = 1.0,
        traj_loss_weight: float = 0.5,
        dir_eps: float = 1.0e-2,
        mag_scale: float = 5.0e-2,
        mag_head_init_bias: float = -2.0,
        oa_step_convention: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert action_dim == 2, "Direction/magnitude baseline currently assumes 2D actions."
        assert num_dir_bins > 1
        assert depth >= 1
        assert hidden_dim > 0

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_dir_bins = num_dir_bins
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dir_loss_weight = dir_loss_weight
        self.mag_loss_weight = mag_loss_weight
        self.traj_loss_weight = traj_loss_weight
        self.dir_eps = dir_eps
        self.mag_scale = mag_scale
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        self.normalizer = LinearNormalizer()

        if activation.lower() == "mish":
            act_cls = nn.Mish
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        elif activation.lower() == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        input_dim = obs_dim * n_obs_steps
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.dir_head = nn.Linear(hidden_dim, n_action_steps * num_dir_bins)
        self.mag_head = nn.Linear(hidden_dim, n_action_steps)

        # Start with small positive magnitudes instead of large random moves.
        nn.init.constant_(self.mag_head.bias, mag_head_init_bias)

        unit_dirs = self._build_unit_dirs(num_dir_bins)
        self.register_buffer("unit_dirs", unit_dirs, persistent=False)
        self.last_loss_info = dict()

    @staticmethod
    def _build_unit_dirs(num_bins: int) -> torch.Tensor:
        # Bin centers from [-pi, pi). Class 0 is centered at -pi + half_bin.
        half_bin = math.pi / num_bins
        angles = torch.linspace(
            -math.pi + half_bin,
            math.pi - half_bin,
            num_bins,
            dtype=torch.float32,
        )
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _obs_cond(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, Do], already normalized.
        return obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)

    def _action_start_end(self) -> Tuple[int, int]:
        start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
        end = start + self.n_action_steps
        return start, end

    def _current_agent_pos_action_normalized(self, raw_obs: torch.Tensor) -> torch.Tensor:
        # raw_obs: [B, T, Do]. Last two dims are raw agent_pos in Push-T lowdim.
        raw_agent_pos = raw_obs[:, self.n_obs_steps - 1, -2:]
        return self.normalizer["action"].normalize(raw_agent_pos)

    def _forward_heads(self, nobs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(self._obs_cond(nobs))
        B = nobs.shape[0]
        dir_logits = self.dir_head(h).reshape(B, self.n_action_steps, self.num_dir_bins)
        # Predict log(1 + r / mag_scale), constrained non-negative.
        pred_log_mag = F.softplus(self.mag_head(h).reshape(B, self.n_action_steps))
        pred_mag = self.mag_scale * torch.expm1(pred_log_mag).clamp_min(0.0)
        return dir_logits, pred_log_mag, pred_mag

    def _target_delta(self, naction_target: torch.Tensor, n_agent_pos: torch.Tensor) -> torch.Tensor:
        # naction_target: [B, Ta, 2], normalized action targets.
        prev = torch.cat([n_agent_pos[:, None, :], naction_target[:, :-1, :]], dim=1)
        return naction_target - prev

    def _delta_to_dir_mag(self, ndelta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.linalg.norm(ndelta, dim=-1)
        theta = torch.atan2(ndelta[..., 1], ndelta[..., 0])
        # Map theta in [-pi, pi] to class in [0, K-1].
        cls = torch.floor((theta + math.pi) / (2.0 * math.pi) * self.num_dir_bins).long()
        cls = torch.clamp(cls, min=0, max=self.num_dir_bins - 1)
        return cls, mag

    def _cumsum_actions(self, ndelta: torch.Tensor, n_agent_pos: torch.Tensor) -> torch.Tensor:
        return n_agent_pos[:, None, :] + torch.cumsum(ndelta, dim=1)

    # ========= inference ==========
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict

        raw_obs = obs_dict["obs"]
        nobs = self.normalizer["obs"].normalize(raw_obs)
        B, _, Do = nobs.shape
        assert Do == self.obs_dim

        dir_logits, pred_log_mag, pred_mag = self._forward_heads(nobs)
        dir_idx = torch.argmax(dir_logits, dim=-1)
        unit = self.unit_dirs.to(device=dir_logits.device, dtype=dir_logits.dtype)[dir_idx]
        ndelta = pred_mag[..., None] * unit

        n_agent_pos = self._current_agent_pos_action_normalized(raw_obs)
        naction = self._cumsum_actions(ndelta, n_agent_pos)
        action = self.normalizer["action"].unnormalize(naction)

        return {
            "action": action,
            "action_pred": action,
            "dir_logits": dir_logits,
            "mag_log_pred": pred_log_mag,
            "mag_pred": pred_mag,
        }

    # ========= training ==========
    def compute_loss(self, batch):
        assert "valid_mask" not in batch

        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch["obs"]
        naction = nbatch["action"]
        raw_obs = batch["obs"]

        start, end = self._action_start_end()
        naction_target = naction[:, start:end, :]
        n_agent_pos = self._current_agent_pos_action_normalized(raw_obs)

        ndelta_gt = self._target_delta(naction_target, n_agent_pos)
        dir_target, mag_target = self._delta_to_dir_mag(ndelta_gt)
        valid_dir = mag_target > self.dir_eps

        dir_logits, pred_log_mag, pred_mag = self._forward_heads(nobs)

        # 1) Direction classification loss. Ignore near-zero displacement frames.
        ce = F.cross_entropy(
            dir_logits.reshape(-1, self.num_dir_bins),
            dir_target.reshape(-1),
            reduction="none",
        ).reshape_as(mag_target)
        valid_float = valid_dir.float()
        dir_loss = (ce * valid_float).sum() / valid_float.sum().clamp_min(1.0)

        # 2) Magnitude regression loss in log space.
        target_log_mag = torch.log1p(mag_target / self.mag_scale)
        mag_loss = F.smooth_l1_loss(pred_log_mag, target_log_mag)

        # 3) Trajectory reconstruction loss. Use GT direction bins so this loss
        # does not push direction logits toward soft averaged directions.
        unit_gt = self.unit_dirs.to(device=dir_logits.device, dtype=dir_logits.dtype)[dir_target]
        ndelta_pred_for_traj = pred_mag[..., None] * unit_gt
        naction_pred_for_traj = self._cumsum_actions(ndelta_pred_for_traj, n_agent_pos)
        traj_loss = F.smooth_l1_loss(naction_pred_for_traj, naction_target)

        loss = (
            self.dir_loss_weight * dir_loss
            + self.mag_loss_weight * mag_loss
            + self.traj_loss_weight * traj_loss
        )

        with torch.no_grad():
            dir_acc = (
                (dir_logits.argmax(dim=-1) == dir_target).float() * valid_float
            ).sum() / valid_float.sum().clamp_min(1.0)
            self.last_loss_info = {
                "loss": float(loss.detach().cpu()),
                "dir_loss": float(dir_loss.detach().cpu()),
                "mag_loss": float(mag_loss.detach().cpu()),
                "traj_loss": float(traj_loss.detach().cpu()),
                "dir_acc": float(dir_acc.detach().cpu()),
                "mean_mag_gt": float(mag_target.detach().mean().cpu()),
                "mean_mag_pred": float(pred_mag.detach().mean().cpu()),
                "valid_dir_ratio": float(valid_float.detach().mean().cpu()),
            }

        return {
            "loss": loss,
            "dir_loss": dir_loss.detach(),
            "mag_loss": mag_loss.detach(),
            "traj_loss": traj_loss.detach(),
            "dir_acc": dir_acc.detach(),
            "mean_mag_gt": mag_target.detach().mean(),
            "mean_mag_pred": pred_mag.detach().mean(),
            "valid_dir_ratio": valid_float.detach().mean(),
        }
