from typing import Dict, Tuple, TYPE_CHECKING, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
if TYPE_CHECKING:
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
else:
    MultiImageObsEncoder = Any
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DirectionMagMLPImagePolicy(BaseImagePolicy):
    """
    Image Push-T MLP baseline aligned with DiffusionUnetImagePolicy.

    It intentionally reuses the same image observation path as
    DiffusionUnetImagePolicy:
      obs_dict -> LinearNormalizer -> MultiImageObsEncoder over the first
      n_obs_steps -> flattened global feature.

    The action head is the same direction/magnitude/bias MLP design used by
    DirectionMagMLPLowdimPolicy. It predicts pred_action_steps absolute action
    targets by autoregressively summing normalized deltas from the current
    agent_pos.

    For every predicted action step and every direction bin it outputs:
      - a direction logit
      - a non-negative magnitude
      - a bounded angular bias around the bin center

    Inference selects argmax(logit), then reconstructs the continuous delta:
        theta = bin_center[k] + bias[k]
        delta = magnitude[k] * [cos(theta), sin(theta)]

    Training uses a circular Gaussian centered at the continuous GT direction:
      - direction CE uses the Gaussian normalized to sum=1
      - magnitude/bias/vector regression uses the Gaussian peak-normalized to 1
      - (1 - weight) regularizes far-away experts toward small magnitudes

    The first delta is from current agent_pos to the first action target; later
    deltas are between consecutive action targets. agent_pos is normalized with
    the action normalizer so it shares coordinates with normalized actions.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        pred_action_steps: int = None,
        num_dir_bins: int = 32,
        hidden_dim: int = 512,
        depth: int = 4,
        activation: str = "relu",
        layer_norm: bool = True,
        dropout: float = 0.0,
        dir_loss_weight: float = 1.0,
        mag_loss_weight: float = 1.0,
        bias_loss_weight: float = 0.3,
        traj_loss_weight: float = 0.1,
        mag_all_reg_weight: float = 5.0e-2,
        gaussian_sigma_bins: float = 0.8,
        bias_range_bins: float = 0.8,
        dir_eps: float = 1.0e-2,
        mag_scale: float = 5.0e-2,
        mag_head_init_bias: float = -2.0,
        oa_step_convention: bool = True,
        obs_as_global_cond: bool = True,
        # Deprecated; kept so older hydra overrides do not crash.
        dir_neighbor_smoothing: float = None,
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond, "Image MLP baseline only supports global obs conditioning."

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        assert action_dim == 2, "Direction/magnitude/bias baseline currently assumes 2D actions."
        assert "agent_pos" in shape_meta["obs"], "Push-T image baseline needs obs['agent_pos']."
        assert tuple(shape_meta["obs"]["agent_pos"]["shape"]) == (2,)
        assert num_dir_bins > 1
        assert n_action_steps >= 1
        if pred_action_steps is None:
            pred_action_steps = n_action_steps
        assert pred_action_steps >= n_action_steps, "pred_action_steps should be >= n_action_steps."
        assert depth >= 1
        assert hidden_dim > 0
        assert 0.0 <= dropout < 1.0
        assert mag_all_reg_weight >= 0.0
        assert gaussian_sigma_bins > 0.0
        assert bias_range_bins > 0.0

        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.pred_action_steps = pred_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_dir_bins = num_dir_bins
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.dir_loss_weight = dir_loss_weight
        self.mag_loss_weight = mag_loss_weight
        self.bias_loss_weight = bias_loss_weight
        self.traj_loss_weight = traj_loss_weight
        self.mag_all_reg_weight = mag_all_reg_weight
        self.gaussian_sigma_bins = gaussian_sigma_bins
        self.bias_range_bins = bias_range_bins
        self.dir_eps = dir_eps
        self.mag_scale = mag_scale
        self.oa_step_convention = oa_step_convention
        self.obs_as_global_cond = obs_as_global_cond
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

        input_dim = obs_feature_dim * n_obs_steps
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.dir_head = nn.Linear(hidden_dim, self.pred_action_steps * num_dir_bins)
        self.mag_head = nn.Linear(hidden_dim, self.pred_action_steps * num_dir_bins)
        self.bias_head = nn.Linear(hidden_dim, self.pred_action_steps * num_dir_bins)

        nn.init.constant_(self.mag_head.bias, mag_head_init_bias)
        nn.init.zeros_(self.bias_head.bias)

        angle_centers = self._build_angle_centers(num_dir_bins)
        unit_dirs = torch.stack([torch.cos(angle_centers), torch.sin(angle_centers)], dim=-1)
        self.register_buffer("angle_centers", angle_centers, persistent=False)
        self.register_buffer("unit_dirs", unit_dirs, persistent=False)
        self.last_loss_info = dict()

    @staticmethod
    def _build_angle_centers(num_bins: int) -> torch.Tensor:
        half_bin = math.pi / num_bins
        return torch.linspace(
            -math.pi + half_bin,
            math.pi - half_bin,
            num_bins,
            dtype=torch.float32,
        )

    @staticmethod
    def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    @property
    def bin_width(self) -> float:
        return 2.0 * math.pi / self.num_dir_bins

    @property
    def bias_angle_range(self) -> float:
        return self.bias_range_bins * self.bin_width

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _action_start_end(self) -> Tuple[int, int]:
        start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
        end = start + self.pred_action_steps
        return start, end

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode first n_obs_steps exactly like DiffusionUnetImagePolicy global cond."""
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _current_agent_pos_action_normalized(self, raw_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw_agent_pos = raw_obs["agent_pos"][:, self.n_obs_steps - 1, :]
        return self.normalizer["action"].normalize(raw_agent_pos)

    def _forward_heads(self, nobs: Dict[str, torch.Tensor]):
        h = self.trunk(self._encode_obs(nobs))
        B = h.shape[0]
        dir_logits = self.dir_head(h).reshape(B, self.pred_action_steps, self.num_dir_bins)

        pred_log_mag = F.softplus(
            self.mag_head(h).reshape(B, self.pred_action_steps, self.num_dir_bins)
        )
        pred_mag = self.mag_scale * torch.expm1(pred_log_mag).clamp_min(0.0)

        raw_bias = self.bias_head(h).reshape(B, self.pred_action_steps, self.num_dir_bins)
        pred_bias_norm = torch.tanh(raw_bias)
        pred_bias_angle = pred_bias_norm * self.bias_angle_range
        return dir_logits, pred_log_mag, pred_mag, pred_bias_norm, pred_bias_angle

    @staticmethod
    def _gather_by_dir(values: torch.Tensor, dir_idx: torch.Tensor) -> torch.Tensor:
        return values.gather(dim=-1, index=dir_idx.unsqueeze(-1)).squeeze(-1)

    def _target_delta(self, naction_target: torch.Tensor, n_agent_pos: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([n_agent_pos[:, None, :], naction_target[:, :-1, :]], dim=1)
        return naction_target - prev

    def _delta_to_angle_mag(self, ndelta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.linalg.norm(ndelta, dim=-1)
        theta = torch.atan2(ndelta[..., 1], ndelta[..., 0])
        return theta, mag

    def _nearest_dir_idx(self, theta: torch.Tensor) -> torch.Tensor:
        centers = self.angle_centers.to(device=theta.device, dtype=theta.dtype)
        diff = self._angle_diff(theta.unsqueeze(-1), centers)
        return diff.abs().argmin(dim=-1)

    def _gaussian_dir_weights(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centers = self.angle_centers.to(device=theta.device, dtype=theta.dtype)
        delta = self._angle_diff(theta.unsqueeze(-1), centers)
        sigma = self.gaussian_sigma_bins * self.bin_width
        weights = torch.exp(-0.5 * (delta / sigma).square())
        weights = weights / weights.amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return weights, delta

    def _cumsum_actions(self, ndelta: torch.Tensor, n_agent_pos: torch.Tensor) -> torch.Tensor:
        return n_agent_pos[:, None, :] + torch.cumsum(ndelta, dim=1)

    # ========= inference ==========
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        assert "image" in obs_dict and "agent_pos" in obs_dict

        raw_obs = obs_dict
        nobs = self.normalizer.normalize(obs_dict)

        dir_logits, pred_log_mag_all, pred_mag_all, pred_bias_norm_all, pred_bias_angle_all = self._forward_heads(nobs)
        dir_idx = torch.argmax(dir_logits, dim=-1)
        pred_log_mag = self._gather_by_dir(pred_log_mag_all, dir_idx)
        pred_mag = self._gather_by_dir(pred_mag_all, dir_idx)
        pred_bias_norm = self._gather_by_dir(pred_bias_norm_all, dir_idx)
        pred_bias_angle = self._gather_by_dir(pred_bias_angle_all, dir_idx)

        centers = self.angle_centers.to(device=dir_logits.device, dtype=dir_logits.dtype)
        theta = centers[dir_idx] + pred_bias_angle
        unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        ndelta = pred_mag[..., None] * unit

        n_agent_pos = self._current_agent_pos_action_normalized(raw_obs)
        naction_pred = self._cumsum_actions(ndelta, n_agent_pos)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action = action_pred[:, : self.n_action_steps, :]

        return {
            "action": action,
            "action_pred": action_pred,
            "dir_logits": dir_logits,
            "dir_idx": dir_idx,
            "mag_log_pred": pred_log_mag,
            "mag_pred": pred_mag,
            "mag_log_pred_all": pred_log_mag_all,
            "mag_pred_all": pred_mag_all,
            "bias_norm_pred": pred_bias_norm,
            "bias_angle_pred": pred_bias_angle,
            "bias_norm_pred_all": pred_bias_norm_all,
            "bias_angle_pred_all": pred_bias_angle_all,
        }

    # ========= training ==========
    def compute_loss(self, batch):
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])
        raw_obs = batch["obs"]

        start, end = self._action_start_end()
        naction_target = naction[:, start:end, :]
        n_agent_pos = self._current_agent_pos_action_normalized(raw_obs)

        ndelta_gt = self._target_delta(naction_target, n_agent_pos)
        theta_target, mag_target = self._delta_to_angle_mag(ndelta_gt)
        dir_target = self._nearest_dir_idx(theta_target)
        valid_dir = mag_target > self.dir_eps
        valid_float = valid_dir.float()

        dir_logits, pred_log_mag_all, pred_mag_all, pred_bias_norm_all, pred_bias_angle_all = self._forward_heads(nobs)

        # 1) Direction classification loss: circular Gaussian soft CE.
        gauss_w, angle_delta = self._gaussian_dir_weights(theta_target)
        target_prob = gauss_w / gauss_w.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        log_prob = F.log_softmax(dir_logits, dim=-1)
        dir_ce = -(target_prob * log_prob).sum(dim=-1)
        dir_loss = (dir_ce * valid_float).sum() / valid_float.sum().clamp_min(1.0)

        # 2) Gaussian-weighted magnitude regression in log space.
        target_log_mag = torch.log1p(mag_target / self.mag_scale)
        mag_loss_all_bins = F.smooth_l1_loss(
            pred_log_mag_all,
            target_log_mag.unsqueeze(-1).expand_as(pred_log_mag_all),
            reduction="none",
        )
        reg_w = gauss_w * valid_float.unsqueeze(-1)
        mag_loss = (reg_w * mag_loss_all_bins).sum() / reg_w.sum().clamp_min(1.0)

        # 3) Gaussian-weighted angular bias regression.
        bias_target_norm = angle_delta / self.bias_angle_range
        bias_expr_mask = (bias_target_norm.abs() <= 1.0).float()
        bias_w = reg_w * bias_expr_mask
        bias_loss_all_bins = F.smooth_l1_loss(
            pred_bias_norm_all,
            bias_target_norm.clamp(min=-1.0, max=1.0),
            reduction="none",
        )
        bias_loss = (bias_w * bias_loss_all_bins).sum() / bias_w.sum().clamp_min(1.0)

        # 4) Far-away magnitude prior.
        far_w = (1.0 - gauss_w) * valid_float.unsqueeze(-1) + (1.0 - valid_float).unsqueeze(-1)
        mag_all_reg_loss = (far_w * pred_log_mag_all.square()).sum() / far_w.sum().clamp_min(1.0)

        # 5) Gaussian-weighted vector reconstruction in normalized delta space.
        centers = self.angle_centers.to(device=dir_logits.device, dtype=dir_logits.dtype)
        theta_pred_all = centers.view(1, 1, -1) + pred_bias_angle_all
        unit_pred_all = torch.stack([torch.cos(theta_pred_all), torch.sin(theta_pred_all)], dim=-1)
        ndelta_pred_all = pred_mag_all.unsqueeze(-1) * unit_pred_all
        traj_loss_all_bins = F.smooth_l1_loss(
            ndelta_pred_all,
            ndelta_gt.unsqueeze(-2).expand_as(ndelta_pred_all),
            reduction="none",
        ).mean(dim=-1)
        traj_loss = (reg_w * traj_loss_all_bins).sum() / reg_w.sum().clamp_min(1.0)

        loss = (
            self.dir_loss_weight * dir_loss
            + self.mag_loss_weight * mag_loss
            + self.bias_loss_weight * bias_loss
            + self.traj_loss_weight * traj_loss
            + self.mag_all_reg_weight * mag_all_reg_loss
        )

        with torch.no_grad():
            dir_pred = dir_logits.argmax(dim=-1)
            pred_mag_pred_dir = self._gather_by_dir(pred_mag_all, dir_pred)
            pred_bias_pred_dir = self._gather_by_dir(pred_bias_norm_all, dir_pred)
            pred_mag_gt_dir = self._gather_by_dir(pred_mag_all, dir_target)
            pred_bias_gt_dir = self._gather_by_dir(pred_bias_norm_all, dir_target)

            dir_acc = ((dir_pred == dir_target).float() * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            circular_dist = torch.abs(dir_pred - dir_target)
            circular_dist = torch.minimum(circular_dist, self.num_dir_bins - circular_dist).float()
            dir_within1_acc = ((circular_dist <= 1).float() * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            dir_within2_acc = ((circular_dist <= 2).float() * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            dir_mean_bin_error = (circular_dist * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            dir_mean_angle_error_deg = dir_mean_bin_error * (360.0 / self.num_dir_bins)

            self.last_loss_info = {
                "loss": float(loss.detach().cpu()),
                "dir_loss": float(dir_loss.detach().cpu()),
                "mag_loss": float(mag_loss.detach().cpu()),
                "bias_loss": float(bias_loss.detach().cpu()),
                "traj_loss": float(traj_loss.detach().cpu()),
                "mag_all_reg_loss": float(mag_all_reg_loss.detach().cpu()),
                "dir_acc": float(dir_acc.detach().cpu()),
                "dir_within1_acc": float(dir_within1_acc.detach().cpu()),
                "dir_within2_acc": float(dir_within2_acc.detach().cpu()),
                "dir_mean_bin_error": float(dir_mean_bin_error.detach().cpu()),
                "dir_mean_angle_error_deg": float(dir_mean_angle_error_deg.detach().cpu()),
                "mean_mag_gt": float(mag_target.detach().mean().cpu()),
                "mean_mag_pred": float(pred_mag_pred_dir.detach().mean().cpu()),
                "mean_mag_pred_gt_dir": float(pred_mag_gt_dir.detach().mean().cpu()),
                "mean_mag_pred_all": float(pred_mag_all.detach().mean().cpu()),
                "mean_bias_pred": float(pred_bias_pred_dir.detach().mean().cpu()),
                "mean_bias_pred_gt_dir": float(pred_bias_gt_dir.detach().mean().cpu()),
                "mean_gaussian_weight": float(gauss_w.detach().mean().cpu()),
                "mean_reg_weight_sum": float(reg_w.detach().sum(dim=-1).mean().cpu()),
                "valid_dir_ratio": float(valid_float.detach().mean().cpu()),
            }

        return {
            "loss": loss,
            "dir_loss": dir_loss.detach(),
            "mag_loss": mag_loss.detach(),
            "bias_loss": bias_loss.detach(),
            "traj_loss": traj_loss.detach(),
            "mag_all_reg_loss": mag_all_reg_loss.detach(),
            "dir_acc": dir_acc.detach(),
            "dir_within1_acc": dir_within1_acc.detach(),
            "dir_within2_acc": dir_within2_acc.detach(),
            "dir_mean_bin_error": dir_mean_bin_error.detach(),
            "dir_mean_angle_error_deg": dir_mean_angle_error_deg.detach(),
            "mean_mag_gt": mag_target.detach().mean(),
            "mean_mag_pred": pred_mag_pred_dir.detach().mean(),
            "mean_mag_pred_gt_dir": pred_mag_gt_dir.detach().mean(),
            "mean_mag_pred_all": pred_mag_all.detach().mean(),
            "mean_bias_pred": pred_bias_pred_dir.detach().mean(),
            "mean_bias_pred_gt_dir": pred_bias_gt_dir.detach().mean(),
            "mean_gaussian_weight": gauss_w.detach().mean(),
            "mean_reg_weight_sum": reg_w.detach().sum(dim=-1).mean(),
            "valid_dir_ratio": valid_float.detach().mean(),
        }
