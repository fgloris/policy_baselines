"""GMM-classified conditional flow-matching policy for Push-T lowdim.

This policy implements the user's proposed baseline:
  1. classify the observation into a GMM action-chunk mode;
  2. sample a source action chunk from that GMM component;
  3. run conditional flow matching from the component source to the final action.

The GMM is fitted offline on *normalized* action chunks with
`scripts/fit_gmm_flow_gmm.py`.  At training time the ground-truth action chunk is
assigned to the most likely GMM component and the flow head is conditioned on the
GT component.  At inference time the classifier predicts the component.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

try:
    from hydra.utils import to_absolute_path
except Exception:  # pragma: no cover - hydra is present in normal training
    to_absolute_path = None

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (512, 512),
        activation: str = "mish",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation == "mish":
            act_cls = nn.Mish
        elif activation == "relu":
            act_cls = nn.ReLU
        elif activation == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        dims = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GMMFlowLowdimPolicy(BaseLowdimPolicy):
    """Low-dimensional GMM + flow matching policy.

    The flow head predicts the CondOT velocity
        x_t = (1 - t) x_0 + t x_1,   v_t = x_1 - x_0.

    `x_0` is sampled from the GMM component associated with the action chunk.
    In training, the component is the GT chunk's maximum-posterior GMM label;
    in inference, the component comes from the observation classifier.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        gmm_path: str,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_classes: int,
        class_embed_dim: int = 32,
        classifier_hidden_dims: Sequence[int] = (512, 512),
        classifier_activation: str = "mish",
        classifier_dropout: float = 0.0,
        num_inference_steps: int = 20,
        source_noise_scale: float = 1.0,
        source_std_min: float = 0.03,
        source_std_max: float = 0.5,
        covariance_jitter: float = 1e-6,
        time_scale: float = 100.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        inference_mode: str = "midpoint_euler",
        inference_class_mode: str = "argmax",
        class_temperature: float = 1.0,
        source_sample_mode: str = "sample",
        oa_step_convention: bool = True,
        ce_weight: float = 1.0,
        flow_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        assert num_inference_steps >= 1
        assert 0.0 <= t_min < t_max <= 1.0
        assert source_noise_scale >= 0.0
        assert 0.0 <= source_std_min <= source_std_max
        assert inference_mode in ["euler", "midpoint_euler"]
        assert inference_class_mode in ["argmax", "sample"]
        assert class_temperature > 0.0
        assert source_sample_mode in ["sample", "mean"]
        assert ce_weight >= 0.0 and flow_weight > 0.0

        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.num_inference_steps = num_inference_steps
        self.source_noise_scale = source_noise_scale
        self.source_std_min = source_std_min
        self.source_std_max = source_std_max
        self.covariance_jitter = covariance_jitter
        self.time_scale = time_scale
        self.t_min = t_min
        self.t_max = t_max
        self.inference_mode = inference_mode
        self.inference_class_mode = inference_class_mode
        self.class_temperature = class_temperature
        self.source_sample_mode = source_sample_mode
        self.oa_step_convention = oa_step_convention
        self.ce_weight = ce_weight
        self.flow_weight = flow_weight
        self.kwargs = kwargs

        self.classifier = MLPClassifier(
            input_dim=obs_dim * n_obs_steps,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            activation=classifier_activation,
            dropout=classifier_dropout,
        )
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)

        self._load_gmm(gmm_path)

    # ---------------------------------------------------------------------
    # GMM utilities
    # ---------------------------------------------------------------------
    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        if to_absolute_path is not None:
            return to_absolute_path(path)
        return os.path.abspath(path)

    def _load_gmm(self, gmm_path: str):
        gmm_path = self._resolve_path(gmm_path)
        if not os.path.exists(gmm_path):
            raise FileNotFoundError(
                f"GMM file not found: {gmm_path}\n"
                "Please fit it first, for example:\n"
                "  python scripts/fit_gmm_flow_gmm.py "
                "--n-components 16 --covariance-type diag "
                "--output data/pusht/action_gmm_K16_diag.npz"
            )

        data = np.load(gmm_path, allow_pickle=True)
        means = np.asarray(data["means"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        covariance_type = str(data["covariance_type"].item() if data["covariance_type"].shape == () else data["covariance_type"])
        covariances = np.asarray(data["covariances"], dtype=np.float32)

        expected_dim = self.n_action_steps * self.action_dim
        if means.ndim != 2:
            raise ValueError(f"GMM means should be [K, D], got {means.shape}")
        if means.shape[0] != self.num_classes:
            raise ValueError(
                f"Config num_classes={self.num_classes}, but GMM has {means.shape[0]} components."
            )
        if means.shape[1] != expected_dim:
            raise ValueError(
                f"GMM was fitted on D={means.shape[1]}, but policy expects "
                f"n_action_steps*action_dim={self.n_action_steps}*{self.action_dim}={expected_dim}."
            )
        if covariance_type not in ["diag", "full"]:
            raise ValueError(
                f"Only diag/full GMM covariance is supported by this policy, got {covariance_type}."
            )

        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()

        self.gmm_path = gmm_path
        self.gmm_covariance_type = covariance_type
        self.flat_action_dim = expected_dim

        self.register_buffer("gmm_means", torch.from_numpy(means))
        self.register_buffer("gmm_log_weights", torch.log(torch.from_numpy(weights)))

        if covariance_type == "diag":
            if covariances.shape != means.shape:
                raise ValueError(
                    f"Diag covariances should be [K, D], got {covariances.shape}, means {means.shape}."
                )
            covariances = np.maximum(covariances, self.covariance_jitter).astype(np.float32)
            self.register_buffer("gmm_diag_vars", torch.from_numpy(covariances))
            self.register_buffer("gmm_log_diag_vars", torch.log(torch.from_numpy(covariances)))
        else:
            if covariances.shape != (means.shape[0], means.shape[1], means.shape[1]):
                raise ValueError(
                    f"Full covariances should be [K, D, D], got {covariances.shape}."
                )
            eye = np.eye(means.shape[1], dtype=np.float32)[None]
            covariances = covariances + self.covariance_jitter * eye
            inv_covariances = np.linalg.inv(covariances).astype(np.float32)
            sign, logdet = np.linalg.slogdet(covariances)
            if not np.all(sign > 0):
                raise ValueError("Full GMM covariance is not positive definite even after jitter.")
            cholesky = np.linalg.cholesky(covariances).astype(np.float32)
            self.register_buffer("gmm_inv_covariances", torch.from_numpy(inv_covariances))
            self.register_buffer("gmm_logdet_covariances", torch.from_numpy(logdet.astype(np.float32)))
            self.register_buffer("gmm_cholesky", torch.from_numpy(cholesky))

        # Helpful non-persistent metadata for quick inspection.
        if "bic" in data:
            self.gmm_bic = float(data["bic"])
        if "aic" in data:
            self.gmm_aic = float(data["aic"])

    @torch.no_grad()
    def _gmm_log_prob(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Return component-wise log probability, shape [B, K]."""
        diff = x_flat[:, None, :] - self.gmm_means[None, :, :]
        d = x_flat.shape[-1]
        if self.gmm_covariance_type == "diag":
            inv_vars = torch.reciprocal(self.gmm_diag_vars)
            maha = (diff.pow(2) * inv_vars[None, :, :]).sum(dim=-1)
            logdet = self.gmm_log_diag_vars.sum(dim=-1)
        else:
            maha = torch.einsum("bkd,kde,bke->bk", diff, self.gmm_inv_covariances, diff)
            logdet = self.gmm_logdet_covariances
        return self.gmm_log_weights[None, :] - 0.5 * (maha + logdet[None, :] + d * math.log(2.0 * math.pi))

    @torch.no_grad()
    def _gmm_labels(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)
        return self._gmm_log_prob(x_flat).argmax(dim=-1)

    def _sample_gmm_source(
        self,
        labels: torch.Tensor,
        shape_dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        means = self.gmm_means[labels]
        if self.source_sample_mode == "mean" or self.source_noise_scale == 0.0:
            x_flat = means
        elif self.gmm_covariance_type == "diag":
            vars_ = self.gmm_diag_vars[labels]
            std = torch.sqrt(vars_).clamp(min=self.source_std_min, max=self.source_std_max)
            eps = torch.randn(
                means.shape,
                device=means.device,
                dtype=means.dtype,
                generator=generator,
            )
            x_flat = means + self.source_noise_scale * std * eps
        else:
            chol = self.gmm_cholesky[labels]
            eps = torch.randn(
                means.shape,
                device=means.device,
                dtype=means.dtype,
                generator=generator,
            )
            x_flat = means + self.source_noise_scale * torch.bmm(chol, eps.unsqueeze(-1)).squeeze(-1)

        return x_flat.to(dtype=shape_dtype).reshape(labels.shape[0], self.n_action_steps, self.action_dim)

    # ---------------------------------------------------------------------
    # Conditioning and time utilities
    # ---------------------------------------------------------------------
    def _action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
        end = start + self.n_action_steps
        return action[:, start:end]

    def _obs_cond(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, : self.n_obs_steps].reshape(obs.shape[0], -1)

    def _time_for_model(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.time_scale

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.rand(batch_size, device=device, dtype=dtype)
        if self.t_min != 0.0 or self.t_max != 1.0:
            t = self.t_min + (self.t_max - self.t_min) * t
        return t

    def _global_cond(self, obs_cond: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs_cond, self.class_embedding(labels)], dim=-1)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def conditional_sample(
        self,
        labels: torch.Tensor,
        global_cond: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        trajectory = self._sample_gmm_source(labels, shape_dtype=self.dtype, generator=generator)
        dt = 1.0 / float(self.num_inference_steps)
        bsz = trajectory.shape[0]

        for i in range(self.num_inference_steps):
            if self.inference_mode == "midpoint_euler":
                t_value = (i + 0.5) * dt
            else:
                t_value = i * dt
            t = torch.full(
                (bsz,),
                t_value,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            velocity = self.model(
                trajectory,
                self._time_for_model(t),
                local_cond=None,
                global_cond=global_cond,
            )
            trajectory = trajectory + dt * velocity
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        assert nobs.shape[-1] == self.obs_dim
        obs_cond = self._obs_cond(nobs)

        logits = self.classifier(obs_cond)
        if self.inference_class_mode == "sample":
            probs = F.softmax(logits / self.class_temperature, dim=-1)
            labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            labels = logits.argmax(dim=-1)

        global_cond = self._global_cond(obs_cond, labels)
        naction_pred = self.conditional_sample(labels=labels, global_cond=global_cond)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        return {
            "action": action_pred,
            "action_pred": action_pred,
            "class_logits": logits,
            "class_pred": labels,
        }

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        obs_cond = self._obs_cond(obs)
        x1 = self._action_chunk(action)
        labels = self._gmm_labels(x1)

        logits = self.classifier(obs_cond)
        ce_loss = F.cross_entropy(logits, labels)

        x0 = self._sample_gmm_source(labels, shape_dtype=x1.dtype)
        bsz = x1.shape[0]
        t = self._sample_time(bsz, device=x1.device, dtype=x1.dtype)
        t_view = t.reshape(bsz, *([1] * (x1.ndim - 1)))
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0

        global_cond = self._global_cond(obs_cond, labels)
        pred_velocity = self.model(
            xt,
            self._time_for_model(t),
            local_cond=None,
            global_cond=global_cond,
        )

        flow_loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        flow_loss = reduce(flow_loss, "b ... -> b (...)", "mean").mean()
        loss = self.flow_weight * flow_loss + self.ce_weight * ce_loss

        # Useful for debugging in a REPL/checkpoint, even though the existing
        # workspace only logs the scalar returned by compute_loss().
        with torch.no_grad():
            self.last_loss_info = {
                "loss": float(loss.detach().cpu()),
                "flow_matching_loss": float(flow_loss.detach().cpu()),
                "cls_loss": float(ce_loss.detach().cpu()),
                "class_acc": float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu()),
            }
        return loss