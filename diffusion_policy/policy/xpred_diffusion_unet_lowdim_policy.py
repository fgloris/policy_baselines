from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class XPredDiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    """JiT-style x-prediction diffusion / ODE policy for low-dimensional actions.

    This policy intentionally mirrors DiffusionUnetLowdimPolicy's public API so it
    can reuse TrainDiffusionUnetLowdimWorkspace, the same dataset, the same
    normalizer, and the same ConditionalUnet1D backbone.

    Difference from the original epsilon-prediction diffusion policy:
        - the network directly predicts the clean trajectory x_1 (x-prediction);
        - training noising path is z_t = t * x_1 + (1 - t) * eps;
        - by default, the loss is JiT's velocity-space loss:
              v      = (x_1      - z_t) / max(1 - t, denom_clip)
              v_pred = (x_pred   - z_t) / max(1 - t, denom_clip)
          while the direct network output remains x_pred;
        - sampling starts from Gaussian noise and integrates the ODE velocity
          induced by x_pred.

    If you want the most literal "DDPM sample-prediction" baseline, set
    loss_type='x' and solver='midpoint_euler'.  The default loss_type='velocity'
    follows the JiT paper more closely.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 50,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        source_noise_std: float = 1.0,
        time_embed_scale: float = 100.0,
        t_min: float = 1.0e-4,
        t_max: float = 0.9999,
        time_sampler: str = "logit_normal",
        time_mu: float = -0.8,
        time_sigma: float = 0.8,
        denom_clip: float = 0.05,
        loss_type: str = "velocity",
        solver: str = "heun",
        clamp_sample: bool = True,
        clamp_value: float = 1.2,
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        assert num_inference_steps >= 1
        assert source_noise_std > 0
        assert 0.0 <= t_min < t_max <= 1.0
        assert time_sampler in ["uniform", "logit_normal"]
        assert denom_clip > 0
        assert loss_type in ["velocity", "x"]
        assert solver in ["euler", "midpoint_euler", "heun"]
        assert clamp_value > 0

        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.source_noise_std = source_noise_std
        self.time_embed_scale = time_embed_scale
        self.t_min = t_min
        self.t_max = t_max
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.denom_clip = denom_clip
        self.loss_type = loss_type
        self.solver = solver
        self.clamp_sample = clamp_sample
        self.clamp_value = clamp_value
        self.kwargs = kwargs

    # ========= helpers ==========
    def _time_for_model(self, t: torch.Tensor) -> torch.Tensor:
        # ConditionalUnet1D was written for discrete diffusion-step embeddings.
        # Scaling continuous t back to roughly [0, 100] keeps the embedding scale
        # comparable to the original DDPM lowdim baseline.
        return t * self.time_embed_scale

    def _make_time(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if self.time_sampler == "logit_normal":
            t = torch.randn(batch_size, device=device, dtype=dtype)
            t = torch.sigmoid(self.time_mu + self.time_sigma * t)
        else:
            t = torch.rand(batch_size, device=device, dtype=dtype)

        if self.t_min != 0.0 or self.t_max != 1.0:
            t = t.clamp(self.t_min, self.t_max)
        return t

    def _sample_source(
        self,
        shape,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ):
        return self.source_noise_std * torch.randn(
            size=shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def _denom(self, t: torch.Tensor, ndim: int):
        denom = (1.0 - t).clamp_min(self.denom_clip)
        return denom.reshape(t.shape[0], *([1] * (ndim - 1)))

    def _velocity_from_xpred(self, z: torch.Tensor, t: torch.Tensor, x_pred: torch.Tensor):
        return (x_pred - z) / self._denom(t, z.ndim)

    def _post_step(self, z: torch.Tensor, condition_data: torch.Tensor, condition_mask: torch.Tensor):
        z = torch.where(condition_mask, condition_data, z)
        if self.clamp_sample:
            z = z.clamp(-self.clamp_value, self.clamp_value)
        return z

    def _build_cond_and_trajectory(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        local_cond = None
        global_cond = None
        trajectory = action

        if self.obs_as_local_cond:
            assert action is not None
            local_cond = obs.clone()
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if action is not None and self.pred_action_steps_only:
                start = self.n_obs_steps
                if self.oa_step_convention:
                    start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            assert action is not None
            trajectory = torch.cat([action, obs], dim=-1)

        return trajectory, local_cond, global_cond

    def _make_inference_tensors(self, nobs: torch.Tensor):
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None

        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(B, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        return cond_data, cond_mask, local_cond, global_cond

    # ========= inference ==========
    def _predict_velocity(self, z, t, local_cond=None, global_cond=None):
        x_pred = self.model(
            z,
            self._time_for_model(t),
            local_cond=local_cond,
            global_cond=global_cond,
        )
        v_pred = self._velocity_from_xpred(z, t, x_pred)
        return v_pred, x_pred

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond=None,
        global_cond=None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        del kwargs
        z = self._sample_source(
            condition_data.shape,
            device=condition_data.device,
            dtype=condition_data.dtype,
            generator=generator,
        )
        z = torch.where(condition_mask, condition_data, z)

        B = z.shape[0]
        dt = 1.0 / float(self.num_inference_steps)

        for i in range(self.num_inference_steps):
            if self.solver == "midpoint_euler":
                t_value = (i + 0.5) * dt
                t = torch.full((B,), t_value, device=z.device, dtype=z.dtype)
                z = torch.where(condition_mask, condition_data, z)
                v, _ = self._predict_velocity(z, t, local_cond, global_cond)
                z = z + dt * v
            elif self.solver == "euler":
                t_value = i * dt
                t = torch.full((B,), t_value, device=z.device, dtype=z.dtype)
                z = torch.where(condition_mask, condition_data, z)
                v, _ = self._predict_velocity(z, t, local_cond, global_cond)
                z = z + dt * v
            else:  # heun
                t0_value = i * dt
                t1_value = min((i + 1) * dt, 1.0)
                t0 = torch.full((B,), t0_value, device=z.device, dtype=z.dtype)
                t1 = torch.full((B,), t1_value, device=z.device, dtype=z.dtype)

                z = torch.where(condition_mask, condition_data, z)
                v0, _ = self._predict_velocity(z, t0, local_cond, global_cond)
                z_euler = z + dt * v0
                z_euler = self._post_step(z_euler, condition_data, condition_mask)
                v1, _ = self._predict_velocity(z_euler, t1, local_cond, global_cond)
                z = z + 0.5 * dt * (v0 + v1)

            z = self._post_step(z, condition_data, condition_mask)

        z = torch.where(condition_mask, condition_data, z)
        return z

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        assert Do == self.obs_dim
        To = self.n_obs_steps
        Da = self.action_dim

        cond_data, cond_mask, local_cond, global_cond = self._make_inference_tensors(nobs)
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
            start = None
            end = None
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            result["obs_pred"] = obs_pred
            if start is not None:
                result["action_obs_pred"] = obs_pred[:, start:end]
        return result

    # ========= training ==========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        x1, local_cond, global_cond = self._build_cond_and_trajectory(obs, action)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(x1, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(x1.shape)
        loss_mask = ~condition_mask

        eps = self._sample_source(x1.shape, device=x1.device, dtype=x1.dtype)
        B = x1.shape[0]
        t = self._make_time(B, device=x1.device, dtype=x1.dtype)
        t_view = t.reshape(B, *([1] * (x1.ndim - 1)))

        z = t_view * x1 + (1.0 - t_view) * eps
        z = torch.where(condition_mask, x1, z)

        x_pred = self.model(
            z,
            self._time_for_model(t),
            local_cond=local_cond,
            global_cond=global_cond,
        )

        if self.loss_type == "velocity":
            target = (x1 - z) / self._denom(t, x1.ndim)
            pred = (x_pred - z) / self._denom(t, x1.ndim)
        else:
            target = x1
            pred = x_pred

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
