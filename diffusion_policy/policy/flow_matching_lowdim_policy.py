from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class FlowMatchingLowdimPolicy(BaseLowdimPolicy):
    """Conditional flow-matching policy for low-dimensional action chunks.

    This is the continuous CondOT / rectified-flow objective:
        x_t = (1 - t) * x_0 + t * x_1
        u_t = d x_t / d t = x_1 - x_0

    where x_1 is the normalized target action trajectory and x_0 is sampled from
    the same source distribution used at inference time.  The default source is
    N(0, source_noise_std^2 I).  The network predicts u_t.

    The class intentionally mirrors DiffusionUnetLowdimPolicy so it can reuse
    TrainDiffusionUnetLowdimWorkspace without changing train.py or the workspace.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 20,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        source_noise_std: float = 1.0,
        time_scale: float = 100.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        inference_mode: str = "midpoint_euler",
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        assert num_inference_steps >= 1
        assert source_noise_std > 0
        assert 0.0 <= t_min < t_max <= 1.0
        assert inference_mode in ["euler", "midpoint_euler"]

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
        self.time_scale = time_scale
        self.t_min = t_min
        self.t_max = t_max
        self.inference_mode = inference_mode
        self.kwargs = kwargs

    # ========= helpers ==========
    def _make_time(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        t = torch.rand(batch_size, device=device, dtype=dtype)
        if self.t_min != 0.0 or self.t_max != 1.0:
            t = self.t_min + (self.t_max - self.t_min) * t
        return t

    def _time_for_model(self, t: torch.Tensor) -> torch.Tensor:
        # ConditionalUnet1D uses sinusoidal embeddings.  Scaling t from [0, 1]
        # to roughly the old DDPM range makes the time embedding less tiny.
        return t * self.time_scale

    def _sample_source(self, shape, device, dtype, generator: Optional[torch.Generator] = None):
        return self.source_noise_std * torch.randn(
            size=shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def _build_cond_and_trajectory(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Build conditioning tensors, matching DiffusionUnetLowdimPolicy.

        During training, action is provided and returns the normalized target
        trajectory x_1.  During inference, action is None and only shapes / cond
        tensors are returned.
        """
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
        trajectory = self._sample_source(
            condition_data.shape,
            device=condition_data.device,
            dtype=condition_data.dtype,
            generator=generator,
        )

        dt = 1.0 / float(self.num_inference_steps)
        B = trajectory.shape[0]

        for i in range(self.num_inference_steps):
            if self.inference_mode == "midpoint_euler":
                t_value = (i + 0.5) * dt
            else:
                t_value = i * dt
            t = torch.full(
                (B,),
                t_value,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )

            # Enforce known observations for inpainting-style conditioning.
            trajectory = torch.where(condition_mask, condition_data, trajectory)
            velocity = self.model(
                trajectory,
                self._time_for_model(t),
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = trajectory + dt * velocity

        trajectory = torch.where(condition_mask, condition_data, trajectory)
        return trajectory

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

        trajectory, local_cond, global_cond = self._build_cond_and_trajectory(obs, action)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask

        x1 = trajectory
        x0 = self._sample_source(x1.shape, device=x1.device, dtype=x1.dtype)

        B = x1.shape[0]
        t = self._make_time(B, device=x1.device, dtype=x1.dtype)
        t_view = t.reshape(B, *([1] * (x1.ndim - 1)))

        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0

        # Known conditioning values should remain clean, just like the diffusion
        # policy's inpainting path.
        xt = torch.where(condition_mask, x1, xt)

        pred_velocity = self.model(
            xt,
            self._time_for_model(t),
            local_cond=local_cond,
            global_cond=global_cond,
        )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
