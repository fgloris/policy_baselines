#!/usr/bin/env python3
"""
Check action-direction alignment between a Diffusion Policy checkpoint and a
DirectionMagMLP checkpoint on Push-T train/test dataset splits.

Supports both lowdim and image checkpoints. It does NOT rollout in the
environment. It loads the dataset split, feeds the same observation batch to
both policies, then compares:
  1) category consistency: MLP argmax direction bin vs discretized DP final action
  2) angle error: circular angle error between MLP final action and DP final action

Both metrics are computed in the MLP policy's normalized action coordinate space,
because DirectionMagMLP's direction bins are trained in normalized action space.

Examples:
python align_check.py \
  --dp data/outputs/.../checkpoints/latest.ckpt \
  --mlp data/outputs/.../checkpoints/latest.ckpt \
  --device cuda:0

python align_check.py \
  --diffusion-ckpt data/outputs/.../checkpoints/latest.ckpt \
  --mlp-ckpt data/outputs/.../checkpoints/latest.ckpt \
  --device cuda:0
"""

import argparse
import copy
import datetime as _datetime
import json
import math
import os
import pathlib
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


# Make the script runnable from the project root or by absolute path.
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in {"1", "true", "yes", "y"}:
        return True
    if x in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {x!r}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_checkpoint(path: pathlib.Path) -> Dict[str, Any]:
    """Load older diffusion_policy workspace checkpoints robustly."""
    with path.open("rb") as f:
        try:
            return torch.load(
                f,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
        except TypeError:
            f.seek(0)
            return torch.load(f, map_location="cpu", pickle_module=dill)


def load_workspace_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    use_ema: str = "auto",
):
    ckpt_path = pathlib.Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch_load_checkpoint(ckpt_path)
    if "cfg" not in payload:
        raise RuntimeError(f"{ckpt_path} does not look like a BaseWorkspace checkpoint: missing cfg")

    cfg = copy.deepcopy(payload["cfg"])
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)

    target = cfg.get("_target_", None)
    if target is None:
        raise RuntimeError(f"Checkpoint cfg has no _target_: {ckpt_path}")

    workspace_cls = hydra.utils.get_class(target)
    workspace = workspace_cls(cfg, output_dir=None)

    # Do not waste time restoring optimizer state for pure inference.
    workspace.load_payload(
        payload,
        exclude_keys=("optimizer",),
        include_keys=payload.get("pickles", {}).keys(),
    )

    model = getattr(workspace, "model", None)
    ema_model = getattr(workspace, "ema_model", None)
    if model is None:
        raise RuntimeError(f"Checkpoint workspace has no model: {ckpt_path}")

    if use_ema not in {"auto", "true", "false"}:
        raise ValueError("use_ema must be one of: auto, true, false")
    if use_ema == "true":
        policy = ema_model
        used_ema = True
        if policy is None:
            raise RuntimeError(f"--use-ema true was requested, but checkpoint has no ema_model: {ckpt_path}")
    elif use_ema == "false":
        policy = model
        used_ema = False
    else:
        cfg_use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
        used_ema = cfg_use_ema and ema_model is not None
        policy = ema_model if used_ema else model

    policy.to(device)
    policy.eval()
    return workspace, policy, ckpt_path, used_ema


def dataset_kind_from_batch(batch: Dict[str, Any]) -> str:
    obs = batch.get("obs", None)
    if isinstance(obs, torch.Tensor):
        return "lowdim"
    if isinstance(obs, dict):
        return "image"
    return f"unknown:{type(obs)}"


def make_policy_obs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert a dataset batch to the observation object expected by policy.predict_action.

    Lowdim policies expect {"obs": tensor}. Image policies expect the obs dict
    directly, e.g. {"image": ..., "agent_pos": ...}.
    """
    obs = batch["obs"]
    if isinstance(obs, torch.Tensor):
        return {"obs": obs}
    if isinstance(obs, dict):
        return obs
    raise TypeError(f"Unsupported batch['obs'] type: {type(obs)}")


def make_dataset_and_loaders(
    cfg: OmegaConf,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[Dict[str, DataLoader], str, str]:
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    if not hasattr(dataset, "get_validation_dataset"):
        raise TypeError(
            f"Dataset does not implement get_validation_dataset(): {type(dataset)}"
        )

    test_dataset = dataset.get_validation_dataset()
    loaders = {
        "train": DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ),
        # This is the validation split from the Push-T dataset. We name it
        # "test" because the request asks for train/test split.
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ),
    }

    dataset_type = dataset.__class__.__module__ + "." + dataset.__class__.__name__
    dataset_kind = "unknown"
    if len(dataset) > 0:
        sample_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        sample_batch = next(iter(sample_loader))
        dataset_kind = dataset_kind_from_batch(sample_batch)

    return loaders, dataset_kind, dataset_type


def to_device_float(batch: Any, device: torch.device) -> Any:
    def _move(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device, non_blocking=True)
        if torch.is_floating_point(x):
            x = x.float()
        return x

    return dict_apply(batch, _move)


def get_policy_attr(policy: torch.nn.Module, name: str, default=None):
    return getattr(policy, name, default)


def normalize_actions_with_mlp(policy_mlp, action: torch.Tensor) -> torch.Tensor:
    return policy_mlp.normalizer["action"].normalize(action)


def current_agent_pos_normalized(policy_mlp, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    n_obs_steps = int(get_policy_attr(policy_mlp, "n_obs_steps"))

    if "agent_pos" in obs:
        # Image Push-T: obs is {"image": ..., "agent_pos": ...}.
        raw_agent_pos = obs["agent_pos"][:, n_obs_steps - 1, :]
    elif "obs" in obs:
        # Lowdim Push-T: last two dims are raw agent_pos. This follows
        # DirectionMagMLPLowdimPolicy._current_agent_pos_action_normalized().
        raw_agent_pos = obs["obs"][:, n_obs_steps - 1, -2:]
    else:
        raise KeyError(
            "Cannot find current agent position. Expected image obs key "
            "'agent_pos' or lowdim obs key 'obs'."
        )

    return policy_mlp.normalizer["action"].normalize(raw_agent_pos)


def actions_to_normalized_delta(
    policy_mlp,
    raw_action: torch.Tensor,
    obs: Dict[str, torch.Tensor],
    compare_steps: int,
) -> torch.Tensor:
    """Convert absolute action targets to DirectionMag-style normalized deltas."""
    naction = normalize_actions_with_mlp(policy_mlp, raw_action[:, :compare_steps, :])
    n_agent_pos = current_agent_pos_normalized(policy_mlp, obs)
    prev = torch.cat([n_agent_pos[:, None, :], naction[:, :-1, :]], dim=1)
    return naction - prev


def delta_to_theta_mag(delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = torch.atan2(delta[..., 1], delta[..., 0])
    mag = torch.linalg.norm(delta, dim=-1)
    return theta, mag


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def nearest_dir_idx(theta: torch.Tensor, angle_centers: torch.Tensor) -> torch.Tensor:
    diff = angle_diff(theta.unsqueeze(-1), angle_centers.to(device=theta.device, dtype=theta.dtype))
    return diff.abs().argmin(dim=-1)


def stats_from_array(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "max": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "max": float(values.max()),
    }


def run_split(
    split_name: str,
    loader: DataLoader,
    diffusion_policy,
    mlp_policy,
    device: torch.device,
    seed: int,
    max_batches: Optional[int],
    compare_steps_arg: Optional[int],
    valid_mag_eps: Optional[float],
) -> Dict[str, Any]:
    if len(loader.dataset) == 0:
        return {
            "num_dataset_items": 0,
            "num_batches": 0,
            "num_windows": 0,
            "num_valid_windows": 0,
            "num_valid_steps": 0,
            "compare_steps": None,
            "category_consistency_rate": {"mean": None, "std": None, "max": None, "global": None},
            "angle_error_deg": {"mean": None, "std": None, "max": None},
        }

    mlp_bins = int(get_policy_attr(mlp_policy, "num_dir_bins", 32))
    if hasattr(mlp_policy, "angle_centers"):
        angle_centers = mlp_policy.angle_centers.detach()
    else:
        half_bin = math.pi / mlp_bins
        angle_centers = torch.linspace(-math.pi + half_bin, math.pi - half_bin, mlp_bins)
    angle_centers = angle_centers.to(device=device)

    if valid_mag_eps is None:
        valid_mag_eps = float(get_policy_attr(mlp_policy, "dir_eps", 1.0e-2))

    per_window_category_rates: List[np.ndarray] = []
    per_step_matches: List[np.ndarray] = []
    per_step_angle_errors_deg: List[np.ndarray] = []

    num_batches = 0
    num_windows = 0
    num_valid_steps = 0
    compare_steps_used: Optional[int] = None

    pbar = tqdm(loader, desc=f"align_check/{split_name}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = to_device_float(batch, device)
        obs = make_policy_obs(batch)

        # DP sampling is stochastic. Reset per batch so repeated runs are comparable.
        seed_everything(seed + batch_idx)
        with torch.no_grad():
            diffusion_out = diffusion_policy.predict_action(obs)
            mlp_out = mlp_policy.predict_action(obs)

        dp_action = diffusion_out["action"]
        mlp_action = mlp_out["action"]
        compare_steps = min(dp_action.shape[1], mlp_action.shape[1])
        if compare_steps_arg is not None:
            compare_steps = min(compare_steps, int(compare_steps_arg))
        if compare_steps <= 0:
            raise RuntimeError("No action step available for comparison.")
        if compare_steps_used is None:
            compare_steps_used = compare_steps
        elif compare_steps_used != compare_steps:
            raise RuntimeError(
                f"compare_steps changed across batches: {compare_steps_used} -> {compare_steps}"
            )

        dp_delta = actions_to_normalized_delta(mlp_policy, dp_action, obs, compare_steps)
        mlp_delta = actions_to_normalized_delta(mlp_policy, mlp_action, obs, compare_steps)
        dp_theta, dp_mag = delta_to_theta_mag(dp_delta)
        mlp_theta, mlp_mag = delta_to_theta_mag(mlp_delta)

        dp_dir_idx = nearest_dir_idx(dp_theta, angle_centers)
        if "dir_idx" in mlp_out:
            mlp_dir_idx = mlp_out["dir_idx"][:, :compare_steps].to(device=device)
        else:
            mlp_dir_idx = nearest_dir_idx(mlp_theta, angle_centers)

        valid = (dp_mag > valid_mag_eps) & (mlp_mag > valid_mag_eps)
        match = (dp_dir_idx == mlp_dir_idx) & valid
        angle_err = angle_diff(mlp_theta, dp_theta).abs() * (180.0 / math.pi)

        valid_np = valid.detach().cpu().numpy().astype(bool)
        match_np = match.detach().cpu().numpy().astype(bool)
        angle_np = angle_err.detach().cpu().numpy()

        valid_count_per_window = valid_np.sum(axis=1)
        match_count_per_window = match_np.sum(axis=1)
        rate = np.full((valid_np.shape[0],), np.nan, dtype=np.float64)
        non_empty = valid_count_per_window > 0
        rate[non_empty] = match_count_per_window[non_empty] / valid_count_per_window[non_empty]

        per_window_category_rates.append(rate)
        per_step_matches.append(match_np[valid_np].astype(np.float64))
        per_step_angle_errors_deg.append(angle_np[valid_np].astype(np.float64))

        num_batches += 1
        num_windows += valid_np.shape[0]
        num_valid_steps += int(valid_np.sum())
        if num_valid_steps > 0:
            pbar.set_postfix(valid_steps=num_valid_steps)

    if per_window_category_rates:
        category_rates = np.concatenate(per_window_category_rates, axis=0)
    else:
        category_rates = np.asarray([], dtype=np.float64)

    if per_step_matches:
        step_matches = np.concatenate(per_step_matches, axis=0)
    else:
        step_matches = np.asarray([], dtype=np.float64)

    if per_step_angle_errors_deg:
        angle_errors_deg = np.concatenate(per_step_angle_errors_deg, axis=0)
    else:
        angle_errors_deg = np.asarray([], dtype=np.float64)

    cat_stats = stats_from_array(category_rates)
    cat_stats["global"] = None if step_matches.size == 0 else float(step_matches.mean())

    return {
        "num_dataset_items": int(len(loader.dataset)),
        "num_batches": int(num_batches),
        "num_windows": int(num_windows),
        "num_valid_windows": int(np.isfinite(category_rates).sum()),
        "num_valid_steps": int(num_valid_steps),
        "compare_steps": None if compare_steps_used is None else int(compare_steps_used),
        "valid_mag_eps_normalized": float(valid_mag_eps),
        "category_consistency_rate": cat_stats,
        "angle_error_deg": stats_from_array(angle_errors_deg),
    }


def policy_info(policy, ckpt_path: pathlib.Path, used_ema: bool) -> Dict[str, Any]:
    return {
        "checkpoint": str(ckpt_path),
        "class": policy.__class__.__module__ + "." + policy.__class__.__name__,
        "used_ema": bool(used_ema),
        "horizon": int(get_policy_attr(policy, "horizon", -1)),
        "n_obs_steps": int(get_policy_attr(policy, "n_obs_steps", -1)),
        "n_action_steps": int(get_policy_attr(policy, "n_action_steps", -1)),
        "pred_action_steps": (
            None
            if get_policy_attr(policy, "pred_action_steps", None) is None
            else int(get_policy_attr(policy, "pred_action_steps"))
        ),
        "num_dir_bins": (
            None
            if get_policy_attr(policy, "num_dir_bins", None) is None
            else int(get_policy_attr(policy, "num_dir_bins"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion-ckpt", "--dp",
        dest="diffusion_ckpt",
        required=True,
        help="Path to DiffusionUnet checkpoint. Alias: --dp",
    )
    parser.add_argument(
        "--mlp-ckpt", "--mlp",
        dest="mlp_ckpt",
        required=True,
        help="Path to DirectionMagMLP checkpoint. Alias: --mlp",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", choices=["auto", "true", "false"], default="auto")
    parser.add_argument(
        "--dataset-source",
        choices=["diffusion", "mlp"],
        default="diffusion",
        help="Which checkpoint cfg.task.dataset to instantiate. Default: diffusion.",
    )
    parser.add_argument(
        "--diffusion-num-inference-steps",
        type=int,
        default=None,
        help="Override diffusion_policy.num_inference_steps during this check.",
    )
    parser.add_argument(
        "--compare-steps",
        type=int,
        default=None,
        help="Compare only the first N executed action steps. Default: min(policy outputs).",
    )
    parser.add_argument(
        "--valid-mag-eps",
        type=float,
        default=None,
        help="Ignore angle/category for steps whose normalized delta magnitude is <= eps. Default: mlp.dir_eps.",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Debug: limit batches per split")
    parser.add_argument("--output-dir", default="data/eval/align_check")
    parser.add_argument("--name", default=None, help="Optional output json stem")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    diffusion_ws, diffusion_policy, diffusion_ckpt, diffusion_used_ema = load_workspace_from_ckpt(
        args.diffusion_ckpt, device=device, use_ema=args.use_ema
    )
    mlp_ws, mlp_policy, mlp_ckpt, mlp_used_ema = load_workspace_from_ckpt(
        args.mlp_ckpt, device=device, use_ema=args.use_ema
    )

    if args.diffusion_num_inference_steps is not None:
        if not hasattr(diffusion_policy, "num_inference_steps"):
            raise RuntimeError("Diffusion policy has no num_inference_steps attribute to override.")
        diffusion_policy.num_inference_steps = int(args.diffusion_num_inference_steps)

    # Make image random crop deterministic center crop during eval.
    diffusion_policy.eval()
    mlp_policy.eval()

    dataset_cfg = diffusion_ws.cfg if args.dataset_source == "diffusion" else mlp_ws.cfg
    loaders, dataset_kind, dataset_type = make_dataset_and_loaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    warnings: List[str] = []
    if int(get_policy_attr(diffusion_policy, "n_obs_steps", -1)) != int(get_policy_attr(mlp_policy, "n_obs_steps", -2)):
        warnings.append(
            "diffusion_policy.n_obs_steps != mlp_policy.n_obs_steps; comparison still runs, "
            "but the two policies consume different observation windows."
        )
    if int(get_policy_attr(diffusion_policy, "n_action_steps", -1)) != int(get_policy_attr(mlp_policy, "n_action_steps", -2)):
        warnings.append(
            "diffusion_policy.n_action_steps != mlp_policy.n_action_steps; using min output length unless --compare-steps is set."
        )

    result = {
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "coordinate_space": "mlp_policy.normalizer['action'] normalized action space",
        "dataset_kind": dataset_kind,
        "dataset_type": dataset_type,
        "note": (
            "test split is dataset.get_validation_dataset(); no environment rollout is used. "
            "category_consistency_rate.mean/std/max are computed over per-window valid-step match rates; "
            "category_consistency_rate.global is the valid per-step match mean."
        ),
        "args": vars(args),
        "warnings": warnings,
        "diffusion_policy": policy_info(diffusion_policy, diffusion_ckpt, diffusion_used_ema),
        "mlp_policy": policy_info(mlp_policy, mlp_ckpt, mlp_used_ema),
        "splits": {},
    }

    for split_name, loader in loaders.items():
        split_result = run_split(
            split_name=split_name,
            loader=loader,
            diffusion_policy=diffusion_policy,
            mlp_policy=mlp_policy,
            device=device,
            seed=args.seed + (0 if split_name == "train" else 1_000_000),
            max_batches=args.max_batches,
            compare_steps_arg=args.compare_steps,
            valid_mag_eps=args.valid_mag_eps,
        )
        result["splits"][split_name] = split_result

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.name is None:
        stem = _datetime.datetime.now().strftime("align_check_%Y%m%d_%H%M%S")
    else:
        stem = args.name
    output_path = output_dir / f"{stem}.json"
    latest_path = output_dir / "latest.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result["splits"], indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()