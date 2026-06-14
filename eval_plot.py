"""
Evaluate a Diffusion Policy checkpoint on Push-T and save score-vs-step curves.

This is a drop-in replacement for the original eval.py CLI, with extra outputs:
  - eval_log.json: scalar metrics + per-seed max score
  - reward_curves.npz: raw padded per-step rewards
  - reward_curve.png: mean step-score curve

Important split behavior:
  - --split test:  run only test rollouts and draw one curve.
  - --split train: run only train rollouts and draw one curve.
  - --split all:   run train and test separately, then draw two different
                   curves on the same figure.

Compatibility:
  - Supports current lowdim/keypoints checkpoints.
  - Supports current image checkpoints.
  - Also remaps older official image/hybrid checkpoint target names to the
    current image workspace/policy names when possible.

Typical usage:
  python eval_plot_train_test.py \
    --checkpoint data/outputs/.../checkpoints/latest.ckpt \
    --device cuda:0 \
    --split all

By default, output_dir is derived from the checkpoint filename without its
parent folders, e.g. /path/to/epoch_1750.ckpt -> data/eval/epoch_1750.
You can still override it with --output_dir.
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import math
import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib
import types

import click
import dill
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf, DictConfig, ListConfig

# Use a non-interactive backend so the script works on servers without DISPLAY.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def torch_load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load old/new PyTorch checkpoints robustly."""
    with open(path, "rb") as f:
        try:
            return torch.load(
                f,
                map_location=device,
                pickle_module=dill,
                weights_only=False,
            )
        except TypeError:
            f.seek(0)
            return torch.load(
                f,
                map_location=device,
                pickle_module=dill,
            )


def omega_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    try:
        if key in container:
            return container[key]
    except Exception:
        pass
    try:
        return getattr(container, key)
    except Exception:
        return default


LEGACY_TARGET_ALIASES: Dict[str, str] = {
    # Official / older Diffusion Policy image checkpoints used "hybrid"
    # naming. This cleaned repo uses "image" naming for the same Push-T image
    # policy/workspace family.
    "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace":
        "diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace",
    "diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy":
        "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy",
}

LEGACY_MODULE_ALIASES: Sequence[Tuple[str, str, str, str]] = (
    (
        "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace",
        "TrainDiffusionUnetHybridWorkspace",
        "diffusion_policy.workspace.train_diffusion_unet_image_workspace",
        "TrainDiffusionUnetImageWorkspace",
    ),
    (
        "diffusion_policy.policy.diffusion_unet_hybrid_image_policy",
        "DiffusionUnetHybridImagePolicy",
        "diffusion_policy.policy.diffusion_unet_image_policy",
        "DiffusionUnetImagePolicy",
    ),
)


def install_legacy_import_aliases() -> None:
    """Install import aliases for old checkpoint target names.

    Some official image checkpoints store target strings such as
    diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.
    This repo has the equivalent implementation under image names. Patching the
    cfg usually handles this, but module aliases make Hydra/dill fallback-safe.
    """
    for old_module, old_class, new_module, new_class in LEGACY_MODULE_ALIASES:
        if old_module in sys.modules:
            continue
        try:
            new_mod = importlib.import_module(new_module)
            new_cls = getattr(new_mod, new_class)
        except Exception:
            continue
        alias_mod = types.ModuleType(old_module)
        setattr(alias_mod, old_class, new_cls)
        sys.modules[old_module] = alias_mod
        try:
            parent_name, leaf_name = old_module.rsplit(".", 1)
            parent_mod = importlib.import_module(parent_name)
            setattr(parent_mod, leaf_name, alias_mod)
        except Exception:
            pass


def _patch_legacy_targets_in_node(node: Any, path: str, patches: List[Tuple[str, str, str]]) -> None:
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            try:
                value = node[key]
            except Exception:
                continue
            child_path = f"{path}.{key}"
            if isinstance(value, str):
                new_value = LEGACY_TARGET_ALIASES.get(value)
                if new_value is not None:
                    node[key] = new_value
                    patches.append((child_path, value, new_value))
            else:
                _patch_legacy_targets_in_node(value, child_path, patches)
    elif isinstance(node, ListConfig):
        for idx in range(len(node)):
            try:
                value = node[idx]
            except Exception:
                continue
            child_path = f"{path}[{idx}]"
            if isinstance(value, str):
                new_value = LEGACY_TARGET_ALIASES.get(value)
                if new_value is not None:
                    node[idx] = new_value
                    patches.append((child_path, value, new_value))
            else:
                _patch_legacy_targets_in_node(value, child_path, patches)


def patch_legacy_targets(cfg: OmegaConf) -> List[Tuple[str, str, str]]:
    """Rewrite known old target strings in-place and return applied patches."""
    OmegaConf.set_struct(cfg, False)
    patches: List[Tuple[str, str, str]] = []
    _patch_legacy_targets_in_node(cfg, "cfg", patches)
    return patches


def get_workspace_class_from_cfg(cfg: OmegaConf):
    """Resolve workspace class, with a final image/lowdim fallback."""
    try:
        return hydra.utils.get_class(cfg._target_)
    except Exception as first_error:
        # Last-resort fallback. This helps if a checkpoint has an unknown old
        # workspace target but the policy/env_runner clearly identifies which
        # implementation family it needs.
        policy_target = str(omega_get(omega_get(cfg, "policy", None), "_target_", ""))
        runner_target = str(omega_get(omega_get(omega_get(cfg, "task", None), "env_runner", None), "_target_", ""))
        combined = (str(omega_get(cfg, "_target_", "")) + " " + policy_target + " " + runner_target).lower()

        if "direction_mag_mlp" in combined:
            fallback = "diffusion_policy.workspace.train_direction_mag_mlp_lowdim_workspace.TrainDirectionMagMLPLowdimWorkspace"
        elif "image" in combined or "hybrid" in combined:
            fallback = "diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace"
        elif "lowdim" in combined or "keypoint" in combined or "flow" in combined or "xpred" in combined or "gmm" in combined:
            fallback = "diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace.TrainDiffusionUnetLowdimWorkspace"
        else:
            raise first_error

        print(f"[compat] Failed to import workspace target {cfg._target_!r}; falling back to {fallback!r}.")
        cfg._target_ = fallback
        return hydra.utils.get_class(fallback)


def default_output_dir_from_checkpoint(checkpoint: str) -> str:
    """Use checkpoint filename as output directory when --output_dir is omitted."""
    name = pathlib.Path(checkpoint).name
    stem = pathlib.Path(name).stem
    directory = stem if stem else name
    directory = pathlib.Path("data") / "eval" / directory
    return str(directory)


def apply_eval_overrides(
    cfg: OmegaConf,
    split: str,
    num_rollouts: Optional[int],
    max_steps: Optional[int],
    n_envs: Optional[int],
    n_vis: Optional[int],
) -> None:
    """Mutate cfg.task.env_runner before hydra instantiation.

    For split=all, train and test are both kept. If --num-rollouts is set, it
    means that many train rollouts AND that many test rollouts, not a shared
    total budget that is split between them.
    """
    OmegaConf.set_struct(cfg, False)
    runner_cfg = cfg.task.env_runner

    if max_steps is not None:
        runner_cfg.max_steps = int(max_steps)
    if n_envs is not None:
        runner_cfg.n_envs = int(n_envs)

    if split == "test":
        runner_cfg.n_train = 0
        runner_cfg.n_train_vis = 0
        if num_rollouts is not None:
            runner_cfg.n_test = int(num_rollouts)
        if n_vis is not None:
            runner_cfg.n_test_vis = min(int(n_vis), int(omega_get(runner_cfg, "n_test", 0)))

    elif split == "train":
        runner_cfg.n_test = 0
        runner_cfg.n_test_vis = 0
        if num_rollouts is not None:
            runner_cfg.n_train = int(num_rollouts)
        if n_vis is not None:
            runner_cfg.n_train_vis = min(int(n_vis), int(omega_get(runner_cfg, "n_train", 0)))

    elif split == "all":
        if num_rollouts is not None:
            runner_cfg.n_train = int(num_rollouts)
            runner_cfg.n_test = int(num_rollouts)
        if n_vis is not None:
            # Treat n_vis as per-split for split=all.
            runner_cfg.n_train_vis = min(int(n_vis), int(omega_get(runner_cfg, "n_train", 0)))
            runner_cfg.n_test_vis = min(int(n_vis), int(omega_get(runner_cfg, "n_test", 0)))

    else:
        raise ValueError(f"Unsupported split: {split}")


def instantiate_env_runner(cfg: OmegaConf, output_dir: str):
    return hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)


def make_obs_dict(runner: Any, obs: Any, past_action: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Build policy input exactly like the repo's PushT runners."""
    if isinstance(obs, dict):
        # Image runner: obs is already a batched dict with keys like image/agent_pos.
        np_obs_dict = dict(obs)
        if getattr(runner, "past_action", False) and past_action is not None:
            np_obs_dict["past_action"] = past_action[:, -(runner.n_obs_steps - 1):].astype(np.float32)
        return np_obs_dict

    # Keypoint lowdim runner: env observation concatenates obs and obs_mask.
    do = obs.shape[-1] // 2
    np_obs_dict = {
        "obs": obs[..., :runner.n_obs_steps, :do].astype(np.float32),
        "obs_mask": obs[..., :runner.n_obs_steps, do:] > 0.5,
    }
    if getattr(runner, "past_action", False) and past_action is not None:
        np_obs_dict["past_action"] = past_action[:, -(runner.n_obs_steps - 1):].astype(np.float32)
    return np_obs_dict


def get_action_for_env(runner: Any, action_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
    action = np_action_dict["action"]

    # Lowdim runner simulates latency by discarding the first n_latency_steps.
    if hasattr(runner, "n_latency_steps"):
        action = action[:, int(runner.n_latency_steps):]
    return action


def pad_rewards(rewards: Sequence[float], max_steps: int) -> np.ndarray:
    """Pad early-ended episodes with 1.0 to keep score-step curves full length."""
    rewards = [float(x) for x in rewards]
    if len(rewards) < max_steps:
        rewards = rewards + [1.0] * (max_steps - len(rewards))
    return np.asarray(rewards[:max_steps], dtype=np.float32)


def select_indices(prefixes: Sequence[str], split: str) -> List[int]:
    if split not in {"train", "test"}:
        raise ValueError("run_step_curve_eval expects split='train' or split='test'. Use main(split='all') to run both separately.")
    wanted = split + "/"
    return [i for i, prefix in enumerate(prefixes) if prefix == wanted]


def run_step_curve_eval(
    runner: Any,
    policy: Any,
    split: str,
) -> Dict[str, Any]:
    device = policy.device
    env = runner.env
    max_steps = int(runner.max_steps)

    indices = select_indices(runner.env_prefixs, split)
    if len(indices) == 0:
        raise RuntimeError(f"No environments selected for split={split!r}.")

    n_envs = len(runner.env_fns)
    n_inits = len(indices)
    n_chunks = math.ceil(n_inits / n_envs)

    all_rewards_raw: List[List[float]] = [None] * n_inits
    all_video_paths: List[Optional[str]] = [None] * n_inits
    all_seeds: List[int] = [runner.env_seeds[i] for i in indices]
    all_prefixes: List[str] = [runner.env_prefixs[i] for i in indices]

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end = min(n_inits, start + n_envs)
        local_count = end - start
        selected = indices[start:end]

        init_fns = [runner.env_init_fn_dills[i] for i in selected]
        if local_count < n_envs:
            init_fns = init_fns + [init_fns[0]] * (n_envs - local_count)
        assert len(init_fns) == n_envs

        env.call_each("run_dill_function", args_list=[(x,) for x in init_fns])

        obs = env.reset()
        past_action = None
        policy.reset()

        pbar = tqdm.tqdm(
            total=max_steps,
            desc=f"Eval step curve {split} {chunk_idx + 1}/{n_chunks}",
            leave=False,
            mininterval=float(getattr(runner, "tqdm_interval_sec", 1.0)),
        )

        done = np.zeros((n_envs,), dtype=np.bool_)
        approx_steps = 0
        while not np.all(done[:local_count]) and approx_steps < max_steps:
            np_obs_dict = make_obs_dict(runner, obs, past_action)
            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device),
            )

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            action = get_action_for_env(runner, action_dict)
            obs, reward, done, info = env.step(action)
            past_action = action

            # action.shape[1] is the attempted chunk length. MultiStepWrapper stops
            # internally if the episode terminates or hits max_episode_steps.
            step_inc = int(action.shape[1]) if action.ndim >= 3 else 1
            step_inc = min(step_inc, max_steps - approx_steps)
            approx_steps = min(max_steps, approx_steps + step_inc)
            pbar.update(step_inc)
        pbar.close()

        rewards_chunk = env.call("get_attr", "reward")[:local_count]
        video_paths_chunk = env.render()[:local_count]

        for j in range(local_count):
            all_rewards_raw[start + j] = list(rewards_chunk[j])
            all_video_paths[start + j] = video_paths_chunk[j]

    padded = np.stack([pad_rewards(r, max_steps) for r in all_rewards_raw], axis=0)
    episode_max = np.nanmax(padded, axis=1)
    final_score = padded[:, -1]
    best_so_far = np.maximum.accumulate(np.nan_to_num(padded, nan=0.0), axis=1)

    return {
        "split": split,
        "seeds": all_seeds,
        "prefixes": all_prefixes,
        "video_paths": all_video_paths,
        "raw_lengths": [len(r) for r in all_rewards_raw],
        "rewards": padded,
        "mean_curve": np.nanmean(padded, axis=0),
        "std_curve": np.nanstd(padded, axis=0),
        "best_so_far_mean_curve": np.nanmean(best_so_far, axis=0),
        "episode_max": episode_max,
        "final_score": final_score,
        "metrics": {
            # Same aggregate idea as original env_runner: mean over per-episode max rewards.
            f"{split}/mean_score": float(np.nanmean(episode_max)),
            f"{split}/final_step_mean_score": float(np.nanmean(final_score)),
            f"{split}/success_rate_0.95": float(np.nanmean(episode_max >= 0.95)),
            f"{split}/success_rate_0.99": float(np.nanmean(episode_max >= 0.99)),
            f"{split}/auc_mean_score": float(np.nanmean(padded)),
        },
    }


def plot_curve(eval_data: Dict[str, Any], output_path: str, title: Optional[str] = None, show_best_so_far: bool = True) -> None:
    rewards = eval_data["rewards"]
    mean = eval_data["mean_curve"]
    std = eval_data["std_curve"]
    x = np.arange(mean.shape[0])
    split = eval_data["split"]

    plt.figure(figsize=(11, 7))
    plt.plot(x, mean, label=f"{split} mean score", linewidth=2.2)
    plt.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15, label=f"{split} ±1 std")

    if show_best_so_far:
        plt.plot(x, eval_data["best_so_far_mean_curve"], label=f"{split} mean best-so-far score", linestyle="--", linewidth=1.8)

    if title is None:
        title = f"Push-T Score-Step Curve ({split}, {rewards.shape[0]} rollouts)"
    plt.title(title)
    plt.xlabel("Environment step")
    plt.ylabel("Score / target coverage")
    plt.xlim(0, mean.shape[0] - 1)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_curves(
    eval_datas: Sequence[Dict[str, Any]],
    output_path: str,
    title: Optional[str] = None,
    show_best_so_far: bool = True,
) -> None:
    plt.figure(figsize=(11, 7))

    max_len = 0
    for eval_data in eval_datas:
        mean = eval_data["mean_curve"]
        std = eval_data["std_curve"]
        x = np.arange(mean.shape[0])
        split = eval_data["split"]
        max_len = max(max_len, mean.shape[0])

        plt.plot(x, mean, label=f"{split} mean score", linewidth=2.2)
        plt.fill_between(
            x,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            alpha=0.12,
            label=f"{split} ±1 std",
        )

        if show_best_so_far:
            plt.plot(
                x,
                eval_data["best_so_far_mean_curve"],
                label=f"{split} mean best-so-far score",
                linestyle="--",
                linewidth=1.5,
            )

    if title is None:
        parts = [f"{x['split']}={x['rewards'].shape[0]}" for x in eval_datas]
        title = "Push-T Train/Test Score-Step Curves (" + ", ".join(parts) + " rollouts)"

    plt.title(title)
    plt.xlabel("Environment step")
    plt.ylabel("Score / target coverage")
    plt.xlim(0, max_len - 1)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_log_data(eval_data: Dict[str, Any], plot_path: pathlib.Path, npz_path: pathlib.Path) -> Dict[str, Any]:
    log_data: Dict[str, Any] = dict(eval_data["metrics"])
    log_data.update({
        "split": eval_data["split"],
        "num_rollouts": int(eval_data["rewards"].shape[0]),
        "max_steps": int(eval_data["rewards"].shape[1]),
        "reward_curve_png": str(plot_path),
        "reward_curves_npz": str(npz_path),
    })

    for seed, prefix, max_reward, final_reward, length, video_path in zip(
        eval_data["seeds"],
        eval_data["prefixes"],
        eval_data["episode_max"],
        eval_data["final_score"],
        eval_data["raw_lengths"],
        eval_data["video_paths"],
    ):
        key_base = f"{prefix}sim_{seed}"
        log_data[f"{key_base}/max_reward"] = float(max_reward)
        log_data[f"{key_base}/final_reward"] = float(final_reward)
        log_data[f"{key_base}/length"] = int(length)
        if video_path is not None:
            log_data[f"{key_base}/video_path"] = str(video_path)

    return log_data


def save_outputs(eval_data: Dict[str, Any], output_dir: str, show_best_so_far: bool) -> Dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "reward_curves.npz"
    np.savez_compressed(
        npz_path,
        rewards=eval_data["rewards"],
        mean_curve=eval_data["mean_curve"],
        std_curve=eval_data["std_curve"],
        best_so_far_mean_curve=eval_data["best_so_far_mean_curve"],
        episode_max=eval_data["episode_max"],
        final_score=eval_data["final_score"],
        seeds=np.asarray(eval_data["seeds"], dtype=np.int64),
        raw_lengths=np.asarray(eval_data["raw_lengths"], dtype=np.int64),
    )

    plot_path = output_dir / "reward_curve.png"
    plot_curve(eval_data, str(plot_path), show_best_so_far=show_best_so_far)

    log_data = build_log_data(eval_data, plot_path=plot_path, npz_path=npz_path)

    json_path = output_dir / "eval_log.json"
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2, sort_keys=True)

    return log_data


def save_comparison_outputs(
    eval_datas: Sequence[Dict[str, Any]],
    output_dir: str,
    show_best_so_far: bool,
) -> Dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {}
    for eval_data in eval_datas:
        split = eval_data["split"]
        arrays[f"{split}_rewards"] = eval_data["rewards"]
        arrays[f"{split}_mean_curve"] = eval_data["mean_curve"]
        arrays[f"{split}_std_curve"] = eval_data["std_curve"]
        arrays[f"{split}_best_so_far_mean_curve"] = eval_data["best_so_far_mean_curve"]
        arrays[f"{split}_episode_max"] = eval_data["episode_max"]
        arrays[f"{split}_final_score"] = eval_data["final_score"]
        arrays[f"{split}_seeds"] = np.asarray(eval_data["seeds"], dtype=np.int64)
        arrays[f"{split}_raw_lengths"] = np.asarray(eval_data["raw_lengths"], dtype=np.int64)

    npz_path = output_dir / "reward_curves_train_test.npz"
    np.savez_compressed(npz_path, **arrays)

    plot_path = output_dir / "reward_curve_train_test.png"
    plot_comparison_curves(eval_datas, str(plot_path), show_best_so_far=show_best_so_far)

    log_data: Dict[str, Any] = {
        "split": "all",
        "reward_curve_png": str(plot_path),
        "reward_curves_npz": str(npz_path),
    }

    for eval_data in eval_datas:
        split = eval_data["split"]
        log_data.update(eval_data["metrics"])
        log_data[f"{split}/num_rollouts"] = int(eval_data["rewards"].shape[0])
        log_data[f"{split}/max_steps"] = int(eval_data["rewards"].shape[1])

        for seed, prefix, max_reward, final_reward, length, video_path in zip(
            eval_data["seeds"],
            eval_data["prefixes"],
            eval_data["episode_max"],
            eval_data["final_score"],
            eval_data["raw_lengths"],
            eval_data["video_paths"],
        ):
            key_base = f"{prefix}sim_{seed}"
            log_data[f"{key_base}/max_reward"] = float(max_reward)
            log_data[f"{key_base}/final_reward"] = float(final_reward)
            log_data[f"{key_base}/length"] = int(length)
            if video_path is not None:
                log_data[f"{key_base}/video_path"] = str(video_path)

    json_path = output_dir / "eval_log.json"
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2, sort_keys=True)

    return log_data


@click.command()
@click.option("-c", "--checkpoint", required=True, type=str, help="Path to .ckpt checkpoint.")
@click.option("-o", "--output_dir", default=None, type=str, help="Directory for eval outputs. Defaults to checkpoint filename without parent folders.")
@click.option("-d", "--device", default="cuda:0", show_default=True, type=str)
@click.option("--split", default="all", show_default=True, type=click.Choice(["test", "train", "all"]), help="Which rollout split to run/plot. split=all runs train and test separately and plots both curves together.")
@click.option("--num-rollouts", default=None, type=int, help="Override rollouts. For split=all, this is per split, not train+test total.")
@click.option("--max-steps", default=1000, show_default=True, type=int, help="Override env_runner.max_steps.")
@click.option("--n-envs", default=None, type=int, help="Override number of vectorized envs.")
@click.option("--n-vis", default=None, type=int, help="Override number of recorded videos. For split=all, this is per split.")
@click.option("--show-best-so-far/--hide-best-so-far", default=True, show_default=True, help="Overlay cumulative best-so-far score curve.")
def main(
    checkpoint: str,
    output_dir: str,
    device: str,
    split: str,
    num_rollouts: Optional[int],
    max_steps: Optional[int],
    n_envs: Optional[int],
    n_vis: Optional[int],
    show_best_so_far: bool,
):
    if output_dir is None:
        output_dir = default_output_dir_from_checkpoint(checkpoint)
        print(f"[info] --output_dir not provided; using {output_dir!r} from checkpoint filename.")

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)

    install_legacy_import_aliases()

    payload = torch_load_checkpoint(checkpoint, device=torch.device("cpu"))
    cfg = payload["cfg"]
    patches = patch_legacy_targets(cfg)
    for path, old, new in patches:
        print(f"[compat] patched {path}: {old} -> {new}")

    apply_eval_overrides(
        cfg=cfg,
        split=split,
        num_rollouts=num_rollouts,
        max_steps=max_steps,
        n_envs=n_envs,
        n_vis=n_vis,
    )

    # Instantiate workspace/model exactly like the original eval.py, while
    # tolerating old official image/hybrid target names.
    cls = get_workspace_class_from_cfg(cfg)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if bool(getattr(cfg.training, "use_ema", False)):
        policy = workspace.ema_model

    policy.to(device_obj)
    policy.eval()

    runner = instantiate_env_runner(cfg, output_dir=output_dir)

    if split == "all":
        eval_datas = [
            run_step_curve_eval(runner, policy, split="train"),
            run_step_curve_eval(runner, policy, split="test"),
        ]
        log_data = save_comparison_outputs(eval_datas, output_dir, show_best_so_far=show_best_so_far)
    else:
        eval_data = run_step_curve_eval(runner, policy, split=split)
        log_data = save_outputs(eval_data, output_dir, show_best_so_far=show_best_so_far)

    print("\n=== Step-curve evaluation summary ===")
    for key, value in sorted(log_data.items()):
        if isinstance(value, (int, float)) and (
            key.endswith("mean_score") or "success_rate" in key or key.endswith("auc_mean_score")
        ):
            print(f"{key}: {value:.6f}")
    print(f"Saved JSON: {pathlib.Path(output_dir) / 'eval_log.json'}")
    if split == "all":
        print(f"Saved curves: {pathlib.Path(output_dir) / 'reward_curves_train_test.npz'}")
        print(f"Saved plot: {pathlib.Path(output_dir) / 'reward_curve_train_test.png'}")
    else:
        print(f"Saved curves: {pathlib.Path(output_dir) / 'reward_curves.npz'}")
        print(f"Saved plot: {pathlib.Path(output_dir) / 'reward_curve.png'}")


if __name__ == "__main__":
    main()