"""
Generate Push-T human-dataset upper-bound score-vs-step curves.

This script is intentionally NOT a CLI. Put it in the project root and run:

    python gen_human_dataset_eval_plot.py

It reads the same zarr dataset used by the image/lowdim Push-T configs, splits
episodes exactly like the dataset class, then measures the reward curve of the
recorded human states. The train curve and the held-out split curve are plotted
on the same figure.

Outputs:
    data/eval/human_dataset_upper_bound/human_eval_log.json
    data/eval/human_dataset_upper_bound/human_reward_curves_train_split.npz
    data/eval/human_dataset_upper_bound/human_reward_curve_train_split.png
"""

import json
import pathlib
from typing import Any, Dict, List, Sequence

import numpy as np
import zarr
import tqdm

# Use a non-interactive backend so this works on servers without DISPLAY.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_policy.common.sampler import get_val_mask, downsample_mask
from diffusion_policy.env.pusht.pusht_env import PushTEnv


# -----------------------------------------------------------------------------
# Edit constants here if your project layout changes. No command-line options.
# -----------------------------------------------------------------------------
ZARR_PATH = "data/pusht_cchi_v7_replay.zarr"
OUTPUT_DIR = "data/eval/human_dataset_upper_bound"

# Match diffusion_policy/config/task/pusht_image.yaml and pusht_lowdim.yaml.
SPLIT_SEED = 42
VAL_RATIO = 0.02
MAX_TRAIN_EPISODES = 90

# Push-T image/lowdim env_runner default max_steps in this repo is 300.
MAX_STEPS = 300

# The official image/lowdim Push-T runner uses legacy_test=True in the task cfg.
# For measuring recorded states from pusht_cchi_v7_replay.zarr, legacy=True is
# the safer compatibility setting.
LEGACY_ENV = True

# If True, stop a human episode once PushTEnv reports success, then pad with 1.0.
# This mirrors the eval_plot_train_test.py behavior: rollout ends on done, and
# pad_rewards fills the rest with 1.0.
STOP_ON_SUCCESS = True


# -----------------------------------------------------------------------------
# Utility functions, intentionally close to eval_plot_train_test.py.
# -----------------------------------------------------------------------------
def pad_rewards(rewards: Sequence[float], max_steps: int) -> np.ndarray:
    """Pad early-ended successful episodes with 1.0, like eval_plot_train_test.py."""
    rewards = [float(x) for x in rewards]
    if len(rewards) < max_steps:
        rewards = rewards + [1.0] * (max_steps - len(rewards))
    return np.asarray(rewards[:max_steps], dtype=np.float32)


def episode_slice(episode_ends: np.ndarray, episode_idx: int) -> slice:
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return slice(start, end)


def load_zarr_arrays(zarr_path: str) -> Dict[str, Any]:
    root = zarr.open(str(pathlib.Path(zarr_path).expanduser()), mode="r")
    state = root["data"]["state"]
    action = root["data"].get("action", None)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return {
        "root": root,
        "state": state,
        "action": action,
        "episode_ends": episode_ends,
    }


def make_dataset_split_masks(n_episodes: int) -> Dict[str, np.ndarray]:
    """Match PushTImageDataset / PushTLowdimDataset split behavior.

    Important: after max_train_episodes downsampling, validation/split episodes
    are not just the original val_mask; they are everything outside train_mask,
    because dataset.get_validation_dataset() uses episode_mask=~self.train_mask.
    """
    val_mask = get_val_mask(
        n_episodes=n_episodes,
        val_ratio=VAL_RATIO,
        seed=SPLIT_SEED,
    )
    train_mask = ~val_mask
    train_mask = downsample_mask(
        mask=train_mask,
        max_n=MAX_TRAIN_EPISODES,
        seed=SPLIT_SEED,
    )
    split_mask = ~train_mask
    return {
        "train": train_mask,
        "split": split_mask,
        "raw_val": val_mask,
    }


def score_state_trajectory(states: np.ndarray, max_steps: int) -> List[float]:
    """Compute Push-T coverage rewards by replaying recorded states directly.

    This measures the dataset trajectory itself, not a policy/action replay.
    Direct state scoring avoids tiny physics/controller drift and is the cleanest
    estimate of the human-data upper-bound curve.
    """
    env = PushTEnv(legacy=LEGACY_ENV, render_action=False)
    try:
        env.reset()
        rewards: List[float] = []
        horizon = min(int(states.shape[0]), int(max_steps))
        for t in range(horizon):
            env._set_state(np.asarray(states[t], dtype=np.float64))
            # action=None: do not advance the controller; just compute reward for
            # the recorded state using the same PushTEnv scoring function.
            _obs, reward, done, _info = env.step(None)
            rewards.append(float(reward))
            if STOP_ON_SUCCESS and bool(done):
                break
        return rewards
    finally:
        env.close()


def run_split_curve(
    split_name: str,
    episode_indices: Sequence[int],
    state_array: Any,
    episode_ends: np.ndarray,
    max_steps: int,
) -> Dict[str, Any]:
    all_rewards_raw: List[List[float]] = []
    raw_lengths: List[int] = []

    pbar = tqdm.tqdm(
        list(episode_indices),
        desc=f"Human dataset curve: {split_name}",
        mininterval=1.0,
    )
    for epi in pbar:
        sl = episode_slice(episode_ends, int(epi))
        states = np.asarray(state_array[sl], dtype=np.float64)
        rewards = score_state_trajectory(states, max_steps=max_steps)
        all_rewards_raw.append(rewards)
        raw_lengths.append(len(rewards))

    if len(all_rewards_raw) == 0:
        raise RuntimeError(f"No episodes selected for split={split_name!r}.")

    padded = np.stack([pad_rewards(r, max_steps) for r in all_rewards_raw], axis=0)
    episode_max = np.nanmax(padded, axis=1)
    final_score = padded[:, -1]
    best_so_far = np.maximum.accumulate(np.nan_to_num(padded, nan=0.0), axis=1)

    return {
        "split": split_name,
        "episode_indices": np.asarray(episode_indices, dtype=np.int64),
        "raw_lengths": np.asarray(raw_lengths, dtype=np.int64),
        "rewards": padded,
        "mean_curve": np.nanmean(padded, axis=0),
        "std_curve": np.nanstd(padded, axis=0),
        "best_so_far_mean_curve": np.nanmean(best_so_far, axis=0),
        "episode_max": episode_max,
        "final_score": final_score,
        "metrics": {
            f"{split_name}/mean_score": float(np.nanmean(episode_max)),
            f"{split_name}/final_step_mean_score": float(np.nanmean(final_score)),
            f"{split_name}/success_rate_0.95": float(np.nanmean(episode_max >= 0.95)),
            f"{split_name}/success_rate_0.99": float(np.nanmean(episode_max >= 0.99)),
            f"{split_name}/auc_mean_score": float(np.nanmean(padded)),
        },
    }


def plot_train_split_curves(
    eval_datas: Sequence[Dict[str, Any]],
    output_path: str,
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

        plt.plot(x, mean, label=f"{split} human mean score", linewidth=2.2)
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
                label=f"{split} human mean best-so-far score",
                linestyle="--",
                linewidth=1.5,
            )

    parts = [f"{x['split']}={x['rewards'].shape[0]}" for x in eval_datas]
    title = "Push-T Human Dataset Upper-Bound Curves (" + ", ".join(parts) + " episodes)"
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


def save_outputs(eval_datas: Sequence[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {}
    log_data: Dict[str, Any] = {
        "zarr_path": ZARR_PATH,
        "max_steps": int(MAX_STEPS),
        "split_seed": int(SPLIT_SEED),
        "val_ratio": float(VAL_RATIO),
        "max_train_episodes": int(MAX_TRAIN_EPISODES),
        "legacy_env": bool(LEGACY_ENV),
        "stop_on_success": bool(STOP_ON_SUCCESS),
        "scoring_mode": "recorded_state_direct_scoring",
    }

    for eval_data in eval_datas:
        split = eval_data["split"]
        arrays[f"{split}_episode_indices"] = eval_data["episode_indices"]
        arrays[f"{split}_raw_lengths"] = eval_data["raw_lengths"]
        arrays[f"{split}_rewards"] = eval_data["rewards"]
        arrays[f"{split}_mean_curve"] = eval_data["mean_curve"]
        arrays[f"{split}_std_curve"] = eval_data["std_curve"]
        arrays[f"{split}_best_so_far_mean_curve"] = eval_data["best_so_far_mean_curve"]
        arrays[f"{split}_episode_max"] = eval_data["episode_max"]
        arrays[f"{split}_final_score"] = eval_data["final_score"]

        log_data.update(eval_data["metrics"])
        log_data[f"{split}/num_episodes"] = int(eval_data["rewards"].shape[0])
        log_data[f"{split}/max_steps"] = int(eval_data["rewards"].shape[1])
        log_data[f"{split}/episode_indices"] = [int(x) for x in eval_data["episode_indices"]]

    npz_path = output_dir_path / "human_reward_curves_train_split.npz"
    np.savez_compressed(npz_path, **arrays)

    plot_path = output_dir_path / "human_reward_curve_train_split.png"
    plot_train_split_curves(eval_datas, str(plot_path), show_best_so_far=True)

    log_data["reward_curves_npz"] = str(npz_path)
    log_data["reward_curve_png"] = str(plot_path)

    json_path = output_dir_path / "human_eval_log.json"
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2, sort_keys=True)

    return log_data


def main() -> None:
    zarr_data = load_zarr_arrays(ZARR_PATH)
    state_array = zarr_data["state"]
    episode_ends = zarr_data["episode_ends"]
    n_episodes = int(len(episode_ends))

    masks = make_dataset_split_masks(n_episodes=n_episodes)
    train_indices = np.nonzero(masks["train"])[0]
    split_indices = np.nonzero(masks["split"])[0]

    print("=== Human dataset upper-bound score curve ===")
    print(f"zarr_path: {ZARR_PATH}")
    print(f"n_episodes: {n_episodes}")
    print(f"train episodes: {len(train_indices)}")
    print(f"split episodes: {len(split_indices)}")
    print(f"raw val episodes before train downsampling: {int(np.sum(masks['raw_val']))}")
    print(f"max_steps: {MAX_STEPS}")

    eval_datas = [
        run_split_curve("train", train_indices, state_array, episode_ends, MAX_STEPS),
        run_split_curve("split", split_indices, state_array, episode_ends, MAX_STEPS),
    ]

    log_data = save_outputs(eval_datas, OUTPUT_DIR)

    print("\n=== Summary ===")
    for key, value in sorted(log_data.items()):
        if isinstance(value, (int, float)) and (
            key.endswith("mean_score")
            or "success_rate" in key
            or key.endswith("auc_mean_score")
        ):
            print(f"{key}: {value:.6f}")

    print(f"Saved JSON: {pathlib.Path(OUTPUT_DIR) / 'human_eval_log.json'}")
    print(f"Saved curves: {pathlib.Path(OUTPUT_DIR) / 'human_reward_curves_train_split.npz'}")
    print(f"Saved plot: {pathlib.Path(OUTPUT_DIR) / 'human_reward_curve_train_split.png'}")


if __name__ == "__main__":
    main()
