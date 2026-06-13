#!/usr/bin/env python3
"""Fit a GMM on Push-T lowdim relative action chunks for gmm_flow.

Example:
  python scripts/fit_gmm_flow_gmm.py \
    --n-components 16 \
    --covariance-type diag

The updated policy classifies relative action chunks:
    relative_action[t] = action[t] - current_agent_pos
where current_agent_pos is the last visible lowdim observation position.

The GMM is fitted on normalized relative chunks with a relative-action normalizer
saved in this npz file. The policy then samples a relative source chunk and
converts it back to an absolute action source for the flow head.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import List

# Make the script runnable from repo root without installing extra path hacks.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, default="data/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--covariance-type", type=str, default="diag", choices=["diag", "full"])
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--n-latency-steps", type=int, default=0)
    parser.add_argument("--oa-step-convention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-train-episodes", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--normalizer-mode", type=str, default="limits", choices=["limits", "gaussian"], help="Kept for dataset normalizer compatibility; the GMM itself uses --relative-normalizer-mode.")
    parser.add_argument("--relative-normalizer-mode", type=str, default="limits", choices=["limits", "gaussian"], help="How to normalize raw relative actions before fitting the GMM.")
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    from sklearn.mixture import GaussianMixture
    if args.output is None:
        args.output = f"data/gmm/pusht/action_gmm_K{args.n_components}_diag.npz"

    action_steps = args.n_action_steps + args.n_latency_steps
    pad_before = args.n_obs_steps - 1 + args.n_latency_steps
    pad_after = args.n_action_steps - 1
    start = args.n_obs_steps - 1 if args.oa_step_convention else args.n_obs_steps
    end = start + action_steps

    dataset = PushTLowdimDataset(
        zarr_path=args.zarr_path,
        horizon=args.horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_train_episodes=args.max_train_episodes,
    )
    # Keep the same dataset slicing parameters as training. The GMM feature itself
    # is normalized by rel_action_scale/rel_action_offset below, not by the dataset normalizer.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    seen = 0
    rel_chunks_raw: List[np.ndarray] = []
    print(
        f"Collecting raw relative action chunks: start={start}, end={end}, "
        f"chunk_shape=({action_steps}, 2), dataset_len={len(dataset)}"
    )
    for batch in tqdm(loader):
        # PushTLowdimDataset obs = [keypoints..., agent_x, agent_y].
        # Use the latest visible obs position as the current circular end-effector position.
        current_agent_pos = batch["obs"][:, args.n_obs_steps - 1, -2:]
        action_chunk = batch["action"][:, start:end]
        rel = action_chunk - current_agent_pos[:, None, :]
        rel_np = rel.detach().cpu().numpy().astype(np.float32)
        if args.max_samples is not None and seen + rel_np.shape[0] > args.max_samples:
            rel_np = rel_np[: args.max_samples - seen]
        rel_chunks_raw.append(rel_np)
        seen += rel_np.shape[0]
        if args.max_samples is not None and seen >= args.max_samples:
            break

    rel_raw = np.concatenate(rel_chunks_raw, axis=0)
    rel_flat_for_stats = rel_raw.reshape(-1, 2)
    if args.relative_normalizer_mode == "limits":
        rel_min = rel_flat_for_stats.min(axis=0)
        rel_max = rel_flat_for_stats.max(axis=0)
        rel_range = np.maximum(rel_max - rel_min, 1e-7)
        rel_action_scale = (2.0 / rel_range).astype(np.float32)
        rel_action_offset = (-1.0 - rel_action_scale * rel_min).astype(np.float32)
    else:
        rel_mean = rel_flat_for_stats.mean(axis=0)
        rel_std = np.maximum(rel_flat_for_stats.std(axis=0), 1e-6)
        rel_action_scale = (1.0 / rel_std).astype(np.float32)
        rel_action_offset = (-rel_mean / rel_std).astype(np.float32)

    x_chunks = rel_raw * rel_action_scale.reshape(1, 1, 2) + rel_action_offset.reshape(1, 1, 2)
    x = x_chunks.reshape(x_chunks.shape[0], -1).astype(np.float32)
    if x.shape[0] < args.n_components:
        raise ValueError(f"Need at least K samples, got {x.shape[0]} samples for K={args.n_components}.")
    print(f"Fitting GaussianMixture on {x.shape[0]} relative chunks with dim={x.shape[1]}...")
    print(f"Relative normalizer mode: {args.relative_normalizer_mode}")
    print(f"rel_action_scale={rel_action_scale}, rel_action_offset={rel_action_offset}")

    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=1,
    )
    labels = gmm.fit_predict(x)
    counts = np.bincount(labels, minlength=args.n_components)
    proportions = counts / counts.sum()
    bic = gmm.bic(x)
    aic = gmm.aic(x)

    print("Done.")
    print(f"BIC: {bic:.3f}")
    print(f"AIC: {aic:.3f}")
    print(f"BIC/AIC: {(bic / aic) :.3f}")
    print("Component counts:")
    for idx, (count, prop) in enumerate(zip(counts, proportions)):
        print(f"  {idx:02d}: {count:6d}  ({100.0 * prop:6.2f}%)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        means=gmm.means_.astype(np.float32),
        covariances=gmm.covariances_.astype(np.float32),
        weights=gmm.weights_.astype(np.float32),
        covariance_type=np.array(args.covariance_type),
        n_components=np.array(args.n_components, dtype=np.int64),
        horizon=np.array(args.horizon, dtype=np.int64),
        n_obs_steps=np.array(args.n_obs_steps, dtype=np.int64),
        n_action_steps=np.array(action_steps, dtype=np.int64),
        action_dim=np.array(2, dtype=np.int64),
        oa_step_convention=np.array(args.oa_step_convention),
        chunk_start=np.array(start, dtype=np.int64),
        chunk_end=np.array(end, dtype=np.int64),
        normalizer_mode=np.array(args.normalizer_mode),
        gmm_feature_space=np.array("relative_action_minus_current_agent_pos"),
        relative_normalizer_mode=np.array(args.relative_normalizer_mode),
        rel_action_scale=rel_action_scale.astype(np.float32),
        rel_action_offset=rel_action_offset.astype(np.float32),
        rel_action_min=rel_flat_for_stats.min(axis=0).astype(np.float32),
        rel_action_max=rel_flat_for_stats.max(axis=0).astype(np.float32),
        rel_action_mean=rel_flat_for_stats.mean(axis=0).astype(np.float32),
        rel_action_std=rel_flat_for_stats.std(axis=0).astype(np.float32),
        seed=np.array(args.seed, dtype=np.int64),
        bic=np.array(bic, dtype=np.float64),
        aic=np.array(aic, dtype=np.float64),
        counts=counts.astype(np.int64),
        proportions=proportions.astype(np.float32),
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
