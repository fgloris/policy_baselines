#!/usr/bin/env python3
"""Fit a GMM on normalized Push-T lowdim action chunks for gmm_flow.

Example:
  python scripts/fit_gmm_flow_gmm.py \
    --n-components 16 \
    --covariance-type diag

The policy expects the GMM to be fitted in the same normalized action space used
by the training dataset normalizer, and on the same action chunk slice.
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

from diffusion_policy.common.pytorch_util import dict_apply
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
    parser.add_argument("--normalizer-mode", type=str, default="limits", choices=["limits", "gaussian"])
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
    normalizer = dataset.get_normalizer(mode=args.normalizer_mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    chunks: List[np.ndarray] = []
    seen = 0
    print(
        f"Collecting normalized action chunks: start={start}, end={end}, "
        f"chunk_shape=({action_steps}, 2), dataset_len={len(dataset)}"
    )
    for batch in tqdm(loader):
        nbatch = normalizer.normalize(batch)
        action = nbatch["action"][:, start:end]
        flat = action.reshape(action.shape[0], -1).detach().cpu().numpy().astype(np.float32)
        if args.max_samples is not None and seen + flat.shape[0] > args.max_samples:
            flat = flat[: args.max_samples - seen]
        chunks.append(flat)
        seen += flat.shape[0]
        if args.max_samples is not None and seen >= args.max_samples:
            break

    x = np.concatenate(chunks, axis=0)
    if x.shape[0] < args.n_components:
        raise ValueError(f"Need at least K samples, got {x.shape[0]} samples for K={args.n_components}.")
    print(f"Fitting GaussianMixture on {x.shape[0]} chunks with dim={x.shape[1]}...")

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
        seed=np.array(args.seed, dtype=np.int64),
        bic=np.array(bic, dtype=np.float64),
        aic=np.array(aic, dtype=np.float64),
        counts=counts.astype(np.int64),
        proportions=proportions.astype(np.float32),
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
