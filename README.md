# Clean Diffusion Policy: Push-T Low-Dim Baselines

This is a trimmed version of the public `diffusion_policy` repository. It keeps only the **Push-T low-dimensional** experiment and two baselines:

- Diffusion Policy: `train_diffusion_unet_lowdim_workspace.yaml`
- IBC DFO: `train_ibc_dfo_lowdim_workspace.yaml`

## Environment: Python 3.12

```bash
conda env create -f environment_py312.yaml
conda activate dp-pusht-py312
pip install -e .
```

If the PyTorch/CUDA line in the environment file does not match your machine, create the env with Python 3.12, install the right PyTorch build, then run:

```bash
pip install -r requirements_py312.txt
pip install -e .
```

## Dataset

The configs expect the official Push-T replay buffer here:

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

Original data command:

```bash
mkdir -p data
cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip
rm pusht.zip
cd ..
```

## Train

Diffusion Policy:

```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace training.device=cuda:0
python train.py --config-name=train_diffusion_unet_image_workspace training.device=cuda:0
```

IBC:

```bash
python train.py --config-name=train_ibc_dfo_lowdim_workspace training.device=cuda:0
python train.py --config-name=train_ibc_dfo_image_workspace training.device=cuda:0
```

Useful quick smoke-test overrides:

```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace training.device=cpu training.num_epochs=1 training.max_train_steps=2 training.max_val_steps=2 training.rollout_every=999999 training.checkpoint_every=1 dataloader.num_workers=0 val_dataloader.num_workers=0
```

## Eval

```bash
python eval.py --checkpoint path/to/latest.ckpt --output_dir data/eval_dp --device cuda:0
```

`eval.py` was patched so IBC checkpoints work even though IBC does not define `training.use_ema`.


## Push-T image setting

This clean package now also keeps the Push-T image setting for both baselines:

```bash
# Diffusion Policy, image observation
python train.py --config-name=train_diffusion_unet_image_workspace training.device=cuda:0

# IBC, image + low-dim hybrid observation
python train.py --config-name=train_ibc_dfo_hybrid_workspace training.device=cuda:0
```

Both configs default to `task: pusht_image`. The image dataset path is still the original one:

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

Evaluation is the same entry point:

```bash
python eval.py --checkpoint <path-to-checkpoint.ckpt> --output_dir <eval-output-dir>
```

Note: the IBC image baseline depends on `robomimic`, because the original implementation reuses robomimic's image encoder stack.
