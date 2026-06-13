if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDirectionMagMLPLowdimWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def _loss_result_to_log(loss_result, prefix: str):
        """Convert policy.compute_loss output into scalar loss tensor + log dict."""
        if isinstance(loss_result, torch.Tensor):
            return loss_result, {prefix: float(loss_result.detach().cpu())}

        loss = loss_result["loss"]
        log = {prefix: float(loss.detach().cpu())}
        for key, value in loss_result.items():
            if key == "loss":
                continue
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            log[f"{prefix}_{key}"] = value
        return loss, log

    @staticmethod
    def _mean_logs(logs):
        if len(logs) == 0:
            return dict()
        result = dict()
        keys = logs[0].keys()
        for key in keys:
            result[key] = float(np.mean([x[key] for x in logs]))
        return result

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner: BaseLowdimRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseLowdimRunner)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                # ========= train ==========
                train_step_logs = []
                self.model.train()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        loss_result = self.model.compute_loss(batch)
                        raw_loss, loss_log = self._loss_result_to_log(loss_result, "train_loss")
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        postfix = {"loss": loss_log["train_loss"]}
                        if "train_loss_dir_loss" in loss_log:
                            postfix.update(
                                dir=loss_log["train_loss_dir_loss"],
                                mag=loss_log["train_loss_mag_loss"],
                                bias=loss_log.get("train_loss_bias_loss", 0.0),
                                traj=loss_log["train_loss_traj_loss"],
                            )
                        tepoch.set_postfix(**postfix, refresh=False)

                        train_step_logs.append(loss_log)
                        step_log = {
                            **loss_log,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            break

                # epoch-average train losses replace the last minibatch losses
                step_log.update(self._mean_logs(train_step_logs))

                # ========= eval ==========
                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_step_logs = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_result = self.model.compute_loss(batch)
                                _, loss_log = self._loss_result_to_log(loss_result, "val_loss")
                                val_step_logs.append(loss_log)
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break
                        step_log.update(self._mean_logs(val_step_logs))

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = {"obs": batch["obs"]}
                        gt_action = batch["action"]
                        result = policy.predict_action(obs_dict)
                        pred_action = result["action"]
                        pred_action_full = result.get("action_pred", pred_action)

                        start = cfg.n_obs_steps - 1
                        exec_end = start + cfg.n_action_steps
                        pred_end = start + cfg.policy.get("pred_action_steps", cfg.n_action_steps)

                        gt_action_exec = gt_action[:, start:exec_end]
                        gt_action_full = gt_action[:, start:pred_end]

                        exec_mse = torch.nn.functional.mse_loss(pred_action, gt_action_exec)
                        full_mse = torch.nn.functional.mse_loss(pred_action_full, gt_action_full)
                        step_log["train_action_mse_error"] = exec_mse.item()
                        step_log["train_action_pred_full_mse_error"] = full_mse.item()
                        del batch, obs_dict, gt_action, gt_action_exec, gt_action_full, result, pred_action, pred_action_full, exec_mse, full_mse

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        metric_dict[key.replace("/", "_")] = value
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                self.model.train()
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDirectionMagMLPLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
