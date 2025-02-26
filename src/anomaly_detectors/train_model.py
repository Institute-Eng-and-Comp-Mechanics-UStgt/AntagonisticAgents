from typing import Dict

import numpy as np
import pandas as pd
import torch
import wandb
import wandb.wandb_run
from src.anomaly_detectors.anomaly_detector import Coverage_Anomaly_Detector
from src.anomaly_detectors.datasets import CustomDatasetTypedDistances, prepare_data
from src.anomaly_detectors.evaluate_model import eval_loss_per_run
from src.anomaly_detectors.save_load_functions import save_model
from src.anomaly_detectors.train_model import train_model
from src.robots.robot_helpers import load_robot_swarm
from src.visualize.plot_functions import plot_noise, plot_robot_swarm_run
from torch import optim


def train_model(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    watch=False,
    n_epochs=1000,
    run=None,
    cfg={},
    improvement_threshold=10,
):

    if watch:
        wandb.watch(model, log="all", log_freq=50)

    epoch_info = {}
    epoch_info["best_train_loss"] = np.inf
    epoch_info["epoch_loss_train"] = [np.inf]
    epoch_info["best_val_loss"] = np.inf
    epoch_info["epoch_loss_val"] = [np.inf]
    epoch_info["steps_since_best_val_loss"] = 0
    epoch_info["epoch"] = 0

    while (
        not early_stop(
            epoch_info, print_loss=True, improvement_threshold=improvement_threshold
        )
        and epoch_info["epoch"] < n_epochs
    ):

        epoch_train_losses = []
        model.train()
        for _, (x, pi) in enumerate(train_dataloader):

            optimizer.zero_grad()

            log_prob = model.log_prob(
                x,
                context=pi,
            )
            loss = torch.mean(-log_prob)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        epoch_val_losses = []
        with torch.no_grad():
            model.eval()
            for _, (x, pi) in enumerate(val_dataloader):

                log_prob = model.log_prob(
                    x,
                    context=pi,
                )
                loss = torch.mean(-log_prob)
                epoch_val_losses.append(loss.item())

        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_val_loss = np.mean(epoch_val_losses)
        epoch_info["epoch_loss_train"] += [epoch_train_loss]
        epoch_info["epoch_loss_val"] += [epoch_val_loss]
        epoch_info["epoch"] += 1

        if run:
            run.log(
                {
                    "neg_train_log_prob": epoch_train_loss,
                    "neg_val_log_prob": epoch_val_loss,
                }
            )

    return epoch_info


def early_stop(
    epoch_info: Dict,
    improvement_threshold: int = 10,
    conv_epsilon: float = 1e-4,
    conv_threshold: int = 5,
    print_loss: bool = False,
) -> bool:
    """
    Check for validation loss improvement and convergence.

        Parameters
        ----------
        epoch_info:
            Contains the epoch, the best_train_loss and best_val_loss as well as a list of
            epoch_loss_train and epoch_loss_val.
        improvement_threshold:
            An early stop is triggered if the validation loss did not improve over the
            last improvement_threshold epochs.
        conv_epsilon:
            The convergence criterium between two validation losses is fulfilled if their absolute distance
            falls below the conv_epsilon value.
        conv_threshold:
            If the convergence criterium is fulfilled for the last conv_threshold validation losses,
            an early stop is triggered.
        print_loss:
            Print training and validation loss after each epoch.

        Returns
        -------
        early_stop:
            Stops the training if true.
    """

    # Print training and validation error.
    train_sign = " (-)"
    if epoch_info["epoch_loss_train"][-1] < epoch_info["best_train_loss"]:
        epoch_info["best_train_loss"] = epoch_info["epoch_loss_train"][-1]
        train_sign = " (+)"

    val_sign = " (-)"
    if epoch_info["epoch_loss_val"][-1] < epoch_info["best_val_loss"]:
        epoch_info["best_val_loss"] = epoch_info["epoch_loss_val"][-1]
        val_sign = " (+)"
        epoch_info["steps_since_best_val_loss"] = 0
    else:
        epoch_info["steps_since_best_val_loss"] += 1

    if print_loss:
        print(
            "Epoch ",
            epoch_info["epoch"],
            "  Train error: ",
            round(epoch_info["epoch_loss_train"][-1], 8),
            train_sign,
            "  Validation error: ",
            round(epoch_info["epoch_loss_val"][-1], 8),
            val_sign,
        )

    # Check for validation loss improvement during the last improvement_threshold epochs
    if (
        epoch_info["steps_since_best_val_loss"] > improvement_threshold
        and epoch_info["epoch_loss_val"][-1] != np.inf
    ):
        no_improvement_visible = True
        print("no improvement!")
        return no_improvement_visible

    # Check for convergence
    converged = False
    if len(epoch_info["epoch_loss_val"]) > conv_threshold:
        converged = True
        # check if the last validation epoch loss is closer than conv_epsilon to the last
        # conv_threshold validation epoch losses
        for i in range(2, conv_threshold):
            if (
                np.abs(
                    epoch_info["epoch_loss_val"][-i] - epoch_info["epoch_loss_val"][-1]
                )
                > conv_epsilon
            ):
                converged = False
        if converged:
            print(epoch_info["epoch_loss_val"][-20:])
            print("converged!")
    return converged


def add_epsilon(x, cfg):

    epsilon = torch.randn_like(x) * 0.25 + 1
    sign_neg_idx = torch.rand_like(x) > 0.5
    epsilon[sign_neg_idx] *= -1
    x = torch.clamp(x + epsilon, min=-cfg["max_action"], max=cfg["max_action"])
    weights = 0.1 * torch.sqrt(torch.sum(epsilon**2, dim=-1))  # mean 0.14

    return x, weights[:, None]


def initialize_sweep_training(
    config,
    sweep_name: str,
    df: pd.DataFrame,
    model_class: torch.nn.Module,
    dataset_class: torch.utils.data.Dataset,
    swarm_filepath: str,
    detector: Coverage_Anomaly_Detector,
    watch: bool = True,
):

    def wandb_train(
        sweep_name: str = sweep_name,
        df: pd.DataFrame = df,
        config: Dict = config,
        model_class: torch.nn.Module = model_class,
        dataset_class: torch.utils.data.Dataset = dataset_class,
        swarm_filepath: str = swarm_filepath,
        detector: Coverage_Anomaly_Detector = detector,
        watch: bool = watch,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run = wandb.init(
            group=config["group"],
            job_type=config["job_type"],
            tags=config["tags"],
            config=config,
        )
        assert run is not None

        # initialize
        model_config = wandb.config
        torch.manual_seed(model_config.seed)
        np.random.seed(model_config.seed)
        model_cfg = dict(model_config)
        model_cfg.update(config)

        train_dataloader, val_dataloader, *_ = prepare_data(
            df,
            batch_size=model_cfg["batch_size"],
            dataset_class=dataset_class,
            train_perc=0.7,
            val_perc=0.25,
            test_perc=0.05,
            weighted=model_cfg["weighted"],
            **{"config": model_cfg},
        )
        model = model_class(cfg=model_cfg, **model_cfg)
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            "number of trainable parameters: ",
            n_params,
        )

        optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)

        # train
        epoch_info = train_model(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            watch=watch,
            run=run,
            cfg=model_cfg,
            improvement_threshold=25,
        )

        # log data to wandb
        val_loss_per_run = np.mean(
            eval_loss_per_run(detector, val_dataloader.dataset, swarm_filepath)
        )

        if epoch_info["best_val_loss"] < 2.5:

            # save model
            filepath = f"../trained_models/{sweep_name}/" + run.name
            save_model(
                model_cfg,
                model=model,
                optimizer=optimizer,
                loss=epoch_info["epoch_loss_val"][-1],
                filepath=filepath,
            )

            # wandb.log_model(filepath)
            rs0 = load_robot_swarm(swarm_filepath + str(0))
            rs1 = load_robot_swarm(swarm_filepath + str(5))
            assert type(val_dataloader.dataset) == CustomDatasetTypedDistances
            rs2 = load_robot_swarm(swarm_filepath + str(val_dataloader.dataset.run_min))
            wandb.log(
                {
                    "n_params": n_params,
                    "neg_val_log_prob_per_run": val_loss_per_run,
                    "noise dist": wandb.Image(
                        plot_noise(model, train_dataloader, plot=False)
                    ),
                    "train_run_0": wandb.Image(plot_robot_swarm_run(rs0, detector)),
                    "train_run_1": wandb.Image(plot_robot_swarm_run(rs1, detector)),
                    "train_run_2": wandb.Image(plot_robot_swarm_run(rs2, detector)),
                }
            )
        else:
            wandb.log(
                {
                    "n_params": n_params,
                    "neg_val_log_prob_per_run": val_loss_per_run,
                }
            )

    return wandb_train
