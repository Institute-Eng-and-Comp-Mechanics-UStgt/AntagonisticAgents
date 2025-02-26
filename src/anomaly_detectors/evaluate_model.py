import os

import numpy as np
import torch
import wandb
from pandas import DataFrame

from src.anomaly_detectors.anomaly_detector import Coverage_Anomaly_Detector
from src.anomaly_detectors.datasets import CustomDatasetTypedDistances, prepare_data
from src.anomaly_detectors.models.lstm_coupling_nf import ConditionalLSTMCouplingFlow
from src.anomaly_detectors.save_load_functions import load_model
from src.robots.robot_helpers import load_robot_swarm
from src.visualize.plot_functions import (
    plot_action_density,
    plot_noise,
    plot_robot_swarm_run,
)


def run_evaluation(model_filepath, df, swarm_filepath):

    model_cfg, model, _ = load_model(
        model_name=model_filepath,
        model_class=ConditionalLSTMCouplingFlow,
    )
    detector = Coverage_Anomaly_Detector(model, model_cfg)

    _, val_dataloader, test_dataloader, _, val_dataset, test_dataset = prepare_data(
        df,
        batch_size=model_cfg["batch_size"],
        dataset_class=CustomDatasetTypedDistances,
        train_perc=0.7,
        val_perc=0.25,
        test_perc=0.05,
        weighted=False,
        **{"config": model_cfg},
    )
    assert type(val_dataset) == CustomDatasetTypedDistances
    assert type(test_dataset) == CustomDatasetTypedDistances
    print("length datasets: ", val_dataset.__len__(), test_dataset.__len__())
    plot_idx_val = np.arange(val_dataset.run_min, val_dataset.run_min + 2)
    plot_idx_test = np.arange(test_dataset.run_min, test_dataset.run_min + 4)
    plot_idx = np.hstack((plot_idx_val, plot_idx_test))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "number of trainable parameters: ",
        num_params,
    )

    val_loss = test_performance(test_dataloader=val_dataloader, trained_model=model)
    print(
        "validation loss: ",
        val_loss,
    )
    val_losses_per_run = eval_loss_per_run(detector, val_dataset, swarm_filepath)

    thresh = [0.01, 0.05, 0.1]
    val_confs, _ = compute_conf(val_dataloader, model, n_samples=500)
    val_confs_smaller_tresh = {}
    for t in thresh:
        val_confs_smaller_tresh[t] = np.round(
            100 * np.sum(val_confs < t) / val_confs.shape[0], 2
        )
        print(
            f"{val_confs_smaller_tresh[t]}% of actions within smallest {t*100}% of sampled logprobs"
        )

    test_loss = test_performance(test_dataloader=test_dataloader, trained_model=model)
    print("test loss: ", test_loss)
    test_losses_per_run = eval_loss_per_run(detector, test_dataset, swarm_filepath)

    test_confs, _ = compute_conf(test_dataloader, model, n_samples=500)
    test_confs_smaller_tresh = {}
    for t in thresh:
        test_confs_smaller_tresh[t] = np.round(
            100 * np.sum(test_confs < t) / test_confs.shape[0], 2
        )
        print(
            f"{test_confs_smaller_tresh[t]}% of actions within smallest {t*100}% of sampled logprobs"
        )

    log_dict = {
        "data": wandb.Table(dataframe=test_dataset.df),
        "num_params": num_params,
        "neg_val_log_prob": val_loss,
        "neg_val_log_prob_per_run_mean": np.mean(val_losses_per_run),
        "neg_val_log_prob_per_run_std": np.std(val_losses_per_run),
        "neg_test_loss": test_loss,
        "neg_test_loss_per_run_mean": np.mean(test_losses_per_run),
        "neg_test_loss_per_run_std": np.std(test_losses_per_run),
        "threshold": thresh,
        "val_confs_smaller_tresh": val_confs_smaller_tresh,
        "test_confs_smaller_tresh": test_confs_smaller_tresh,
    }

    img = wandb.Image(plot_noise(model, val_dataloader, plot=False))
    log_dict["validation_noise_dist"] = img
    img = wandb.Image(plot_noise(model, test_dataloader, plot=False))
    log_dict["test_noise_dist"] = img

    plot_idx_val = np.arange(val_dataset.run_min, val_dataset.run_min + 2)
    plot_idx_test = np.arange(test_dataset.run_min, test_dataset.run_min + 4)
    plot_idx = np.hstack((plot_idx_val, plot_idx_test))
    for idx in plot_idx:
        img = wandb.Image(
            plot_robot_swarm_run(load_robot_swarm(swarm_filepath + str(idx)), detector),
        )
        log_dict[f"test_images_run_{idx}"] = img

    for idx in range(min(10, test_dataset.__len__())):
        img = wandb.Image(
            plot_action_density(
                idx,
                test_dataset,
                model,
                max_action=model_cfg["max_action"],
                use_3dim_action=model_cfg["use_3dim_action"],
            )
        )
        log_dict[f"lp_images_test_run_{idx}"] = img

    wandb.log(log_dict)


def test_performance(test_dataloader, trained_model, seed=None):

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    test_losses = []

    with torch.no_grad():

        trained_model.eval()
        for _, (x, pi) in enumerate(test_dataloader):

            loss = torch.mean(-trained_model.log_prob(x, pi))
            test_losses.append(loss.item())

    test_loss = np.mean(test_losses)
    return test_loss


def eval_loss_per_run(
    detector,
    dataset,
    rs_path,
):
    """Does not use repeated features.

    Args:
        detector (_type_): _description_
        dataset (_type_): _description_
        rs_path (str, optional): _description_. Defaults to "../data/exp15/rs_".

    Returns:
        _type_: _description_
    """
    loss_per_run = []
    for idx in range(dataset.run_min, dataset.run_max + 1):
        robot_swarm = load_robot_swarm(rs_path + str(idx))
        robot_swarm.set_anomaly_detector(detector)
        assert robot_swarm.anomaly_detector is not None
        log_prob = robot_swarm.anomaly_detector.evaluate_run()
        loss_per_run.append(-1 * log_prob)
    return loss_per_run


def eval_performance_per_robot_type(anomaly_detector, folder_names, seed):

    success = []
    preds = []
    gt = []
    alps = []

    for folder in folder_names:
        files = [name for name in os.listdir(folder)]  # if name.startswith("rs_")]
        for f in files:
            torch.manual_seed(seed)
            np.random.seed(seed)
            rs = load_robot_swarm(f"{folder}{f}")
            _, _, success_per_robot = rs.run_info(print_run=False)
            success.append(success_per_robot)
            rs.set_anomaly_detector(anomaly_detector)
            assert rs.anomaly_detector is not None
            alp, _ = rs.anomaly_detector.evaluate_run()
            pred = rs.anomaly_detector.get_anomaly_prediction_of_swarm()
            preds.append(pred)
            gt.append([robot.label for robot in rs.swarm_robots])
            alps.append(np.vstack(alp).mean(axis=0))

    success = np.hstack(success)
    preds = np.hstack(preds)
    gt = np.hstack(gt)
    alps = np.hstack(alps)

    result = {}
    for robot_type in np.unique(gt):
        type_mask = gt == robot_type

        result[robot_type] = {
            "size": preds[type_mask].size,
            "alps": alps[type_mask],
            "anomal": np.sum(preds[type_mask]),
            "perc_anomal": np.mean(preds[type_mask]),
            "preds": preds[type_mask],
            "success": success[type_mask],
            "perc_anomal_success": preds[type_mask & (success == True)].mean(),
        }
    return result


def compute_conf(
    dataloader, trained_model, n_samples=100
) -> tuple[np.ndarray, np.ndarray]:

    lps = []
    confs = []

    with torch.no_grad():

        trained_model.eval()
        for _, (x, pi) in enumerate(dataloader):

            lp = trained_model.log_prob(x, pi).reshape(-1, 1)
            _, log_prob = trained_model.sample_and_log_prob(n_samples, pi)

            conf = torch.sum(log_prob < lp, dim=-1).cpu().numpy()
            confs.append(conf)
            lps.append(lp.cpu().numpy())

    confs = np.hstack(confs) / n_samples
    lps = np.vstack(lps)

    return confs, lps
