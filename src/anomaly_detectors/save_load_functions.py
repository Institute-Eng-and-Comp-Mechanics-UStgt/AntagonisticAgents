from __future__ import annotations

from typing import Dict

import torch
import yaml


def save_model(
    config: Dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    filepath: str = "model",
) -> None:
    """
    Save the model state dict as well as additional relevant information.

        Parameters
        ----------
        config:
            Contains information about hyperparameters and architecture.
        model:
            Model with state dict to be saved.
        optimizer:
            Save state dict of the optimizer used to train the model.
        loss:
            Final training loss.
        filepath:
            Path and filename the model is saved to.
    """
    model_dict = {}
    model_dict["config"] = config
    model_dict["loss"] = loss
    model_dict["model_class"] = model.__class__.__name__
    model_dict["model_state_dict"] = model.state_dict()
    model_dict["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(model_dict, filepath)
    return


def load_model(
    model_name, model_class=None, run=None, model_version=None, device="cpu"
):

    if run:
        project = "desrob_coverage_problem"
        path = f"iwenger/{project}/{model_name}:{model_version}"
        art = run.use_artifact(path, type="model")
        art_dir = art.download()
        filename = art_dir + f"/{model_name}.tar"
    else:
        filename = model_name

    model_cfg = torch.load(filename, map_location=torch.device(device))

    if model_class is None:
        model_class = model_cfg["model_class"]

    model = model_class(**model_cfg["config"])
    model.load_state_dict(model_cfg["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg["config"]["learning_rate"]
    )
    optimizer.load_state_dict(model_cfg["optimizer_state_dict"])

    return model_cfg["config"], model, optimizer


def read_config(path: str) -> Dict:
    """
    Read in config.yaml file.

        Parameters
        ----------
        path:
            Specifies the path to the config file to be read.

        Returns
        -------
        config:
            Content of config file.
    """
    config = {}

    with open(path) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config
