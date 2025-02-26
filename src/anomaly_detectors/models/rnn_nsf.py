from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
from nflows.flows.base import Flow
from torch.nn import LSTM
from torch.nn import functional as F

from src.anomaly_detectors.models.conditional_maf import (
    ConditionalMaskedAutoregressiveFlow,
)

if TYPE_CHECKING:
    from src.deployment_area.polygon import PolygonWrapper


class RNN_NSF(torch.nn.Module):
    def __init__(self, cfg=None, **kwargs) -> None:
        super().__init__()

        if cfg:
            self.cfg = cfg
        else:
            self.cfg = kwargs

        self.cfg["context_features"] = (
            self.cfg["hidden_size_neighbors"] + self.cfg["hidden_size_poly"]
        )
        self.min_neighbors = self.cfg["min_neighbors"]
        self.max_neighbors = self.cfg["max_neighbors"]

        self.features = self.cfg["features"]
        self.n_neighbours = int(
            len(self.cfg["context_neighbors"]) / self.cfg["env_dim"]
        )

        self.lstm_neighbours = LSTM(
            input_size=self.cfg["feature_dim"],
            hidden_size=self.cfg["hidden_size_neighbors"],
            num_layers=self.cfg["lstm_layers"],
            bias=True,
            batch_first=True,
        )
        self.lstm_polygon = LSTM(
            input_size=self.cfg["feature_dim"],
            hidden_size=self.cfg["hidden_size_poly"],
            num_layers=self.cfg["lstm_layers"],
            bias=True,
            batch_first=True,
        )

        self.nflow = ConditionalMaskedAutoregressiveFlow(**self.cfg)
        self.rnn_nsf = torch.nn.ModuleList(
            [self.nflow, self.lstm_neighbours, self.lstm_polygon]
        )

    def embed_context(self, context: dict):
        pi_n = context["neighbour"]
        pi_p = context["poly"]

        if self.cfg["shuffle_context"]:
            pi_n = pi_n[:, torch.randperm(pi_n.shape[1])]
            pi_p = pi_p[:, torch.randperm(pi_p.shape[1])]

        _, (h_n, _) = self.lstm_neighbours(pi_n.to(self.device))
        _, (h_p, _) = self.lstm_polygon(pi_p.to(self.device))

        # concatenate the hidden states (of the last layer) of each LSTM
        if self.cfg["lstm_layers"] > 1:
            context_embedding = torch.concat((h_n[-1], h_p[-1]), dim=-1)

        else:
            context_embedding = torch.concat((h_n, h_p), dim=-1)

        # return a tensor with shape (batch_size, embedding_size)
        if len(context_embedding.shape) > 2:
            context_embedding = context_embedding.squeeze(dim=0)

        return context_embedding

    def log_prob(self, x, context):
        context_embedding = self.embed_context(context)

        if context_embedding.shape[0] == 1:
            context_embedding = torch.repeat_interleave(
                context_embedding, repeats=x.shape[0], dim=0
            )
        log_prob = self.nflow.log_prob(x.to(self.device), context_embedding)

        return log_prob

    def sample(self, num_samples, context):
        context_embedding = self.embed_context(context)
        samples = self.nflow.sample(num_samples, context_embedding)

        return samples

    def sample_and_log_prob(self, num_samples, context):
        context_embedding = self.embed_context(context)
        samples, log_prob = self.nflow.sample_and_log_prob(
            num_samples, context_embedding
        )

        return samples, log_prob

    def transform_to_noise(self, x, context):
        context_embedding = self.embed_context(context)
        noise, logabsdet = self.nflow._transform(x.to(self.device), context_embedding)

        return noise

    def get_uniform_log_prob(self):
        return self.nflow._distribution.log_prob(torch.zeros(1, 2).to(self.device))

    def to(self, device, *args, **kwargs):
        self.device = device
        self.rnn_nsf.to(device)
        self.nflow.to(device)
        self.lstm_neighbours.to(device)
        self.lstm_polygon.to(device)
        self.nflow._distribution.to(device)
        return self
