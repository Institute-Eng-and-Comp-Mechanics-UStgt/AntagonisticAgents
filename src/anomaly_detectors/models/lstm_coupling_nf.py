from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
import torch.nn.functional as F
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms import (
    BatchNorm,
    CompositeTransform,
    MaskedPiecewiseQuadraticAutoregressiveTransform,
    PiecewiseCubicCouplingTransform,
    PiecewiseLinearCouplingTransform,
    PiecewiseQuadraticCouplingTransform,
    ReversePermutation,
)
from torch.nn import LSTM

from src.anomaly_detectors.models.distributions import Uniform

if TYPE_CHECKING:
    from src.deployment_area.polygon import PolygonWrapper


class ConditionalLSTMCouplingFlow(Flow):

    def __init__(
        self,
        num_layers,
        use_3dim_action,
        coupling_transform,
        embedding_net_fn_typed=True,
        max_action=1.2,
        transform_hidden_size=50,
        num_bins=10,
        **kwargs,
    ):

        if embedding_net_fn_typed:
            embedding_net_fn = EmbeddingNetTypedLSTM
        else:
            embedding_net_fn = EmbeddingNet2LSTMs
        embedding_net = embedding_net_fn(**kwargs)
        self.use_3dim_action = use_3dim_action

        transform_net_create_fn = (
            lambda n_identity_features, n_transform_features: TransformNet(
                n_identity_features,
                input_size=embedding_net.embedding_size,
                hidden_size=transform_hidden_size,
                output_size=n_transform_features,
            )
        )  # output size is n_transform_features * self._transform_dim_multiplier()
        # with n_transform_features: # mask > 0, i.e. 2 in this case

        if self.use_3dim_action:
            dist_bound = torch.ones(3) + 1e-4
            mask = torch.ones(3)
        else:
            dist_bound = max_action * torch.ones(2) + 1e-4
            mask = torch.ones(2)

        layers = []
        if coupling_transform == 1 or coupling_transform == "piecewise_linear":
            coupling_transform = PiecewiseLinearCouplingTransform
        elif coupling_transform == 2 or coupling_transform == "piecewise_quadratic":
            coupling_transform = PiecewiseQuadraticCouplingTransform
        elif coupling_transform == 3 or coupling_transform == "piecewise_cubic":
            coupling_transform = PiecewiseCubicCouplingTransform
        for _ in range(num_layers):
            layers.append(
                coupling_transform(
                    mask=mask,
                    transform_net_create_fn=transform_net_create_fn,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=1 + 1e-5,
                )
            )
            # not using a batch norm layer since it causes the inputs to be outside of the support of the uniform distribution

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=Uniform(
                low=-1 * dist_bound,
                high=dist_bound,
            ),
            embedding_net=embedding_net,
        )

    def log_prob(self, inputs, context=None):
        if context is not None:
            context = context.to(self.device)
        return super()._log_prob(inputs.to(self.device), context)

    def sample_and_log_prob(self, num_samples, context=None):
        if context is not None:
            context = context.to(self.device)
        return super().sample_and_log_prob(num_samples, context)

    def sample(self, num_samples, context=None, batch_size=None):
        if context is not None:
            context = context.to(self.device)
        return super()._sample(num_samples, context)

    def transform_to_noise(self, inputs, context=None):
        if context is not None:
            context = context.to(self.device)
        return super().transform_to_noise(inputs.to(self.device), context)

    def to(self, device, *args, **kwargs):
        self.device = device
        self._transform.to(device)
        self._embedding_net.to(device)
        self._distribution.to(device)
        # self._distribution.set_device(device)
        return self

    def get_uniform_log_prob(self):

        if self.use_3dim_action:
            return self._distribution.log_prob(torch.zeros(1, 3).to(self.device))
        else:
            return self._distribution.log_prob(torch.zeros(1, 2).to(self.device))


class TransformNet(torch.nn.Module):

    def __init__(
        self,
        n_identity_features,
        input_size,
        hidden_size,
        output_size,
    ):
        super().__init__()

        self.transform1 = torch.nn.Linear(
            in_features=input_size + n_identity_features, out_features=hidden_size
        )
        self.transform3 = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size
        )

    def forward(self, identity_features, context):

        context = torch.cat([identity_features, context], dim=-1)
        context = F.relu(self.transform1(context))
        context = F.tanh(self.transform3(context))
        return context


class EmbeddingNet2LSTMs(torch.nn.Module):

    def __init__(
        self,
        hidden_size_neighbors,
        hidden_size_poly,
        n_lstm_layers,
        embedding_size,
        bidirectional,
        cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()

        if cfg:
            self.cfg = cfg
        else:
            self.cfg = kwargs

        self.embedding_size = embedding_size
        self.n_lstm_layers = n_lstm_layers
        self.bidirectional = bidirectional
        input_dim = 3

        self.n_neighbours = int(
            len(self.cfg["context_neighbors"]) / self.cfg["env_dim"]
        )

        self.lstm_neighbours = LSTM(
            input_size=input_dim,
            hidden_size=hidden_size_neighbors,
            num_layers=n_lstm_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm_polygon = LSTM(
            input_size=input_dim,
            hidden_size=hidden_size_poly,
            num_layers=n_lstm_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if bidirectional:
            input_modifier = 2
        else:
            input_modifier = 1

        self.output_layer = torch.nn.Linear(
            in_features=hidden_size_neighbors * input_modifier
            + hidden_size_poly * input_modifier,
            out_features=embedding_size,
        )

    def forward(self, context: torch.Tensor):

        context_n = context[:, : self.n_neighbours, :]
        context_p = context[:, self.n_neighbours :, :]

        if self.cfg["shuffle_context"]:
            context_n = context_n[:, torch.randperm(context_n.shape[1])]
            context_p = context_p[:, torch.randperm(context_p.shape[1])]

        _, (h_n, _) = self.lstm_neighbours(context_n)
        # use all polygon edges
        _, (h_p, _) = self.lstm_polygon(context_p)

        if self.bidirectional:
            # concatenate the last hidden state in the forward and reverse direction
            forward_h_n = h_n[-2]
            reverse_h_n = h_n[-1]
            context_embedding_n = torch.hstack((forward_h_n, reverse_h_n))
            forward_h_p = h_p[-2]
            reverse_h_p = h_p[-1]
            context_embedding_p = torch.hstack((forward_h_p, reverse_h_p))
            context_embedding = torch.concat(
                (context_embedding_n, context_embedding_p), dim=-1
            )

        else:
            # concatenate the hidden states (of the last layer) of each LSTM
            context_embedding = torch.concat((h_n[-1], h_p[-1]), dim=-1)

        context_embedding = torch.nn.functional.relu(
            self.output_layer(context_embedding)
        )

        return context_embedding


class EmbeddingNetTypedLSTM(torch.nn.Module):

    def __init__(
        self,
        hidden_size_lstm,
        n_lstm_layers,
        embedding_size,
        one_hot_type,
        bidirectional,
        cfg=None,
        **kwargs,
    ):
        super().__init__()
        if cfg:
            self.cfg = cfg
        else:
            self.cfg = kwargs
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional

        if one_hot_type:
            input_dim = 5
        else:
            input_dim = 4

        self.lstm = LSTM(
            input_size=input_dim,
            hidden_size=hidden_size_lstm,
            num_layers=n_lstm_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if bidirectional:
            input_modifier = 2
        else:
            input_modifier = 1

        self.output_layer = torch.nn.Linear(
            in_features=hidden_size_lstm * input_modifier,
            out_features=embedding_size,
        )

    def forward(self, context):

        if self.cfg["shuffle_context"]:
            context = context[:, torch.randperm(context.shape[1])]

        _, (h, _) = self.lstm(context)

        if self.bidirectional:
            # concatenate the last hidden state in the forward and reverse direction
            forward_h = h[-2]
            reverse_h = h[-1]
            context_embedding = torch.hstack((forward_h, reverse_h))
        else:
            context_embedding = h[-1]

        context_embedding = torch.nn.functional.relu(
            self.output_layer(context_embedding)
        )

        return context_embedding
