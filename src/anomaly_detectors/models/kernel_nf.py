from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
import torch.nn.functional as F
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms import (
    AffineCouplingTransform,
    BatchNorm,
    CompositeTransform,
    MaskedPiecewiseQuadraticAutoregressiveTransform,
    PiecewiseCubicCouplingTransform,
    PiecewiseLinearCouplingTransform,
    PiecewiseQuadraticCouplingTransform,
    RandomPermutation,
    ReversePermutation,
)
from torch.nn import AdaptiveMaxPool2d, Conv2d, Conv3d, ConvTranspose3d, MaxPool2d

from src.anomaly_detectors.models.distributions import Uniform

if TYPE_CHECKING:
    from src.anomaly_detectors.models.kernel_nf import AnchorMaxPoolEmbeddingNet


class ConditionalKernelCouplingFlow(Flow):

    def __init__(
        self,
        features,
        bound,
        num_layers,
        transform_hidden_size=50,
        num_bins=10,
        batch_norm_between_layers=False,
        device="cpu",
        **kwargs,
    ):
        # self.n_features = features
        # self.n_context = context
        embedding_net = AnchorMaxPoolEmbeddingNet(**kwargs)

        transform_net_create_fn = (
            lambda n_identity_features, n_transform_features: TransformNet(
                n_identity_features,
                output_size=n_transform_features,
                hidden_size=transform_hidden_size,
                input_size=embedding_net.embedding_size,
            )
        )  # output size is n_transform_features * self._transform_dim_multiplier()
        # with n_transform_features: # mask > 0, i.e. 2 in this case

        layers = []
        for _ in range(num_layers):
            # layers.append(ReversePermutation(features))
            layers.append(
                PiecewiseCubicCouplingTransform(
                    mask=[1, 1],
                    transform_net_create_fn=transform_net_create_fn,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=bound,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            # distribution=StandardNormal(shape=[features]),
            distribution=Uniform(
                low=torch.tensor([-bound, -bound]),
                high=torch.tensor([bound, bound]),
            ),
            embedding_net=embedding_net,
        )

        self.set_device(device)

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
        return super().sample(num_samples, context, batch_size=None)

    def transform_to_noise(self, inputs, context=None):
        if context is not None:
            context = context.to(self.device)
        return super().transform_to_noise(inputs.to(self.device), context)

    def set_device(self, device):
        self.device = device
        self._transform.to(device)
        self._embedding_net.to(device)
        self._distribution.to(device)
        self._distribution.set_device(device)

    def get_uniform_log_prob(self):
        return self._distribution.log_prob(torch.zeros(1, 2).to(self.device))


class TransformNet(torch.nn.Module):
    def __init__(self, n_identity_features, output_size, hidden_size, input_size):
        super().__init__()

        self.transform1 = torch.nn.Linear(
            in_features=input_size + n_identity_features, out_features=hidden_size
        )
        # self.transform2 = torch.nn.Linear(
        #     in_features=hidden_size, out_features=hidden_size
        # )
        self.transform3 = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size
        )

    def forward(self, identity_features, context):

        # x = torch.cat([identity_features, context], dim=-1)
        x = F.relu(self.transform1(context))
        x = F.tanh(self.transform3(x))
        return x


class AnchorMaxPoolEmbeddingNet(torch.nn.Module):

    def __init__(
        self,
        feature_dim,
        channels_l1,
        channels_l2,
        h_maxpool_l1,
        w_maxpool_l2,
        embedding_size,
        out_features,
        **kwargs,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_features = out_features

        self.conv1_l1 = Conv2d(
            in_channels=2,
            out_channels=channels_l1,
            kernel_size=(1, feature_dim),
            stride=(1, feature_dim),
        )
        self.maxpool_l1 = AdaptiveMaxPool2d(output_size=(h_maxpool_l1, 10))

        self.conv1_l2 = Conv2d(
            in_channels=channels_l1,
            out_channels=channels_l2,
            kernel_size=(h_maxpool_l1, 2),
        )
        self.maxpool_l2 = AdaptiveMaxPool2d(output_size=(1, w_maxpool_l2))
        self.output_layer1 = torch.nn.Linear(
            in_features=channels_l2 * w_maxpool_l2,
            out_features=embedding_size,
        )
        self.output_layer2 = torch.nn.Linear(
            in_features=embedding_size,
            out_features=out_features,
        )

    def forward(self, context):

        batch_size = context.shape[0]

        # cov: (batch_size, d, n, f, f)
        # print(context.shape)
        x = self.conv1_l1(context)
        x = self.maxpool_l1(x)
        x = self.conv1_l2(x)
        x = self.maxpool_l2(x).reshape(batch_size, -1)
        x = F.relu(self.output_layer1(x))
        x = self.output_layer2(x)

        if self.out_features == 3:
            x = torch.concat((torch.tanh(x[:, :2]), torch.sigmoid(x[:, 2:])), dim=1)
        else:
            x = torch.tanh(x)

        return x


class AnchorKernelEmbeddingNet(torch.nn.Module):

    def __init__(
        self, d: int = 2, channels: int = 6, embedding_size: int = 10, **kwargs
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv2d(
            in_channels=d,
            out_channels=channels,
            kernel_size=(4, 1),
            stride=1,
            groups=2,
        )
        self.conv2 = Conv2d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=(3, 3),
            stride=1,
            groups=2,
        )

        self.conv3 = Conv2d(
            in_channels=channels * 2,
            out_channels=channels * 3,
            kernel_size=(2, 2),
            stride=1,
            groups=1,
        )

        self.maxpool = MaxPool2d(kernel_size=(2, 2), stride=1)

        self.output_layer1 = torch.nn.Linear(
            in_features=channels * 3,
            out_features=20,
        )

        self.output_layer2 = torch.nn.Linear(
            in_features=20, out_features=embedding_size
        )

    def forward(self, context):

        batch_size = context.shape[0]

        # cov: (batch_size, d, n, f, f)
        # print(context.shape)
        x = F.relu(self.conv1(context))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.maxpool(x))
        # print(x.shape)

        x = F.relu(self.output_layer1(x.squeeze()))
        embedding = F.relu(self.output_layer2(x))
        # print(embedding[:5])

        return embedding


class CovKernelEmbeddingNet(torch.nn.Module):

    def __init__(
        self,
        d: int = 2,
        channels: int = 10,
        output_kernel_size: int = 2,
        context_features: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.output_kernel_size = output_kernel_size
        self.conv1 = Conv3d(
            in_channels=d,
            out_channels=channels,
            kernel_size=(4, 3, 3),
            stride=1,
            groups=2,
        )
        self.conv2 = Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(4, 3, 3),
            stride=1,
            groups=2,
        )
        # self.conv3 = Conv3d(
        #     in_channels=channels,
        #     out_channels=channels,
        #     kernel_size=2,
        #     stride=1,
        #     groups=1,
        # )

        self.transconv1 = ConvTranspose3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 2, 2),
            stride=1,
            groups=1,
        )

        self.transconv2 = ConvTranspose3d(
            in_channels=channels,
            out_channels=2,
            kernel_size=(4, self.output_kernel_size, self.output_kernel_size),
            stride=1,
            groups=1,
        )

        self.output_layer1 = torch.nn.Linear(
            in_features=context_features * (self.output_kernel_size + 1) ** 2,
            out_features=20,
        )

        # self.output_layer2 = torch.nn.Linear(in_features=20, out_features=2)

    def forward(self, context):

        edge_features, cov_feature_matrix = context
        # edge_features: (batch_size, n, f, d)
        batch_size = edge_features.shape[0]

        # cov: (batch_size, d, n, f, f)
        # print(cov.shape)
        x = F.relu(self.conv1(cov_feature_matrix))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        # x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.transconv1(x))
        # print(x.shape)
        x = self.transconv2(x)

        x = x.reshape(*x.shape[:-2], -1)  # (batch_size, d, n, h)
        x = x.flatten(1, 2)  # (batch_size, d*n, h)

        edge_features = edge_features.permute(0, 2, 3, 1).flatten(
            2, 3
        )  # (batch_size, f, d*n)
        embedding = edge_features @ x  # (batch_size, f, output_kernel_size**2)
        embedding = embedding.reshape(batch_size, -1)

        return embedding
        # pred = self.output_layer2(F.sigmoid(self.output_layer1(embedding)))

        # pred = F.sigmoid(pred) * self.bound

        # return pred
