from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
from nflows.flows.base import Flow
from nflows.transforms import (
    BatchNorm,
    CompositeTransform,
    MaskedPiecewiseQuadraticAutoregressiveTransform,
    RandomPermutation,
    ReversePermutation,
)
from src.anomaly_detectors.models.distributions import Uniform
from torch.nn import functional as F

if TYPE_CHECKING:
    from src.deployment_area.polygon import PolygonWrapper


class ConditionalMaskedAutoregressiveFlow(Flow):
    """An autoregressive flow that uses affine transforms with masking.
    Reference:
    > G. Papamakarios et al., Masked Autoregressive Flow for Density Estimation,
    > Advances in Neural Information Processing Systems, 2017.
    """

    def __init__(
        self,
        features,
        context_features,
        hidden_features,
        num_layers,
        num_bins=10,
        num_blocks_per_layer=2,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        sigma=1,
        max_gradient=None,
        tails="linear",
        bound=np.sqrt(np.square(0.4) * 2) * 5,
        device="cuda",
        **kwargs,
    ):

        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedPiecewiseQuadraticAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                    tails=tails,
                    tail_bound=bound,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            # distribution=Normal([features], sigma, bound),
            distribution=Uniform(
                low=torch.tensor([-bound - 1e-3, -bound - 1e-3]),
                high=torch.tensor([bound + 1e-3, bound + 1e-3]),
            ),  # .to(device),
        )

        # # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        if max_gradient:
            for p in self.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -max_gradient, max_gradient)
                )
