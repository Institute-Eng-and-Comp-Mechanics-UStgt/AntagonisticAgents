from typing import Tuple, Union

import numpy as np
import torch
from nflows.distributions.base import Distribution
from nflows.distributions.uniform import BoxUniform
from nflows.utils import torchutils

# use uniform distribution for spline functions, normal distribution for unconstrained transforms

class Uniform(Distribution):
    def __init__(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        reinterpreted_batch_ndims: int = 1,
    ):
        """Multidimensionqal uniform distribution defined on a box.

        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """
        super().__init__()
        self._shape = torch.Size(low.shape)
        self.low = low
        self.high = high
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.dist = BoxUniform(low, high, reinterpreted_batch_ndims)
        # due to the spoofing robot, the input can not be validated -> returns -inf instead of throwing an error
        self.dist.base_dist._validate_args = False

    def _sample(self, num_samples: int, context: torch.Tensor):

        if context is None:
            samples = self.dist.sample(torch.Size((num_samples,)))

        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = self.dist.sample(torch.Size((context_size * num_samples,)))
            samples = torchutils.split_leading_dim(samples, [context_size, num_samples])

        return samples.to(torch.float32)

    def _log_prob(self, inputs, context):

        return self.dist.log_prob(inputs)

    def to(self, device=None, *args, **kwargs):
        self.dist.base_dist.low = self.dist.base_dist.low.to(device)
        self.dist.base_dist.high = self.dist.base_dist.high.to(device)
        return self


class Normal(Distribution):
    """A multivariate Normal with zero mean and sigma covariance."""

    def __init__(self, shape, sigma, bound):
        super().__init__()
        self._shape = torch.Size(shape)
        self.sigma = sigma
        self.bound = bound

        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5
                * (
                    np.prod(shape) * np.log(2 * np.pi)
                    + np.log(self.sigma ** np.prod(shape))  # np.prod(shape): -k
                ),  # log of determinant
                dtype=torch.float64,
            ),
            persistent=False,
        )

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        neg_energy = -0.5 * torchutils.sum_except_batch(
            (1 / self.sigma) * inputs**2,  # 1 / self.sigma : multiply diagonal matrix
            num_batch_dims=1,
        )

        log_prob = neg_energy - self._log_z
        mask = self.compute_l2(inputs, scale_in_place=False)
        log_prob[mask.squeeze()] = torch.log(torch.tensor(1e-15))

        return log_prob

    def _sample(self, num_samples, context):
        if context is None:
            samples = torch.randn(
                num_samples, *self._shape, device=self._log_z.device
            ) * np.sqrt(self.sigma)

        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(
                context_size * num_samples, *self._shape, device=context.device
            ) * np.sqrt(self.sigma)
            samples = torchutils.split_leading_dim(samples, [context_size, num_samples])

        # scale samples by their l2 norm
        _, scaled_samples = self.compute_l2(samples, scale_in_place=True)
        return scaled_samples

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)

    def compute_l2(
        self, input: torch.Tensor, scale_in_place: bool = False
    ) -> torch.Tensor:
        # scaling the samples when using a neural spline flow
        # prevents the flow from using the identity function and normal log prob

        l2_val = torch.sqrt(torch.sum(torch.square(input), dim=-1))
        mask = l2_val > self.bound

        if scale_in_place:
            input[mask] = torch.div(input[mask], l2_val[mask].unsqueeze(-1))
            return input

        return mask
