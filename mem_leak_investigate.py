"""
This example was inspired by calling `qNEI` with a `SingleTaskGP` and
noticing surprisingly high memory usage. After that, I narrowed the
spike down to calling `model.posterior()` -- it didn't have to do with
`qNEI` at all -- and then realized it didn't involve BoTorch or a
model at all.

The code in `motivating_example` reproduces relevant parts of the
`posterior` method, recreating almost the same pattern of memory use
I initially saw with qNEI.
"""
import gc
from typing import Optional, Tuple

import torch
from botorch.models.utils import gpt_posterior_settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from memory_profiler import profile


torch.autograd.set_detect_anomaly(True)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_train_x_and_y(
    n: int, dim: int, batch_dim: Optional[int] = None, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    tkwargs = {"device": _get_device(), "dtype": torch.double}
    torch.manual_seed(seed)
    if batch_dim is None:
        x = torch.rand(n, dim, **tkwargs)
    else:
        x = torch.rand(batch_dim, n, dim, **tkwargs)
    y = torch.sin(x).sum(dim=-1, keepdim=True)
    return x, y


@profile
def motivating_example(dim: int, train_n: int, batch_dim: int, seed: int) -> None:
    train_X, train_Y = _get_train_x_and_y(train_n, dim, batch_dim, seed=seed)
    _aug_batch_shape = torch.Size([batch_dim])

    # Memory usage is much higher without gpt_posterior_settings
    with gpt_posterior_settings():
        mean_module = ConstantMean(batch_shape=_aug_batch_shape)
        # hmm mean_module converts to 32-bit
        mean_x = mean_module(train_X).to(train_X.dtype)

        covar_module = MaternKernel(
            ard_num_dims=train_X.shape[-1], batch_shape=_aug_batch_shape
        )

        # hmm this converts to 32-bit too
        covar_x = covar_module(train_X).to(train_X.dtype)

        # normally an attribute of a GPyTorch model
        prediction_strategy_ = prediction_strategy(
            train_inputs=[train_X],
            train_prior_dist=MultivariateNormal(mean_x, covar_x),
            train_labels=train_Y[:, :, 0],
            likelihood=GaussianLikelihood(batch_shape=_aug_batch_shape),
        )

        # Returns a (batch_dim, train_n) size tensor, which is relatively small
        mean = prediction_strategy_.mean_cache
        print(f"Tensor of size {mean.element_size() * mean.numel() / (1024 * 1024)} GB")
        gc.collect()


@profile
def run_several_times(dim: int, train_n: int, batch_dim: int) -> None:
    motivating_example(dim, train_n, batch_dim, seed=0)
    motivating_example(dim, train_n, batch_dim, seed=1)
    motivating_example(dim, train_n, batch_dim, seed=2)
    gc.collect()


if __name__ == "__main__":
    # NOT a memory leak; for sufficiently large tensors, stuff gets cleaned up.
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
    run_several_times(
        dim=30,
        train_n=200,
        batch_dim=400,
    )
