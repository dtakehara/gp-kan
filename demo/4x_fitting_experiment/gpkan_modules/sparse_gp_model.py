"""
SparseGP: Sparse Variational GPモデル

誘導点を用いた変分推論による効率的なGP実装。
"""

from typing import Tuple

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .base_model import BaseGPModel, adjust_target_shape


class SparseVariationalGP(ApproximateGP):
    """Sparse Variational GPの基本実装"""

    def __init__(self, inducing_points: torch.Tensor, input_dims: int):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dims))

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _initialize_inducing_points(
    input_dim: int, num_inducing: int, inducing_range: Tuple[float, float]
) -> torch.Tensor:
    """誘導点を初期化"""
    if input_dim == 1:
        return torch.linspace(
            inducing_range[0], inducing_range[1], num_inducing
        ).reshape(-1, 1)
    elif input_dim == 2:
        grid_size = int(torch.sqrt(torch.tensor(num_inducing)))
        grid_1d = torch.linspace(inducing_range[0], inducing_range[1], grid_size)
        x1, x2 = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        return torch.stack([x1.flatten(), x2.flatten()], dim=1)[:num_inducing]
    else:
        return (
            torch.randn(num_inducing, input_dim)
            * (inducing_range[1] - inducing_range[0])
            / 2
        )


class SparseGP(BaseGPModel):
    """Sparse GP: 効率的な変分GP実装"""

    def __init__(
        self,
        layer_sizes: list,
        num_inducing: int = 10,
        inducing_range: Tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__(layer_sizes, num_inducing)

        if len(layer_sizes) > 2:
            print(
                f"Warning: SparseGP uses only input/output dims: {self.input_dim}→{self.output_dim}"
            )

        inducing_points = _initialize_inducing_points(
            self.input_dim, num_inducing, inducing_range
        )
        self.model = SparseVariationalGP(inducing_points, self.input_dim)
        self.likelihood = GaussianLikelihood()
        self.mll = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_mean = self.likelihood(self.model(x)).mean
        return pred_mean.unsqueeze(-1) if self.output_dim == 1 else pred_mean

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            pred_dist = self.likelihood(self.model(x))
            return pred_dist.mean, pred_dist.variance

    def get_total_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mll is None:
            self.mll = VariationalELBO(self.likelihood, self.model, num_data=x.size(0))
        return -self.mll(self.model(x), adjust_target_shape(y))

    def print_model_info(self):
        self._print_common_info("Sparse Variational GP")
        print("Hyperparameters:")
        print(f"  Noise: {self.likelihood.noise.item():.6f}")
        print(f"  Output scale: {self.model.covar_module.outputscale.item():.6f}")
        print(
            f"  Length scales: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}"
        )
        print("=" * 60)
