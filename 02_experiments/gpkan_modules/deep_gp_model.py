"""
DeepGP: Deep Gaussian Processモデル

複数のGP層を積み重ねた深層モデル。
"""

from typing import Tuple

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP as GPyTorchDeepGP
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .base_model import BaseGPModel, adjust_target_shape


class DeepGPHiddenLayer(DeepGPLayer):
    """DeepGPの隠れ層"""

    def __init__(
        self,
        input_dims: int,
        output_dims: int = None,
        num_inducing: int = 20,
        mean_type: str = "constant",
    ):
        # 誘導点の初期化
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = (
            ConstantMean(batch_shape=batch_shape)
            if mean_type == "constant"
            else LinearMean(input_dims)
        )
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
            processed_inputs = [
                inp.unsqueeze(0).expand(
                    gpytorch.settings.num_likelihood_samples.value(), *inp.shape
                )
                for inp in other_inputs
            ]
            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGPModel(GPyTorchDeepGP):
    """Deep Gaussian Processの内部モデル"""

    def __init__(self, layer_sizes: list, num_inducing: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                DeepGPHiddenLayer(
                    input_dims=layer_sizes[i],
                    output_dims=(
                        layer_sizes[i + 1] if i < len(layer_sizes) - 2 else None
                    ),
                    num_inducing=num_inducing,
                    mean_type="linear" if i < len(layer_sizes) - 2 else "constant",
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.gp_likelihood = GaussianLikelihood()

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        for layer in self.layers:
            x = layer(x)
        return x


class DeepGP(BaseGPModel):
    """Deep GP: 多層GPモデル"""

    def __init__(
        self,
        layer_sizes: list,
        num_inducing: int = 10,
        inducing_range: Tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__(layer_sizes, num_inducing)
        self.model = DeepGPModel(layer_sizes, num_inducing)
        self.likelihood = self.model.gp_likelihood
        self.mll = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with gpytorch.settings.num_likelihood_samples(1):
            pred_mean = self.likelihood(self.model(x)).mean
            if pred_mean.dim() == 2 and pred_mean.size(0) == 1:
                pred_mean = pred_mean.squeeze(0)
            return pred_mean.unsqueeze(-1) if pred_mean.dim() == 1 else pred_mean

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
            pred_dist = self.likelihood(self.model(x))
            pred_mean, pred_var = pred_dist.mean, pred_dist.variance
            if pred_mean.dim() == 2 and pred_mean.size(0) == 1:
                pred_mean, pred_var = pred_mean.squeeze(0), pred_var.squeeze(0)
            return pred_mean, pred_var

    def get_total_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mll is None:
            self.mll = DeepApproximateMLL(
                VariationalELBO(self.likelihood, self.model, x.size(0))
            )
        with gpytorch.settings.num_likelihood_samples(1):
            return -self.mll(self.model(x), adjust_target_shape(y))

    def print_model_info(self):
        self._print_common_info("Deep Gaussian Process")
        for i, layer in enumerate(self.model.layers):
            layer_type = "Hidden" if i < self.num_layers - 1 else "Output"
            output_dim = self.layer_sizes[i + 1] if i < self.num_layers - 1 else 1
            print(f"Layer {i+1} ({layer_type}): {self.layer_sizes[i]}→{output_dim}")
        print("=" * 60)
