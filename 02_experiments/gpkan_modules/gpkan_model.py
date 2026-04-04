"""
GPKAN: GP Kolmogorov-Arnold Network

エッジベースのGPネットワーク実装。
"""

from typing import Tuple

import torch
from gpytorch.likelihoods import GaussianLikelihood

from .base_model import BaseGPModel
from .gpkan_layer import GPKANLayer


class GPKAN(BaseGPModel):
    """GP-KAN: エッジベースのGPネットワーク"""

    def __init__(
        self,
        layer_sizes: list,
        num_inducing: int = 10,
        inducing_range: Tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__(layer_sizes, num_inducing)

        self.layers = torch.nn.ModuleList(
            [
                GPKANLayer(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    num_inducing=num_inducing,
                    inducing_range=inducing_range,
                )
                for i in range(self.num_layers)
            ]
        )

        self.likelihood = GaussianLikelihood()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            # 最終層以外を順伝播
            for layer in self.layers[:-1]:
                x = layer(x)

            # 最終層: 全エッジの GP後験分布 + 線形残差 を集約
            final_layer = self.layers[-1]
            predictions, variances = [], []

            for j in range(final_layer.output_size):
                mean = torch.zeros(x.size(0))
                var = torch.zeros(x.size(0))

                for i in range(final_layer.input_size):
                    gp_edge = final_layer.gp_edges[i][j]
                    input_i = x[:, i:i+1]
                    gp_dist = gp_edge(input_i)
                    residual = (
                        final_layer.residual_weights[i, j]
                        * input_i.squeeze(-1)
                    )
                    mean += gp_dist.mean + residual
                    var += gp_dist.variance

                predictions.append(mean)
                variances.append(var)

            if final_layer.output_size == 1:
                return predictions[0], variances[0]
            return (
                torch.stack(predictions, dim=-1),
                torch.stack(variances, dim=-1),
            )

    def get_total_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 中間層: KL損失を収集しながら決定論的に順伝播
        batch_size = x.size(0)
        total_kl = 0.0

        for layer in self.layers[:-1]:
            total_kl = total_kl + layer.get_kl_loss(batch_size)
            x = layer(x)

        # 最終層: NLL + 最終層自身のKL
        return self.layers[-1].get_loss(x, y) + total_kl

    def train_mode(self):
        super().train_mode()
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self):
        super().eval_mode()
        for layer in self.layers:
            layer.eval_mode()

    def print_model_info(self):
        total_gps = sum(
            layer.input_size * layer.output_size for layer in self.layers
        )
        self._print_common_info("GP-KAN")
        print(f"Total GP edges: {total_gps}")
        print(f"Total inducing points: {total_gps * self.num_inducing}")
        for i, layer in enumerate(self.layers):
            num_edges = layer.input_size * layer.output_size
            print(
                f"Layer {i+1}: "
                f"{layer.input_size}→{layer.output_size} ({num_edges} edges)"
            )
        print("=" * 60)
