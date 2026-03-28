"""
GPKANLayer: エッジベースのレイヤー

GP-KANレイヤーの実装。各入力次元から各出力次元への
GPエッジを持ち、エッジベースの処理を行います。

KAN論文（Liu et al., 2024）に基づく設計:
- 各エッジ: φ(x) = w_residual * x + GP(x)  (base function + spline の対応)
- forward: GP後験平均を使った決定論的伝播（rsampleを使わない）
- 損失: 全エッジのKL項を含む正しいSVGP-ELBO
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from .gpkan_node import GPKANNode


class GPKANLayer(nn.Module):
    """
    GP-KANレイヤー: エッジベースの処理を実装

    各入力次元 i と各出力次元 j に対してGPエッジ (i→j) を作成し、
    各入力次元を独立に1次元GPで変換します。
    出力ノードで全エッジからの寄与を総和します（KAN制約）。

    KAN論文との対応:
      φ_{l,j,i}(x) = residual_weights[i,j] * x + GP_{i→j}(x)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
                      base function (線形残差)       spline の代替
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_inducing: int = 10,
        inducing_range: tuple = (-2.0, 2.0),
        jitter: float = 1e-4
    ):
        super(GPKANLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_inducing = num_inducing

        # GPエッジとlikelihoodの作成
        self.gp_edges = nn.ModuleList()
        self.likelihoods = nn.ModuleList()

        for i in range(input_size):
            row_edges = nn.ModuleList()
            row_likelihoods = nn.ModuleList()

            for j in range(output_size):
                inducing_points = torch.linspace(
                    inducing_range[0], inducing_range[1], num_inducing
                ).reshape(-1, 1)

                gp_edge = GPKANNode(inducing_points, jitter=jitter)
                likelihood = GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-2)
                )

                row_edges.append(gp_edge)
                row_likelihoods.append(likelihood)

            self.gp_edges.append(row_edges)
            self.likelihoods.append(row_likelihoods)

        # KAN論文の base function に対応する学習可能な線形残差重み
        # 初期値ゼロ: 学習初期はGPのみで予測し、必要に応じて線形成分を学習
        self.residual_weights = nn.Parameter(
            torch.zeros(input_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播: 決定論的なエッジベースの処理

        KAN論文に従い、各エッジの出力を GP後験平均 + 線形残差 で計算します。
        rsample() による確率的サンプリングは行わず、多層スタック時の
        分散累積と勾配不安定化を防ぎます。

        Args:
            x: 入力テンソル (shape: [batch_size, input_size])
        Returns:
            出力テンソル (shape: [batch_size, output_size])
        """
        outputs = []

        for j in range(self.output_size):
            node_output = 0.0

            for i in range(self.input_size):
                gp_edge = self.gp_edges[i][j]
                input_i = x[:, i:i+1]

                # GP後験平均（決定論的）+ 線形残差
                gp_mean = gp_edge(input_i).mean
                residual = self.residual_weights[i, j] * input_i.squeeze(-1)
                node_output = node_output + gp_mean + residual

            outputs.append(node_output.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)

    def get_kl_loss(self, batch_size: int) -> torch.Tensor:
        """
        全GPエッジのKL損失を計算（中間層用）

        KL[q(u) || p(u)] は入力xに依存しない変分分布パラメータのみの関数。
        中間層はNLLを直接計算できないため、このKL項だけを上位モデルに返す。

        Args:
            batch_size: バッチサイズ（正規化用）
        Returns:
            KL損失の合計
        """
        kl_loss = 0.0
        num_edges = self.input_size * self.output_size

        for i in range(self.input_size):
            for j in range(self.output_size):
                kl = (
                    self.gp_edges[i][j]
                    .variational_strategy.kl_divergence()
                    .sum()
                )
                kl_loss = kl_loss + kl / (batch_size * num_edges)

        return kl_loss

    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        最終層の損失を計算（NLL + KL）

        全エッジの出力分布を合成し、ターゲットとの NLL を計算します。
        合成後の分布:
          mean = Σ_i (GP_i.mean + residual_i * x_i)
          var  = Σ_i GP_i.variance  （残差は決定論的なのでゼロ分散）

        Args:
            x: 入力テンソル (shape: [batch_size, input_size])
            y: ターゲットテンソル (shape: [batch_size, 1] or [batch_size, output_size])
        Returns:
            損失値（スカラー）
        """
        batch_size = x.size(0)
        total_loss = 0.0

        for j in range(self.output_size):
            edge_means = []
            edge_vars = []

            for i in range(self.input_size):
                gp_edge = self.gp_edges[i][j]
                input_i = x[:, i:i+1]
                gp_output = gp_edge(input_i)

                # GP平均に線形残差を加算
                residual = self.residual_weights[i, j] * input_i.squeeze(-1)
                edge_means.append(gp_output.mean + residual)
                edge_vars.append(gp_output.variance)

            # 全エッジの合計分布（独立性の仮定）
            combined_mean = sum(edge_means)
            combined_var = torch.clamp(sum(edge_vars), min=1e-6)
            combined_dist = Normal(combined_mean, torch.sqrt(combined_var))

            target = y.squeeze(-1) if self.output_size == 1 else y[:, j]
            loss = -combined_dist.log_prob(target).mean()

            # このレイヤーの全エッジのKL損失
            for i in range(self.input_size):
                kl = (
                    self.gp_edges[i][j]
                    .variational_strategy.kl_divergence()
                    .sum()
                )
                loss = loss + kl / (batch_size * self.input_size)

            total_loss += loss

        return total_loss

    def train_mode(self):
        """全GPノードとlikelihoodを学習モードに"""
        self.train()
        for i in range(self.input_size):
            for j in range(self.output_size):
                self.gp_edges[i][j].train()
                self.likelihoods[i][j].train()

    def eval_mode(self):
        """全GPノードとlikelihoodを評価モードに"""
        self.eval()
        for i in range(self.input_size):
            for j in range(self.output_size):
                self.gp_edges[i][j].eval()
                self.likelihoods[i][j].eval()
