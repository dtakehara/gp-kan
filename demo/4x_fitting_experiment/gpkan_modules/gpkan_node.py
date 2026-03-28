"""
GPKANNode: 1次元ガウス過程ノード

GP-KANの基本構成要素である1次元ガウス過程ノードの実装。
変分推論を使用して学習可能なGPを提供します。
"""

import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch import constraints


class GPKANNode(gpytorch.models.ApproximateGP):
    """
    GP-KANの基本構成要素：1次元ガウス過程ノード（数値安定化版）
    
    変分推論を使用して学習可能な1次元ガウス過程を実装。
    各ノードは以下を持つ：
    - 誘導点 (inducing points): スパース近似のための代表点
    - RBFカーネル: 入力間の類似度を測る関数
    - 変分分布: 真の事後分布の近似
    """
    
    def __init__(self, inducing_points: torch.Tensor, jitter: float = 1e-4):
        """
        Args:
            inducing_points: 誘導点の初期位置 (shape: [num_inducing, 1])
            jitter: 数値安定性のための微小値（デフォルト: 1e-4）
        """
        # 変分分布の設定（Cholesky分解を使った多変量正規分布）
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        # 変分戦略の設定（誘導点の位置も学習可能）
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        
        super(GPKANNode, self).__init__(variational_strategy)
        
        # 平均関数: 定数（学習可能）
        self.mean_module = ConstantMean()
        
        # 共分散関数: スケール付きRBFカーネル（制約付き）
        self.covar_module = ScaleKernel(
            RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(0.1, 10.0)
            ),
            outputscale_constraint=gpytorch.constraints.Positive()
        )
        
        # Jitterの保存
        self.jitter = jitter
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        順伝播: 入力に対するGP分布を返す（数値安定化版）
        
        Args:
            x: 入力テンソル (shape: [batch_size, 1])
        Returns:
            多変量正規分布（平均と共分散を持つGP分布）
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Jitterを追加して数値安定性を向上
        if self.training:
            covar_x = covar_x.add_jitter(self.jitter)
        
        return MultivariateNormal(mean_x, covar_x)
    
    def get_inducing_points(self) -> torch.Tensor:
        """誘導点の現在位置を取得"""
        return self.variational_strategy.inducing_points
