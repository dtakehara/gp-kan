"""
GPKANLayer: エッジベースのレイヤー

GP-KANレイヤーの実装。各入力次元から各出力次元への
GPエッジを持ち、エッジベースの処理を行います。
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from .gpkan_node import GPKANNode


class GPKANLayer(nn.Module):
    """
    GP-KANレイヤー: エッジベースの処理を実装（数値安定化版）
    
    各入力次元 i と各出力次元 j に対してGPエッジ (i→j) を作成し、
    各入力次元を独立に1次元GPで変換します。
    出力ノードで全エッジからの寄与を総和します（KAN制約2）。
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_inducing: int = 10,
        inducing_range: tuple = (-2.0, 2.0),
        jitter: float = 1e-4
    ):
        """
        Args:
            input_size: 入力次元数
            output_size: 出力次元数
            num_inducing: 誘導点の数
            inducing_range: 誘導点の初期配置範囲
        """
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
                
                # 数値安定化パラメータ付きGPノード
                gp_edge = GPKANNode(inducing_points, jitter=jitter)
                
                # ノイズ制約付きlikelihood
                likelihood = GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-2)
                )
                
                row_edges.append(gp_edge)
                row_likelihoods.append(likelihood)
            
            self.gp_edges.append(row_edges)
            self.likelihoods.append(row_likelihoods)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播: エッジベースの処理
        
        Args:
            x: 入力テンソル (shape: [batch_size, input_size])
        Returns:
            出力テンソル (shape: [batch_size, output_size])
        """
        batch_size = x.size(0)
        outputs = []
        
        for j in range(self.output_size):
            node_output = 0.0
            
            for i in range(self.input_size):
                gp_edge = self.gp_edges[i][j]
                input_i = x[:, i:i+1]
                gp_output = gp_edge(input_i)
                samples = gp_output.rsample()
                node_output = node_output + samples
            
            outputs.append(node_output.unsqueeze(-1))
        
        output = torch.cat(outputs, dim=-1)
        return output
    
    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        合計出力に対する損失を計算
        
        各エッジが独立にyを予測するのではなく、
        全エッジの合計出力がyに一致するように学習します。
        
        損失関数は以下の2項から構成されます：
        1. NLL (Negative Log-Likelihood): データ適合項
        2. KL Divergence: 変分推論の正則化項
        
        Args:
            x: 入力テンソル (shape: [batch_size, input_size])
            y: ターゲットテンソル (shape: [batch_size, output_size] or [batch_size, 1])
        Returns:
            損失値（スカラー）
        """
        batch_size = x.size(0)
        
        # 各出力次元について処理
        total_loss = 0.0
        
        for j in range(self.output_size):
            # このノードへの全エッジの寄与を収集
            edge_contributions = []
            
            for i in range(self.input_size):
                gp_edge = self.gp_edges[i][j]
                likelihood = self.likelihoods[i][j]
                
                input_i = x[:, i:i+1]
                gp_output = gp_edge(input_i)
                
                edge_contributions.append(gp_output)
            
            # 全エッジの合計分布を計算
            # 各エッジの平均と分散を合計（独立性の仮定）
            combined_mean = sum(gp.mean for gp in edge_contributions)
            combined_var = sum(gp.variance for gp in edge_contributions)
            
            # 数値安定性のためのクリッピング
            combined_var = torch.clamp(combined_var, min=1e-6)
            
            # 合計分布を作成
            combined_dist = Normal(combined_mean, torch.sqrt(combined_var))
            
            # ターゲット
            if self.output_size == 1:
                target = y.squeeze(-1)
            else:
                target = y[:, j]
            
            # 合計出力に対する負の対数尤度（NLL）
            loss = -combined_dist.log_prob(target).mean()
            
            # KL損失の追加（各エッジの変分推論項）
            for i in range(self.input_size):
                gp_edge = self.gp_edges[i][j]
                # 変分分布と事前分布のKLダイバージェンス
                kl_divergence = gp_edge.variational_strategy.kl_divergence().sum()
                loss = loss + kl_divergence / (batch_size * self.input_size)
            
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
