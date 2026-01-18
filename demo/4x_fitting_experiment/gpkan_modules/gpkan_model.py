"""
GPKAN: 完全なネットワークモデル

複数のGPKANLayerを積み重ねた深層ネットワークモデルの実装。
"""

from typing import Tuple

import torch
import torch.nn as nn
from gpytorch.likelihoods import GaussianLikelihood

from .gpkan_layer import GPKANLayer


class GPKAN(nn.Module):
    """
    GP-KAN (Gaussian Process Kolmogorov-Arnold Network)
    
    複数のGPKANLayerを積み重ねて深層ネットワークを構築します。
    """
    
    def __init__(
        self,
        layer_sizes: list,
        num_inducing: int = 10,
        inducing_range: Tuple[float, float] = (-2.0, 2.0)
    ):
        """
        Args:
            layer_sizes: 各層のサイズリスト（例: [2, 1] は 2入力→1出力）
            num_inducing: 各GPエッジの誘導点数
            inducing_range: 誘導点の初期配置範囲
        """
        super(GPKAN, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.num_inducing = num_inducing
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = GPKANLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                num_inducing=num_inducing,
                inducing_range=inducing_range
            )
            self.layers.append(layer)
        
        self.likelihood = GaussianLikelihood()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
        Returns:
            出力テンソル
        """
        current_input = x
        for layer in self.layers:
            current_input = layer(current_input)
        return current_input
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        予測を実行（全エッジの寄与を集約）
        
        Args:
            x: 入力テンソル
        Returns:
            (予測平均, 予測分散) のタプル
        """
        self.eval()
        with torch.no_grad():
            # 最終層まで順伝播
            current_input = x
            for layer in self.layers[:-1]:
                current_input = layer(current_input)
            
            # 最終層の処理（全エッジからの寄与を集約）
            final_layer = self.layers[-1]
            batch_size = current_input.size(0)
            
            # 各出力次元について予測
            predictions = []
            variances = []
            
            for j in range(final_layer.output_size):
                node_mean = torch.zeros(batch_size)
                node_var = torch.zeros(batch_size)
                
                # 全入力次元からの寄与を集約
                for i in range(final_layer.input_size):
                    gp_edge = final_layer.gp_edges[i][j]
                    likelihood = final_layer.likelihoods[i][j]
                    
                    input_i = current_input[:, i:i+1]
                    gp_dist = gp_edge(input_i)
                    pred_dist = likelihood(gp_dist)
                    
                    node_mean = node_mean + pred_dist.mean
                    node_var = node_var + pred_dist.variance
                
                predictions.append(node_mean)
                variances.append(node_var)
            
            if final_layer.output_size == 1:
                return predictions[0], variances[0]
            else:
                return torch.stack(predictions, dim=-1), torch.stack(variances, dim=-1)
    
    def get_total_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        全レイヤーの損失を計算
        
        Args:
            x: 入力テンソル
            y: ターゲットテンソル
        Returns:
            損失値（スカラー）
        """
        current_input = x
        total_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                # 最終層: 全エッジの損失を layer.get_loss() で計算
                loss = layer.get_loss(current_input, y)
                total_loss += loss
            else:
                # 中間層: 次の層への伝播
                current_input = layer(current_input)
        
        return total_loss
    
    def train_mode(self):
        """全レイヤーを学習モードに設定"""
        self.train()
        self.likelihood.train()
        for layer in self.layers:
            layer.train_mode()
    
    def eval_mode(self):
        """全レイヤーを評価モードに設定"""
        self.eval()
        self.likelihood.eval()
        for layer in self.layers:
            layer.eval_mode()
    
    def print_model_info(self):
        """モデル構造の詳細を表示"""
        total_gps = sum(
            layer.input_size * layer.output_size for layer in self.layers
        )
        total_inducing = total_gps * self.num_inducing
        
        print("=" * 60)
        print("GP-KAN Model Architecture")
        print("=" * 60)
        print(f"Layer structure: {self.layer_sizes}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Total GP edges: {total_gps}")
        print(f"Inducing points per GP: {self.num_inducing}")
        print(f"Total inducing points: {total_inducing}")
        print(f"Likelihood: {self.likelihood}")
        print("-" * 60)
        
        for i, layer in enumerate(self.layers):
            num_edges = layer.input_size * layer.output_size
            print(f"Layer {i+1}: {layer.input_size}→{layer.output_size} " +
                  f"({num_edges} GP edges)")
        print("=" * 60)
