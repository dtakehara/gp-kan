"""
BaseGPModel: すべてのGPモデルの抽象基底クラス

共通のインターフェースと機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseGPModel(nn.Module, ABC):
    """
    GPモデルの抽象基底クラス

    すべてのGPモデル（GPKAN、SparseGP、DeepGP）が継承する共通インターフェース。
    """

    def __init__(self, layer_sizes: list, num_inducing: int):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_inducing = num_inducing
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.num_layers = len(layer_sizes) - 1

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播（各モデルで実装）"""
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """予測（平均と分散を返す）"""
        pass

    @abstractmethod
    def get_total_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """損失を計算"""
        pass

    def train_mode(self):
        """学習モードに設定"""
        self.train()
        if hasattr(self, "likelihood"):
            self.likelihood.train()

    def eval_mode(self):
        """評価モードに設定"""
        self.eval()
        if hasattr(self, "likelihood"):
            self.likelihood.eval()

    @abstractmethod
    def print_model_info(self):
        """モデル情報を表示"""
        pass

    def _print_common_info(self, model_name: str):
        """共通のモデル情報を表示"""
        print("=" * 60)
        print(f"{model_name} Model")
        print("=" * 60)
        print(f"Layer structure: {self.layer_sizes}")
        print(f"Input dimension: {self.input_dim}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Inducing points per layer: {self.num_inducing}")
        if hasattr(self, "likelihood"):
            print(f"Likelihood: {self.likelihood.__class__.__name__}")
        print("-" * 60)


def adjust_output_shape(tensor: torch.Tensor, target_dim: int = 1) -> torch.Tensor:
    """
    出力テンソルの形状を調整

    Args:
        tensor: 調整するテンソル
        target_dim: 目標の出力次元
    Returns:
        形状調整後のテンソル
    """
    if target_dim == 1 and tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    return tensor


def adjust_target_shape(y: torch.Tensor) -> torch.Tensor:
    """
    ターゲットテンソルの形状を調整

    Args:
        y: ターゲットテンソル
    Returns:
        形状調整後のテンソル
    """
    if y.dim() == 2 and y.size(1) == 1:
        return y.squeeze(-1)
    return y
