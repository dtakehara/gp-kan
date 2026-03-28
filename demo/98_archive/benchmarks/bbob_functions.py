"""
BBOB (Black-Box Optimization Benchmarking) 関数群

BBOB標準ベンチマーク関数の実装。
"""

import numpy as np
from typing import Union


class BBOBSphere:
    """
    BBOB Sphere 関数: f₁(x) = ||x - x_opt||² + f_opt
    
    最もシンプルな凸最適化問題。最適解は x_opt。
    """
    
    def __init__(
        self,
        dimension: int = 2,
        x_opt: Union[np.ndarray, None] = None,
        f_opt: float = 10.0
    ):
        """
        Args:
            dimension: 入力次元数
            x_opt: 最適解位置（Noneの場合は原点）
            f_opt: 最適値
        """
        self.dimension = dimension
        self.f_opt = f_opt
        
        if x_opt is None:
            self.x_opt = np.zeros(dimension)
        else:
            self.x_opt = np.array(x_opt)
            if len(self.x_opt) != dimension:
                raise ValueError(
                    f"x_opt dimension {len(self.x_opt)} does not match "
                    f"specified dimension {dimension}"
                )
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        """
        関数値を計算
        
        Args:
            x: 入力ベクトル (N, dimension) または (dimension,)
        Returns:
            関数値 (N,) または スカラー
        """
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # z = x - x_opt
        z = x - self.x_opt.reshape(1, -1)
        
        # ||z||²の計算
        z_squared_norm = np.sum(z**2, axis=1)
        
        # f₁(x) = ||z||² + f_opt
        result = z_squared_norm + self.f_opt
        
        return result if len(result) > 1 else result[0]
    
    def get_optimum(self) -> tuple:
        """
        最適解と最適値を取得
        
        Returns:
            (x_opt, f_opt) のタプル
        """
        return self.x_opt.copy(), self.f_opt
    
    def info(self) -> str:
        """関数の情報を文字列で返す"""
        return (
            f"BBOB Sphere Function (f₁)\n"
            f"  Dimension: {self.dimension}\n"
            f"  Optimal point: {self.x_opt}\n"
            f"  Optimal value: {self.f_opt}\n"
            f"  Formula: f(x) = ||x - x_opt||² + f_opt"
        )
