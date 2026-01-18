"""
Feynman Equations: 物理学の基礎方程式群

Feynman Lectures on Physicsに登場する方程式をベンチマーク関数として実装。
"""

import numpy as np
from typing import Union, List


class FeynmanEquation:
    """Feynman方程式の基底クラス"""
    
    def __init__(self, equation_id: str, variable_names: List[str], kan_shape: List[int]):
        self.equation_id = equation_id
        self.variable_names = variable_names
        self.kan_shape = kan_shape
        self.dimension = len(variable_names)
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        raise NotImplementedError("Subclass must implement __call__ method")
    
    def info(self) -> str:
        raise NotImplementedError("Subclass must implement info method")


class FeynmanI6_2(FeynmanEquation):
    """I.6.2: exp(-θ²/(2σ²)) / √(2πσ²)"""
    
    def __init__(self):
        super().__init__("I.6.2", ["θ", "σ"], [2, 2, 1, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        theta, sigma = x[:, 0], x[:, 1]
        # 数値安定性: sigmaが0に近い場合の保護
        sigma = np.clip(sigma, 0.01, None)
        result = np.exp(-theta**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: exp(-θ²/(2σ²)) / √(2πσ²)")


class FeynmanI6_2b(FeynmanEquation):
    """I.6.2b: exp(-(θ-θ₁)²/(2σ²)) / √(2πσ²)"""
    
    def __init__(self):
        super().__init__("I.6.2b", ["θ", "θ₁", "σ"], [3, 2, 2, 1, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        theta, theta1, sigma = x[:, 0], x[:, 1], x[:, 2]
        # 数値安定性: sigmaが0に近い場合の保護
        sigma = np.clip(sigma, 0.01, None)
        result = np.exp(-(theta - theta1)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: exp(-(θ-θ₁)²/(2σ²)) / √(2πσ²)")


class FeynmanI9_18(FeynmanEquation):
    """I.9.18: (G*m₁*m₂)/(r₁-r₂)²"""
    
    def __init__(self):
        super().__init__("I.9.18", ["G", "m₁", "m₂", "r₁", "r₂"], [5, 4, 2, 2, 1, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        G, m1, m2, r1, r2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        result = (G * m1 * m2) / ((r1 - r2)**2 + 1e-10)  # 数値安定性のため
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: (G*m₁*m₂)/(r₁-r₂)²")


class FeynmanI12_11(FeynmanEquation):
    """I.12.11: q(Eⱼ + Bvssinθ)"""
    
    def __init__(self):
        super().__init__("I.12.11", ["q", "Eⱼ", "B", "v", "s", "θ"], [6, 2, 2, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        q, Ej, B, v, s, theta = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        result = q * (Ej + B * v * s * np.sin(theta))
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: q(Eⱼ + Bvssinθ)")


class FeynmanI13_12(FeynmanEquation):
    """I.13.12: Gm₁m₂(1/r₂ - 1/r₁)"""
    
    def __init__(self):
        super().__init__("I.13.12", ["G", "m₁", "m₂", "r₁", "r₂"], [5, 2, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        G, m1, m2, r1, r2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        result = G * m1 * m2 * (1/(r2 + 1e-10) - 1/(r1 + 1e-10))
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: Gm₁m₂(1/r₂ - 1/r₁)")


class FeynmanI15_3x(FeynmanEquation):
    """I.15.3x: (x-u*t)/√(1-(u/c)²)"""
    
    def __init__(self):
        super().__init__("I.15.3x", ["x", "u", "t", "c"], [4, 2, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x_arr = np.array(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        
        x_val, u, t, c = x_arr[:, 0], x_arr[:, 1], x_arr[:, 2], x_arr[:, 3]
        result = (x_val - u * t) / np.sqrt(1 - (u/c)**2 + 1e-10)
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: (x-u*t)/√(1-(u/c)²)")


class FeynmanI16_6(FeynmanEquation):
    """I.16.6: (u+v)/(1+u*v/c²)"""
    
    def __init__(self):
        super().__init__("I.16.6", ["u", "v", "c"], [3, 2, 2, 2, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        u, v, c = x[:, 0], x[:, 1], x[:, 2]
        result = (u + v) / (1 + u * v / c**2 + 1e-10)
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: (u+v)/(1+u*v/c²)")


class FeynmanI18_4(FeynmanEquation):
    """I.18.4: (m₁*r₁ + m₂*r₂)/(m₁ + m₂)"""
    
    def __init__(self):
        super().__init__("I.18.4", ["m₁", "r₁", "m₂", "r₂"], [4, 2, 2, 2, 1, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        m1, r1, m2, r2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        result = (m1 * r1 + m2 * r2) / (m1 + m2 + 1e-10)
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: (m₁*r₁ + m₂*r₂)/(m₁ + m₂)")


class FeynmanI26_2(FeynmanEquation):
    """I.26.2: arcsin(n*sin(θ₂))"""
    
    def __init__(self):
        super().__init__("I.26.2", ["n", "θ₂"], [2, 2, 2, 2, 1, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n, theta2 = x[:, 0], x[:, 1]
        arg = n * np.sin(theta2)
        arg = np.clip(arg, -1, 1)  # arcsinの定義域に制限
        result = np.arcsin(arg)
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: arcsin(n*sin(θ₂))")


class FeynmanI27_6(FeynmanEquation):
    """I.27.6: 1/(γ-1) * p*V"""
    
    def __init__(self):
        super().__init__("I.27.6", ["γ", "p", "V"], [3, 2, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        gamma, p, V = x[:, 0], x[:, 1], x[:, 2]
        result = 1 / (gamma - 1 + 1e-10) * p * V
        return result if len(result) > 1 else result[0]
    
    def info(self) -> str:
        return (f"Feynman {self.equation_id}\n"
                f"  Variables: {', '.join(self.variable_names)}\n"
                f"  KAN shape: {self.kan_shape}\n"
                f"  Formula: 1/(γ-1) * p*V")


# Feynman方程式のレジストリ
FEYNMAN_EQUATIONS = {
    "I.6.2": FeynmanI6_2,
    "I.6.2b": FeynmanI6_2b,
    "I.9.18": FeynmanI9_18,
    "I.12.11": FeynmanI12_11,
    "I.13.12": FeynmanI13_12,
    "I.15.3x": FeynmanI15_3x,
    "I.16.6": FeynmanI16_6,
    "I.18.4": FeynmanI18_4,
    "I.26.2": FeynmanI26_2,
    "I.27.6": FeynmanI27_6,
}


def get_feynman_equation(equation_id: str) -> FeynmanEquation:
    """
    Feynman方程式を取得
    
    Args:
        equation_id: 方程式ID (例: "I.6.2")
    
    Returns:
        FeynmanEquationインスタンス
    """
    if equation_id not in FEYNMAN_EQUATIONS:
        available = ", ".join(FEYNMAN_EQUATIONS.keys())
        raise ValueError(
            f"Unknown equation ID: {equation_id}\n"
            f"Available equations: {available}"
        )
    
    return FEYNMAN_EQUATIONS[equation_id]()


def list_available_equations() -> List[str]:
    """利用可能なFeynman方程式のリストを取得"""
    return list(FEYNMAN_EQUATIONS.keys())
