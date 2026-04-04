"""
GP-KAN Modules Package

GP-KAN (Gaussian Process Kolmogorov-Arnold Network) の実装モジュール群
"""

from .base_model import BaseGPModel
from .deep_gp_model import DeepGP
from .gpkan_layer import GPKANLayer
from .gpkan_model import GPKAN
from .gpkan_node import GPKANNode
from .sparse_gp_model import SparseGP
from .training_utils import create_dataset, evaluate_model, set_seed, train_gpkan
from .visualization_utils import (
    plot_learning_curves,
    plot_prediction_results_2d,
    visualize_gp_activation_functions,
)

__all__ = [
    "BaseGPModel",
    "GPKANNode",
    "GPKANLayer",
    "GPKAN",
    "SparseGP",
    "DeepGP",
    "set_seed",
    "create_dataset",
    "train_gpkan",
    "evaluate_model",
    "plot_learning_curves",
    "plot_prediction_results_2d",
    "visualize_gp_activation_functions",
]
