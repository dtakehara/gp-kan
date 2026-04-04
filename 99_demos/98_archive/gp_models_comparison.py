"""
GP Models Comparison: GPKAN vs DeepGP vs SparseGP

f1-f5の関数について、3つのGPモデルのfitting性能を比較します。
"""

import csv
import sys
import time
from datetime import datetime
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

# モデルのインポート
sys.path.append("gpkan_modules")
from gpkan_modules.deep_gp_model import DeepGP
from gpkan_modules.gpkan_model import GPKAN
from gpkan_modules.sparse_gp_model import SparseGP

# ========================================
# 関数定義（KAN実験と同じ）
# ========================================


def f1(x: np.ndarray) -> np.ndarray:
    """f(x, y) = x × y"""
    return x[:, 0] * x[:, 1]


def f2(x: np.ndarray) -> np.ndarray:
    """f(x, y) = x / y"""
    return x[:, 0] / (x[:, 1] + 1e-8)


def f3(x: np.ndarray) -> np.ndarray:
    """f(x, y) = exp(sin(πx) + y²)"""
    return np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2)


def f4(x: np.ndarray) -> np.ndarray:
    """f(x, y, z, w) = exp(sin(πx) + y²) + sin(πz) + w²"""
    return (
        np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2)
        + np.sin(np.pi * x[:, 2])
        + x[:, 3] ** 2
    )


def f5_random(x: np.ndarray) -> np.ndarray:
    """f5: ランダム関数（構造なし）"""
    np.random.seed(123)
    return (
        np.sin(x[:, 0] * 3.7) * np.cos(x[:, 1] * 2.3)
        + np.tanh(x[:, 0] + x[:, 1] * 1.5) * 0.5
    )


def generate_data(
    func: Callable,
    n_samples: int,
    input_dim: int,
    x_range: Tuple[float, float] = (-1, 1),
    seed: int = None,
):
    """データ生成"""
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, input_dim))
    y = func(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    ).reshape(-1, 1)


# ========================================
# 実験クラス
# ========================================


class GPModelComparison:
    """GPモデル比較実験"""

    def __init__(
        self,
        func_name: str,
        func: Callable,
        input_dim: int,
        output_dim: int = 1,
        n_train: int = 500,
        n_test: int = 100,
        epochs: int = 100,
        lr: float = 0.01,
    ):
        self.func_name = func_name
        self.func = func
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_train = n_train
        self.n_test = n_test
        self.epochs = epochs
        self.lr = lr

        # データ生成
        self.X_train, self.y_train = generate_data(func, n_train, input_dim, seed=42)
        self.X_test, self.y_test = generate_data(func, n_test, input_dim, seed=123)

        self.results = {}

    def train_gpkan(self, hidden_size: int = 10, num_inducing: int = 10):
        """GPKANの訓練"""
        print(f"\n  Training GPKAN...")
        layer_sizes = [self.input_dim, hidden_size, self.output_dim]
        model = GPKAN(layer_sizes, num_inducing=num_inducing)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        start_time = time.time()
        model.train_mode()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = model.get_total_loss(self.X_train, self.y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        train_time = time.time() - start_time

        # テスト評価
        model.eval_mode()
        mean, var = model.predict(self.X_test)
        # 次元を揃える
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        test_loss = torch.nn.functional.mse_loss(mean, self.y_test).item()

        self.results["GPKAN"] = {
            "train_time": train_time,
            "test_loss": test_loss,
            "test_std": var.mean().sqrt().item(),
            "model": model,
        }

        print(f"    Train time: {train_time:.2f}s, Test loss: {test_loss:.6f}")

    def train_sparse_gp(self, num_inducing: int = 50):
        """SparseGPの訓練"""
        print(f"\n  Training SparseGP...")
        layer_sizes = [self.input_dim, self.output_dim]
        model = SparseGP(
            layer_sizes=layer_sizes,
            num_inducing=num_inducing,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        start_time = time.time()
        model.train_mode()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = model.get_total_loss(self.X_train, self.y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        train_time = time.time() - start_time

        # テスト評価
        model.eval_mode()
        mean, var = model.predict(self.X_test)
        # 次元を揃える
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        test_loss = torch.nn.functional.mse_loss(mean, self.y_test).item()

        self.results["SparseGP"] = {
            "train_time": train_time,
            "test_loss": test_loss,
            "test_std": var.mean().sqrt().item(),
            "model": model,
        }

        print(f"    Train time: {train_time:.2f}s, Test loss: {test_loss:.6f}")

    def train_deep_gp(self, hidden_dims: list = [10], num_inducing: int = 30):
        """DeepGPの訓練"""
        print(f"\n  Training DeepGP...")
        layer_sizes = [self.input_dim] + hidden_dims + [self.output_dim]
        model = DeepGP(
            layer_sizes=layer_sizes,
            num_inducing=num_inducing,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        start_time = time.time()
        model.train_mode()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = model.get_total_loss(self.X_train, self.y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        train_time = time.time() - start_time

        # テスト評価
        model.eval_mode()
        mean, var = model.predict(self.X_test)
        # 次元を揃える
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        test_loss = torch.nn.functional.mse_loss(mean, self.y_test).item()

        self.results["DeepGP"] = {
            "train_time": train_time,
            "test_loss": test_loss,
            "test_std": var.mean().sqrt().item(),
            "model": model,
        }

        print(f"    Train time: {train_time:.2f}s, Test loss: {test_loss:.6f}")

    def run_all(self):
        """全モデルを訓練"""
        print(f"\n{'='*80}")
        print(f"Function: {self.func_name} - {self.func.__doc__}")
        print(f"{'='*80}")

        self.train_gpkan()
        self.train_sparse_gp()
        self.train_deep_gp()

        # 結果サマリー
        print(f"\n  Results Summary:")
        print(f"  {'Model':<15} {'Test Loss':<15} {'Train Time (s)':<15}")
        print(f"  {'-'*45}")
        for model_name, result in self.results.items():
            print(
                f"  {model_name:<15} {result['test_loss']:<15.6f} {result['train_time']:<15.2f}"
            )

    def plot_predictions(self, timestamp: str):
        """予測結果の可視化（2次元入力の場合のみ）"""
        if self.input_dim != 2:
            print(f"  Skipping visualization for {self.input_dim}D input")
            return

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # True function
        x_grid = torch.linspace(-1, 1, 50)
        y_grid = torch.linspace(-1, 1, 50)
        X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing="ij")
        grid_points = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
        z_true = self.func(grid_points.numpy()).reshape(50, 50)

        axes[0].contourf(
            X_grid.numpy(), Y_grid.numpy(), z_true, levels=20, cmap="viridis"
        )
        axes[0].set_title(f"True Function\n{self.func.__doc__}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # 各モデルの予測
        model_names = ["GPKAN", "SparseGP", "DeepGP"]
        for idx, model_name in enumerate(model_names, 1):
            if model_name not in self.results:
                continue

            model = self.results[model_name]["model"]
            model.eval_mode()
            with torch.no_grad():
                mean, _ = model.predict(grid_points)
                z_pred = mean.numpy().reshape(50, 50)

            axes[idx].contourf(
                X_grid.numpy(), Y_grid.numpy(), z_pred, levels=20, cmap="viridis"
            )
            test_loss = self.results[model_name]["test_loss"]
            axes[idx].set_title(f"{model_name}\nTest Loss: {test_loss:.6f}")
            axes[idx].set_xlabel("x")
            axes[idx].set_ylabel("y")

        plt.tight_layout()
        filename = f"gp_comparison_{self.func_name}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved visualization: {filename}")


# ========================================
# メイン実験
# ========================================


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("GP Models Comparison: GPKAN vs DeepGP vs SparseGP")
    print("=" * 80)

    # 実験定義
    experiments = [
        ("f1_multiplication", f1, 2),
        ("f2_division", f2, 2),
        ("f3_composite", f3, 2),
        ("f4_high_dim", f4, 4),
        ("f5_random", f5_random, 2),
    ]

    all_results = []

    for func_name, func, input_dim in experiments:
        comparison = GPModelComparison(
            func_name,
            func,
            input_dim,
            n_train=500,
            n_test=100,
            epochs=100,
            lr=0.01,
        )
        comparison.run_all()
        comparison.plot_predictions(timestamp)

        # 結果を集約
        for model_name, result in comparison.results.items():
            all_results.append(
                {
                    "function": func_name,
                    "model": model_name,
                    "test_loss": result["test_loss"],
                    "train_time": result["train_time"],
                    "test_std": result["test_std"],
                }
            )

    # 結果をCSVに保存
    csv_filename = f"gp_comparison_results_{timestamp}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["function", "model", "test_loss", "train_time", "test_std"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    # テキスト形式でも保存
    txt_filename = f"gp_comparison_results_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GP Models Comparison Results\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for func_name, func, _ in experiments:
            f.write(f"\nFunction: {func_name}\n")
            f.write(f"  {func.__doc__}\n")
            f.write(f"  {'Model':<15} {'Test Loss':<15} {'Train Time':<15}\n")
            f.write(f"  {'-'*45}\n")

            func_results = [r for r in all_results if r["function"] == func_name]
            for result in func_results:
                f.write(
                    f"  {result['model']:<15} {result['test_loss']:<15.6f} "
                    f"{result['train_time']:<15.2f}s\n"
                )

    # 総合サマリー
    print("\n" + "=" * 80)
    print("総合結果サマリー")
    print("=" * 80)
    print(f"{'Function':<20} {'Model':<15} {'Test Loss':<15} {'Train Time':<15}")
    print("-" * 80)

    for result in all_results:
        print(
            f"{result['function']:<20} {result['model']:<15} "
            f"{result['test_loss']:<15.6f} {result['train_time']:<15.2f}s"
        )

    print("\n" + "=" * 80)
    print(f"結果を保存しました:")
    print(f"  - {txt_filename}")
    print(f"  - {csv_filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()
