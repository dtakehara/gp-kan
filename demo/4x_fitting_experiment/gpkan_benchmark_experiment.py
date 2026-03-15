# %%
"""
GPKAN Fitting Experiment: f1-f5 Benchmark Functions

f1-f5のベンチマーク関数を使用してGPKANのfitting実験を行い、
各nodeのGPの活性化関数を可視化します。
"""

# %% [markdown]
# # GPKAN Fitting Experiment: f1-f5 Benchmark Functions
#
# このnotebookでは、f1-f5のベンチマーク関数を使用してGPKANのfitting実験を行い、
# 各nodeのGPの活性化関数を可視化します。
#
# ## 目次
# 1. 環境設定とライブラリのインポート
# 2. ベンチマーク関数の定義
# 3. データ生成とDataLoader作成
# 4. 訓練関数
# 5. 可視化関数
# 6. 実験クラス
# 7. 実験の実行

# %% [markdown]
# ## 1. 環境設定とライブラリのインポート

# %%
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# モジュールのインポート
sys.path.append("gpkan_modules")
from gpkan_modules.base_model import adjust_target_shape
from gpkan_modules.gpkan_model import GPKAN

print("PyTorch version:", torch.__version__)
print("Setup complete!")

# %% [markdown]
# ## 2. ベンチマーク関数の定義
#
# f1-f5の5つのベンチマーク関数を定義します。

# %%


# %%
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


print("Benchmark functions defined:")
print("  f1: Multiplication (x × y)")
print("  f2: Division (x / y)")
print("  f3: Composite (exp(sin(πx) + y²))")
print("  f4: High-dimensional (4D input)")
print("  f5: Random function")

# %% [markdown]
# ## 3. データ生成とDataLoader作成

# %%


# %%
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
        torch.manual_seed(seed)

    X = np.random.uniform(x_range[0], x_range[1], (n_samples, input_dim))
    y = func(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    ).reshape(-1, 1)


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=128):
    """DataLoaderの作成"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


print("Data generation functions ready!")

# %% [markdown]
# ## 4. 訓練関数

# %%


# %%
def train_gpkan(
    model: GPKAN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
    lr: float = 0.01,
):
    """GPKANの訓練"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    epoch_times = []

    model.train_mode()
    total_start = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        # Training
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = model.get_total_loss(X_batch, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation
        model.eval_mode()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                loss = model.get_total_loss(X_batch, y_batch)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))
        model.train_mode()

        epoch_times.append(time.perf_counter() - epoch_start)

        if (epoch + 1) % 50 == 0:
            avg_epoch_time = sum(epoch_times[-50:]) / len(epoch_times[-50:])
            print(
                f"  Epoch {epoch+1}/{num_epochs}"
                f" - Train Loss: {train_losses[-1]:.4f}"
                f", Val Loss: {val_losses[-1]:.4f}"
                f", Avg Epoch Time: {avg_epoch_time:.2f}s"
            )

    total_time = time.perf_counter() - total_start
    print(f"  Training complete: {total_time:.1f}s total"
          f" ({total_time / num_epochs:.2f}s/epoch avg)")

    return train_losses, val_losses, total_time


print("Training function ready!")

# %% [markdown]
# ## 5. 可視化関数

# %%


# %%
def visualize_gp_nodes(
    model: GPKAN, input_range: Tuple[float, float], func_name: str, timestamp: str
):
    """各GPノードの活性化関数を可視化"""
    model.eval_mode()

    # 入力点の生成
    x_viz = torch.linspace(input_range[0], input_range[1], 200).unsqueeze(-1)

    for layer_idx, layer in enumerate(model.layers):
        num_inputs = layer.input_size
        num_outputs = layer.output_size

        fig, axes = plt.subplots(
            num_outputs, num_inputs, figsize=(4 * num_inputs, 3 * num_outputs)
        )
        if num_outputs == 1 and num_inputs == 1:
            axes = np.array([[axes]])
        elif num_outputs == 1:
            axes = axes.reshape(1, -1)
        elif num_inputs == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            f"Layer {layer_idx}: GP Activation Functions\n{func_name}",
            fontsize=14,
            fontweight="bold",
        )

        with torch.no_grad():
            for out_idx in range(num_outputs):
                for in_idx in range(num_inputs):
                    ax = axes[out_idx, in_idx]

                    # GP edgeとlikelihoodを取得
                    gp_edge = layer.gp_edges[in_idx][out_idx]
                    likelihood = layer.likelihoods[in_idx][out_idx]

                    # 予測（純粋なGP後験分布: likelihoodノイズを含まない）
                    gp_dist = gp_edge(x_viz)
                    mean = gp_dist.mean.numpy()
                    std = gp_dist.stddev.numpy()

                    # プロット
                    x_np = x_viz.numpy().flatten()
                    ax.plot(x_np, mean, "b-", linewidth=2, label="Mean")
                    ax.fill_between(
                        x_np,
                        mean - 2 * std,
                        mean + 2 * std,
                        alpha=0.3,
                        label="±2σ (GP posterior)",
                    )

                    # 誘導点をプロット
                    inducing_points = (
                        gp_edge.variational_strategy.inducing_points.detach()
                        .numpy()
                        .flatten()
                    )
                    inducing_values = gp_edge(
                        torch.tensor(inducing_points).unsqueeze(-1)
                    )
                    inducing_means = inducing_values.mean.detach().numpy()
                    ax.scatter(
                        inducing_points,
                        inducing_means,
                        c="red",
                        s=50,
                        zorder=5,
                        label="Inducing Points",
                    )

                    ax.set_title(f"Input {in_idx} → Output {out_idx}")
                    ax.set_xlabel("Input")
                    ax.set_ylabel("Output")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

        plt.tight_layout()
        filename = f"gpkan_nodes_{func_name}_layer{layer_idx}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {filename}")


# %%# %%


def plot_learning_curves(train_losses, val_losses, func_name: str, timestamp: str):
    """学習曲線のプロット"""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Training Loss", linewidth=2)
    ax.plot(val_losses, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning Curves - {func_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filename = f"gpkan_learning_curve_{func_name}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filename}")


# %%# %%
def plot_predictions_2d(
    model: GPKAN, func: Callable, func_name: str, timestamp: str, input_range=(-1, 1)
):
    """2次元関数の予測結果を可視化"""
    model.eval_mode()

    # グリッド生成
    x_grid = torch.linspace(input_range[0], input_range[1], 50)
    y_grid = torch.linspace(input_range[0], input_range[1], 50)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing="ij")
    grid_points = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)

    # 真の値
    z_true = func(grid_points.numpy()).reshape(50, 50)

    # 予測
    with torch.no_grad():
        mean, var = model.predict(grid_points)
        z_pred = mean.numpy().reshape(50, 50)

    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True function
    im0 = axes[0].contourf(
        X_grid.numpy(), Y_grid.numpy(), z_true, levels=20, cmap="viridis"
    )
    axes[0].set_title("True Function")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    # Prediction
    im1 = axes[1].contourf(
        X_grid.numpy(), Y_grid.numpy(), z_pred, levels=20, cmap="viridis"
    )
    axes[1].set_title("GPKAN Prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    # Error
    error = np.abs(z_true - z_pred)
    im2 = axes[2].contourf(
        X_grid.numpy(), Y_grid.numpy(), error, levels=20, cmap="Reds"
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"{func_name}: {func.__doc__}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    filename = f"gpkan_predictions_{func_name}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filename}")


print("Visualization functions ready!")

# %% [markdown]
# ## 6. 実験クラス

# %%


# %%
class GPKANExperiment:
    """GPKAN実験クラス"""

    def __init__(
        self,
        func_name: str,
        func: Callable,
        input_dim: int,
        hidden_size: int = 10,
        n_train: int = 800,
        n_val: int = 200,
        n_test: int = 200,
        epochs: int = 200,
        lr: float = 0.01,
    ):
        self.func_name = func_name
        self.func = func
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.epochs = epochs
        self.lr = lr

        # データ生成
        print(f"\n  Generating data...")
        self.X_train, self.y_train = generate_data(func, n_train, input_dim, seed=42)
        self.X_val, self.y_val = generate_data(func, n_val, input_dim, seed=43)
        self.X_test, self.y_test = generate_data(func, n_test, input_dim, seed=44)

        # DataLoader作成
        self.train_loader, self.val_loader = create_dataloaders(
            self.X_train, self.y_train, self.X_val, self.y_val, batch_size=128
        )

        # モデル作成
        layer_sizes = [input_dim, hidden_size, 1]
        self.model = GPKAN(layer_sizes, num_inducing=15)

        self.results = {}

    def run(self, timestamp: str):
        """実験実行"""
        print(f"\n{'='*80}")
        print(f"Experiment: {self.func_name}")
        print(f"Function: {self.func.__doc__}")
        print(f"Architecture: {self.model.layer_sizes}")
        print(f"{'='*80}")

        # 訓練
        print(f"\n  Training...")
        run_start = time.perf_counter()
        train_losses, val_losses, training_time = train_gpkan(
            self.model,
            self.train_loader,
            self.val_loader,
            num_epochs=self.epochs,
            lr=self.lr,
        )

        # テスト評価
        print(f"\n  Evaluating on test set...")
        self.model.eval_mode()
        with torch.no_grad():
            mean, var = self.model.predict(self.X_test)
            if mean.dim() == 1:
                mean = mean.unsqueeze(-1)
            test_loss = torch.nn.functional.mse_loss(mean, self.y_test).item()
            test_rmse = np.sqrt(test_loss)

        total_run_time = time.perf_counter() - run_start
        print(f"Test MSE:  {test_loss:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Training time: {training_time:.1f}s"
              f" | Total run time: {total_run_time:.1f}s")

        # 可視化
        print(f"\n  Generating visualizations...")
        plot_learning_curves(train_losses, val_losses, self.func_name, timestamp)

        if self.input_dim == 2:
            plot_predictions_2d(self.model, self.func, self.func_name, timestamp)

        visualize_gp_nodes(self.model, (-1.5, 1.5), self.func_name, timestamp)

        self.results = {
            "test_loss": test_loss,
            "test_rmse": test_rmse,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "training_time": training_time,
            "total_run_time": total_run_time,
        }

        return self.results


print("Experiment class ready!")

# %% [markdown]
# ## 7. 実験の実行
#
# f1-f5の全ての関数について実験を実行します。

# %%


# %%
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("GPKAN Fitting Experiment: f1-f5 Benchmark Functions")
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
        exp = GPKANExperiment(
            func_name,
            func,
            input_dim,
            hidden_size=4,
            n_train=800,
            n_val=200,
            n_test=200,
            epochs=100,
            lr=0.01,
        )
        results = exp.run(timestamp)

        all_results.append(
            {
                "function": func_name,
                "test_loss": results["test_loss"],
                "test_rmse": results["test_rmse"],
                "train_loss": results["final_train_loss"],
                "val_loss": results["final_val_loss"],
                "training_time": results["training_time"],
                "total_run_time": results["total_run_time"],
            }
        )

    total_experiment_time = sum(r["total_run_time"] for r in all_results)

    # サマリー出力
    print("\n" + "=" * 90)
    print("実験結果サマリー")
    print("=" * 90)
    print(
        f"{'Function':<20} {'Test RMSE':<12} {'Test MSE':<12}"
        f" {'Val Loss':<12} {'Train Time':>12} {'Total Time':>12}"
    )
    print("-" * 90)

    for result in all_results:
        print(
            f"{result['function']:<20} {result['test_rmse']:<12.6f}"
            f" {result['test_loss']:<12.6f} {result['val_loss']:<12.6f}"
            f" {result['training_time']:>10.1f}s {result['total_run_time']:>10.1f}s"
        )

    print("-" * 90)
    print(f"{'Total':<20} {'':<12} {'':<12} {'':<12} {total_experiment_time:>10.1f}s")

    # テキストファイルに結果を保存
    result_filename = f"gpkan_experiment_results_{timestamp}.txt"
    with open(result_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GPKAN Fitting Experiment Results\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            f.write(f"Function: {result['function']}\n")
            f.write(f"  Test RMSE:        {result['test_rmse']:.6f}\n")
            f.write(f"  Test MSE:         {result['test_loss']:.6f}\n")
            f.write(f"  Final Train Loss: {result['train_loss']:.6f}\n")
            f.write(f"  Final Val Loss:   {result['val_loss']:.6f}\n")
            f.write(f"  Training Time:    {result['training_time']:.1f}s\n")
            f.write(f"  Total Run Time:   {result['total_run_time']:.1f}s\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"Total Experiment Time: {total_experiment_time:.1f}s\n")

    print("\n" + "=" * 80)
    print(f"結果を保存しました: {result_filename}")
    print("=" * 80)


# %% [markdown]
# ## スクリプトとして実行する場合

# %%
if __name__ == "__main__":
    main()

# %%
