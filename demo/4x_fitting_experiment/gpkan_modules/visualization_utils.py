"""
Visualization Utilities: 可視化関連のユーティリティ関数

学習曲線、予測結果、GP活性化関数などの可視化機能を提供します。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import japanize_matplotlib

from gpkan_modules import GPKAN

# 日本語フォント設定
japanize_matplotlib.japanize()


def plot_learning_curves(
    train_losses: list,
    val_losses: list,
    figsize: tuple = (10, 6)
):
    """
    学習曲線を可視化
    
    Args:
        train_losses: 学習損失のリスト
        val_losses: 検証損失のリスト
        figsize: 図のサイズ
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Train Loss', alpha=0.7, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GP-KAN 学習曲線', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"最終 Train Loss: {train_losses[-1]:.4f}")
    print(f"最終 Validation Loss: {val_losses[-1]:.4f}")


def plot_prediction_results_2d(
    model: GPKAN,
    target_function,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    predictions: torch.Tensor,
    variances: torch.Tensor,
    r2_score: float,
    rmse: float,
    input_range: tuple = (-2.0, 2.0),
    x_opt: np.ndarray = None,
    f_opt: float = None,
    figsize: tuple = (18, 12)
):
    """
    2次元入力関数の予測結果を詳細に可視化
    
    Args:
        model: 学習済みGP-KANモデル
        target_function: 真の目標関数
        X_test: テスト入力
        y_test: テスト出力
        predictions: 予測値
        variances: 予測分散
        r2_score: R²スコア
        rmse: RMSE
        input_range: 入力範囲
        x_opt: 最適解位置
        f_opt: 最適値
        figsize: 図のサイズ
    """
    # 密なグリッドでの予測
    grid_density = 50
    x1_dense = np.linspace(input_range[0], input_range[1], grid_density)
    x2_dense = np.linspace(input_range[0], input_range[1], grid_density)
    X1_dense, X2_dense = np.meshgrid(x1_dense, x2_dense)
    grid_dense = torch.FloatTensor(
        np.column_stack([X1_dense.ravel(), X2_dense.ravel()])
    )
    
    # バッチ処理での予測
    with torch.no_grad():
        dense_means = []
        dense_vars = []
        
        dense_batch_size = 100
        for i in range(0, len(grid_dense), dense_batch_size):
            batch_grid = grid_dense[i:i + dense_batch_size]
            pred_mean, pred_var = model.predict(batch_grid)
            dense_means.append(pred_mean)
            dense_vars.append(pred_var)
        
        dense_mean = torch.cat(dense_means, dim=0).reshape(grid_density, grid_density)
        dense_var = torch.cat(dense_vars, dim=0).reshape(grid_density, grid_density)
    
    # 真の関数値
    true_dense = target_function(grid_dense.numpy()).reshape(grid_density, grid_density)
    
    # 6つのサブプロットで可視化
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. 真の関数（等高線）
    contour1 = axes[0, 0].contour(
        X1_dense, X2_dense, true_dense, levels=15, colors='blue', alpha=0.8
    )
    axes[0, 0].clabel(contour1, inline=True, fontsize=8)
    if x_opt is not None:
        axes[0, 0].scatter(
            x_opt[0], x_opt[1], c='red', s=100, marker='*',
            label='Optimum', zorder=5
        )
    axes[0, 0].set_title('真の関数', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x₁')
    axes[0, 0].set_ylabel('x₂')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. GP-KAN予測（等高線）
    contour2 = axes[0, 1].contour(
        X1_dense, X2_dense, dense_mean.numpy(), levels=15, colors='green', alpha=0.8
    )
    axes[0, 1].clabel(contour2, inline=True, fontsize=8)
    if x_opt is not None:
        axes[0, 1].scatter(
            x_opt[0], x_opt[1], c='red', s=100, marker='*',
            label='Optimum', zorder=5
        )
    axes[0, 1].set_title('GP-KAN予測', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x₁')
    axes[0, 1].set_ylabel('x₂')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 予測誤差
    error_dense = np.abs(dense_mean.numpy() - true_dense)
    im1 = axes[0, 2].imshow(
        error_dense,
        extent=[input_range[0], input_range[1], input_range[0], input_range[1]],
        origin='lower', cmap='RdBu_r', alpha=0.8
    )
    if x_opt is not None:
        axes[0, 2].scatter(
            x_opt[0], x_opt[1], c='red', s=100, marker='*',
            label='Optimum', zorder=5
        )
    plt.colorbar(im1, ax=axes[0, 2], label='絶対誤差')
    axes[0, 2].set_title('予測誤差', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x₁')
    axes[0, 2].set_ylabel('x₂')
    axes[0, 2].legend()
    
    # 4. 予測不確実性（分散）
    im2 = axes[1, 0].imshow(
        dense_var.numpy(),
        extent=[input_range[0], input_range[1], input_range[0], input_range[1]],
        origin='lower', cmap='plasma', alpha=0.8
    )
    plt.colorbar(im2, ax=axes[1, 0], label='予測分散')
    axes[1, 0].set_title('予測不確実性', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x₁')
    axes[1, 0].set_ylabel('x₂')
    
    # 5. 散布図: 真の値 vs 予測値
    axes[1, 1].scatter(y_test.numpy(), predictions.numpy(), alpha=0.6, s=20)
    min_val = min(y_test.min().item(), predictions.min().item())
    max_val = max(y_test.max().item(), predictions.max().item())
    axes[1, 1].plot(
        [min_val, max_val], [min_val, max_val], 'r--',
        alpha=0.8, label='完全予測', linewidth=2
    )
    axes[1, 1].set_xlabel('真の値')
    axes[1, 1].set_ylabel('予測値')
    axes[1, 1].set_title(f'予測精度 (R² = {r2_score:.4f})',
                          fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 残差プロット
    residuals = (predictions - y_test.squeeze()).numpy()
    axes[1, 2].scatter(predictions.numpy(), residuals, alpha=0.6, s=20)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
    axes[1, 2].set_xlabel('予測値')
    axes[1, 2].set_ylabel('残差')
    axes[1, 2].set_title(f'残差プロット (RMSE = {rmse:.4f})',
                          fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_gp_activation_functions(
    model: GPKAN,
    input_range: tuple = (-2.0, 2.0),
    num_points: int = 100
):
    """
    各レイヤーのGPエッジの活性化関数を可視化
    
    Args:
        model: 学習済みGP-KANモデル
        input_range: 可視化する入力範囲
        num_points: プロット点数
    """
    model.eval_mode()
    
    # 各レイヤーを可視化
    for layer_idx, layer in enumerate(model.layers):
        input_size = layer.input_size
        output_size = layer.output_size
        
        # サブプロットグリッドの作成
        fig, axes = plt.subplots(
            input_size, output_size,
            figsize=(4 * output_size, 3 * input_size),
            squeeze=False
        )
        
        fig.suptitle(
            f'Layer {layer_idx + 1}: GP活性化関数 ' +
            f'({input_size}D入力 → {output_size}D出力)',
            fontsize=14, fontweight='bold'
        )
        
        # 各GPエッジを可視化
        for i in range(input_size):
            for j in range(output_size):
                ax = axes[i, j]
                gp_edge = layer.gp_edges[i][j]
                likelihood = layer.likelihoods[i][j]
                
                with torch.no_grad():
                    # 誘導点の範囲を取得
                    inducing_points = gp_edge.get_inducing_points()
                    inducing_min = inducing_points.min().item()
                    inducing_max = inducing_points.max().item()
                    
                    # 可視化範囲を誘導点の範囲を含むように拡張
                    plot_min = min(input_range[0], inducing_min - 0.5)
                    plot_max = max(input_range[1], inducing_max + 0.5)
                    
                    # 拡張された範囲でテスト入力を作成
                    x_test = torch.linspace(plot_min, plot_max, num_points).reshape(-1, 1)
                    
                    # GP予測
                    gp_output = gp_edge(x_test)
                    pred_dist = likelihood(gp_output)
                    
                    mean = pred_dist.mean
                    variance = pred_dist.variance
                    std = torch.sqrt(variance)
                    
                    # 誘導点での予測
                    inducing_output = gp_edge(inducing_points)
                    inducing_pred = likelihood(inducing_output)
                    inducing_mean = inducing_pred.mean
                    inducing_std = torch.sqrt(inducing_pred.variance)
                    
                    # サンプル関数（3つ）
                    sample_functions = [gp_output.sample() for _ in range(3)]
                
                # プロット
                x_np = x_test.squeeze().detach().numpy()
                mean_np = mean.detach().numpy()
                std_np = std.detach().numpy()
                
                # GP平均と信頼区間
                ax.plot(x_np, mean_np, 'b-', linewidth=2, label='GP平均')
                ax.fill_between(
                    x_np, mean_np - 2*std_np, mean_np + 2*std_np,
                    alpha=0.3, color='blue', label='95%信頼区間'
                )
                
                # 誘導点
                inducing_x = inducing_points.squeeze().detach().numpy()
                inducing_y = inducing_mean.detach().numpy()
                
                # 誘導点のプロット
                ax.scatter(
                    inducing_x, inducing_y,
                    c='red', s=80, marker='o',
                    alpha=0.7, zorder=5, edgecolors='darkred',
                    label=f'誘導点 ({len(inducing_x)}個)'
                )
                
                # サンプル関数
                colors = ['green', 'orange', 'purple']
                for idx, sample in enumerate(sample_functions):
                    sample_np = sample.detach().numpy()
                    ax.plot(
                        x_np, sample_np, '--', alpha=0.5,
                        linewidth=1, color=colors[idx],
                        label='サンプル' if idx == 0 else None
                    )
                
                # 装飾
                ax.set_title(
                    f'エッジ [{i}→{j}]',
                    fontsize=10, fontweight='bold'
                )
                ax.set_xlabel('入力値')
                ax.set_ylabel('出力値')
                ax.set_xlim(plot_min, plot_max)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')
                
                # データ範囲の可視化
                ax.axvspan(input_range[0], input_range[1], alpha=0.1,
                          color='green', label='データ範囲')
                
                # 統計情報
                mean_range = f"{mean_np.min():.3f} to {mean_np.max():.3f}"
                var_mean = f"{variance.mean().item():.4f}"
                inducing_range_str = f"[{inducing_min:.3f}, {inducing_max:.3f}]"
                ax.text(
                    0.02, 0.98,
                    f'範囲: {mean_range}\n平均分散: {var_mean}\n誘導点範囲: {inducing_range_str}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )
        
        plt.tight_layout()
        plt.show()
        
        # レイヤー統計
        total_edges = input_size * output_size
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx + 1} 統計")
        print(f"{'='*60}")
        print(f"入力次元: {input_size}")
        print(f"出力次元: {output_size}")
        print(f"総GPエッジ数: {total_edges}")
        print(f"エッジあたり誘導点数: {len(gp_edge.get_inducing_points())}")
        print(f"レイヤー総誘導点数: {total_edges * len(gp_edge.get_inducing_points())}")
        print(f"{'='*60}\n")
