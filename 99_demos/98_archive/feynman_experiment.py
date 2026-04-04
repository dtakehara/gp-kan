# %% [markdown]
# # Feynman Dataset 精度検証ノートブック
# 
# このノートブックでは、Feynman Equationsを使用してGP-KANの精度を検証します。
# 各方程式のKAN shapeに基づいてモデルを構築し、学習・評価を行います。

# %% [markdown]
# ## 1. 環境設定とライブラリのインポート

# %%
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
import japanize_matplotlib

warnings.filterwarnings("ignore")

# モジュールのインポート
from gpkan_modules import (
    GPKAN,
    set_seed,
    create_dataset,
    train_gpkan,
    evaluate_model,
    plot_learning_curves,
    visualize_gp_activation_functions,
)
from configs import ExperimentConfig, TrainingConfig, ModelConfig, DataConfig
from benchmarks import get_feynman_equation, list_available_equations

# 日本語フォント設定
japanize_matplotlib.japanize()

print("=" * 80)
print("Feynman Dataset 精度検証ノートブック")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"GPytorch version: {gpytorch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 80)

# %% [markdown]
# ## 2. 利用可能なFeynman方程式の確認

# %%
available_equations = list_available_equations()
print(f"実装済みのFeynman方程式: {len(available_equations)}個\n")

# 各方程式の情報を表示
for eq_id in available_equations:
    eq = get_feynman_equation(eq_id)
    print(f"[{eq_id}]")
    print(f"  変数数: {eq.dimension}")
    print(f"  変数名: {', '.join(eq.variable_names)}")
    print(f"  KAN shape: {eq.kan_shape}")
    print()

# %% [markdown]
# ## 3. 実験設定の定義

# %%
# 共通の実験設定
base_config = ExperimentConfig(
    training=TrainingConfig(
        learning_rate=0.1,
        batch_size=256,
        num_epochs=500,
        scheduler_type="cosine",
        random_seed=42
    ),
    model=ModelConfig(
        architecture=[2, 1],  # 各方程式で上書きされる
        num_inducing=16,
        inducing_range=(-1.5, 1.5)
    ),
    data=DataConfig(
        num_samples=2000,
        input_range=(-2.0, 2.0)
    )
)

base_config.print_config()

# 乱数シード設定
set_seed(base_config.training.random_seed)

# %% [markdown]
# ## 4. 単一方程式での実験（デモ）
# 
# まず、1つの方程式（I.6.2）で詳細な実験を行います。

# %%
# デモ用の方程式ID
demo_equation_id = "I.6.2"

print(f"デモ実験: {demo_equation_id}")
print("=" * 80)

# Feynman方程式の取得
feynman_eq = get_feynman_equation(demo_equation_id)
print(f"\n{feynman_eq.info()}\n")

# 入力範囲の設定
demo_input_range = (-2.0, 2.0)

# モデル設定の更新
demo_config = ExperimentConfig(
    training=base_config.training,
    model=ModelConfig(
        architecture=feynman_eq.kan_shape,
        num_inducing=base_config.model.num_inducing,
        inducing_range=base_config.model.inducing_range
    ),
    data=DataConfig(
        num_samples=2000,
        input_range=demo_input_range
    )
)

print(f"入力次元: {feynman_eq.dimension}")
print(f"KAN shape: {feynman_eq.kan_shape}")
print(f"入力範囲: {demo_input_range}")
print(f"サンプル数: {demo_config.data.num_samples}")

# %% [markdown]
# ### 4.1 データセットの作成

# %%
demo_train_loader, demo_val_loader, demo_test_loader, demo_X_test, demo_y_test = create_dataset(
    target_function=feynman_eq,
    num_samples=demo_config.data.num_samples,
    input_range=demo_input_range,
    input_dim=feynman_eq.dimension,
    train_ratio=demo_config.training.train_ratio,
    val_ratio=demo_config.training.val_ratio,
    batch_size=demo_config.training.batch_size,
    random_seed=demo_config.training.random_seed
)

# データの可視化
if feynman_eq.dimension == 2:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(
        demo_X_test[:, 0].numpy(),
        demo_X_test[:, 1].numpy(),
        c=demo_y_test.numpy(),
        cmap='viridis',
        s=20,
        alpha=0.6
    )
    plt.colorbar(scatter, ax=ax, label='関数値')
    ax.set_xlabel(feynman_eq.variable_names[0])
    ax.set_ylabel(feynman_eq.variable_names[1])
    ax.set_title(f'{demo_equation_id}: テストデータの分布')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 4.2 モデルの作成

# %%
demo_model = GPKAN(
    layer_sizes=demo_config.model.architecture,
    num_inducing=demo_config.model.num_inducing,
    inducing_range=demo_config.model.inducing_range
)

demo_model.print_model_info()

total_params = sum(p.numel() for p in demo_model.parameters())
trainable_params = sum(p.numel() for p in demo_model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ### 4.3 モデルの学習

# %%
print("=" * 80)
print("学習開始")
print("=" * 80)

demo_train_losses, demo_val_losses = train_gpkan(
    model=demo_model,
    train_loader=demo_train_loader,
    val_loader=demo_val_loader,
    num_epochs=demo_config.training.num_epochs,
    learning_rate=demo_config.training.learning_rate,
    scheduler_type=demo_config.training.scheduler_type,
    eta_min_ratio=demo_config.training.eta_min_ratio
)

# %% [markdown]
# ### 4.4 学習曲線の可視化

# %%
plot_learning_curves(demo_train_losses, demo_val_losses)

# %% [markdown]
# ### 4.5 モデルの評価

# %%
print("=" * 80)
print("評価開始")
print("=" * 80)

demo_results = evaluate_model(
    model=demo_model,
    test_loader=demo_test_loader,
    y_test=demo_y_test
)

# 結果の整理
demo_result = {
    "equation_id": demo_equation_id,
    "dimension": feynman_eq.dimension,
    "kan_shape": feynman_eq.kan_shape,
    "metrics": {
        "rmse": demo_results['rmse'],
        "mae": demo_results['mae'],
        "r2": demo_results['r2'],
        "mean_variance": demo_results['mean_variance']
    },
    "training": {
        "final_train_loss": demo_train_losses[-1],
        "final_val_loss": demo_val_losses[-1],
        "min_val_loss": min(demo_val_losses)
    }
}

print("\n" + "=" * 80)
print(f"デモ実験完了: {demo_equation_id}")
print("=" * 80)
print(f"RMSE: {demo_results['rmse']:.6f}")
print(f"MAE: {demo_results['mae']:.6f}")
print(f"R²: {demo_results['r2']:.6f}")
print("=" * 80)

# %% [markdown]
# ### 4.6 予測結果の可視化

# %%
# 予測 vs 真値
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 散布図: 真の値 vs 予測値
axes[0].scatter(demo_y_test.numpy(), demo_results['predictions'].numpy(), alpha=0.6, s=20)
min_val = min(demo_y_test.min().item(), demo_results['predictions'].min().item())
max_val = max(demo_y_test.max().item(), demo_results['predictions'].max().item())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='完全予測', linewidth=2)
axes[0].set_xlabel('真の値')
axes[0].set_ylabel('予測値')
axes[0].set_title(f'予測精度 (R² = {demo_results["r2"]:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 残差プロット
residuals = (demo_results['predictions'] - demo_y_test.squeeze()).numpy()
axes[1].scatter(demo_results['predictions'].numpy(), residuals, alpha=0.6, s=20)
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
axes[1].set_xlabel('予測値')
axes[1].set_ylabel('残差')
axes[1].set_title(f'残差プロット (RMSE = {demo_results["rmse"]:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.7 GP活性化関数の可視化

# %%
visualize_gp_activation_functions(
    model=demo_model,
    input_range=demo_input_range,
    num_points=100
)

# %% [markdown]
# ## 5. 全方程式での一括実験
# 
# 全てのFeynman方程式に対して実験を実行します。

# %%
def generate_input_range(equation_id: str, dimension: int) -> tuple:
    """方程式ごとに適した入力範囲を生成"""
    # ガウス分布系: sigmaは正の値のみ
    if "I.6.2" in equation_id:
        return (-2.0, 2.0)  # theta用、sigmaは別途クリップ
    # arcsin系: [-1, 1]の範囲内に制約
    elif "I.26.2" in equation_id:
        return (-0.9, 0.9)
    # 除算系: 0を避ける
    elif "I.27.6" in equation_id:
        return (1.1, 3.0)
    # デフォルト
    else:
        return (-2.0, 2.0)


def compute_inducing_range(input_range: tuple, margin_ratio: float = 0.3) -> tuple:
    """入力範囲から誘導点の適切な範囲を計算"""
    range_width = input_range[1] - input_range[0]
    margin = range_width * margin_ratio
    return (input_range[0] - margin, input_range[1] + margin)


def run_single_experiment(equation_id: str, config: ExperimentConfig, verbose: bool = False):
    """単一の方程式に対して実験を実行（数値安定化版）"""
    try:
        # Feynman方程式の取得
        feynman_eq = get_feynman_equation(equation_id)
        
        # 入力範囲の設定
        input_range = generate_input_range(equation_id, feynman_eq.dimension)
        
        # 誘導点範囲を動的に設定（データ範囲より少し広く）
        inducing_range = compute_inducing_range(input_range, margin_ratio=0.3)
        
        # モデル設定の更新（KAN shapeを使用）
        config.model.architecture = feynman_eq.kan_shape
        config.data.num_samples = 2000
        
        if verbose:
            print(f"\n{feynman_eq.info()}")
            print(f"KAN shape: {feynman_eq.kan_shape}")
            print(f"Input range: {input_range}")
            print(f"Inducing range: {inducing_range}")
        
        # データセット作成
        train_loader, val_loader, test_loader, X_test, y_test = create_dataset(
            target_function=feynman_eq,
            num_samples=config.data.num_samples,
            input_range=input_range,
            input_dim=feynman_eq.dimension,
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            batch_size=config.training.batch_size,
            random_seed=config.training.random_seed
        )
        
        # モデル作成（誘導点範囲を動的に設定）
        model = GPKAN(
            layer_sizes=config.model.architecture,
            num_inducing=config.model.num_inducing,
            inducing_range=inducing_range  # 動的に計算した範囲を使用
        )
        
        # 学習
        train_losses, val_losses = train_gpkan(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            scheduler_type=config.training.scheduler_type,
            eta_min_ratio=config.training.eta_min_ratio
        )
        
        # 評価
        results = evaluate_model(model=model, test_loader=test_loader, y_test=y_test)
        
        return {
            "equation_id": equation_id,
            "status": "success",
            "dimension": feynman_eq.dimension,
            "kan_shape": feynman_eq.kan_shape,
            "num_samples": config.data.num_samples,
            "input_range": input_range,
            "inducing_range": inducing_range,
            "metrics": {
                "rmse": results['rmse'],
                "mae": results['mae'],
                "r2": results['r2'],
                "mean_variance": results['mean_variance']
            },
            "training": {
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "min_val_loss": min(val_losses)
            }
        }
        
    except RuntimeError as e:
        error_msg = str(e)
        if "cholesky" in error_msg.lower() or "nan" in error_msg.lower():
            print(f"数値エラー ({equation_id}): Cholesky分解失敗")
            return {
                "equation_id": equation_id,
                "status": "numerical_error",
                "error": "Cholesky decomposition failed (numerical instability)"
            }
        else:
            print(f"実行時エラー ({equation_id}): {error_msg}")
            return {
                "equation_id": equation_id,
                "status": "runtime_error",
                "error": error_msg
            }
    except Exception as e:
        print(f"予期しないエラー ({equation_id}): {e}")
        return {
            "equation_id": equation_id,
            "status": "error",
            "error": str(e)
        }

# %% [markdown]
# ### 5.1 実験の実行

# %%
# 実験する方程式のリスト
experiment_equations = list_available_equations()

print(f"実験対象: {len(experiment_equations)} 個の方程式")
print(f"方程式: {', '.join(experiment_equations)}\n")

# 全実験の結果を保存
all_results = []

# 各方程式に対して実験を実行
for i, eq_id in enumerate(experiment_equations, 1):
    print(f"\n{'#' * 80}")
    print(f"進捗: {i}/{len(experiment_equations)} - {eq_id}")
    print(f"{'#' * 80}")
    
    result = run_single_experiment(eq_id, base_config, verbose=False)
    all_results.append(result)
    
    if result['status'] == 'success':
        print(f"✓ {eq_id}: RMSE={result['metrics']['rmse']:.6f}, R²={result['metrics']['r2']:.6f}")
    else:
        print(f"✗ {eq_id}: {result['status']}")

# %% [markdown]
# ### 5.2 結果のサマリー

# %%
print("\n" + "=" * 80)
print("全実験完了")
print("=" * 80)

successful = [r for r in all_results if r['status'] == 'success']
failed = [r for r in all_results if r['status'] != 'success']

print(f"成功: {len(successful)}/{len(all_results)}")
print(f"失敗: {len(failed)}/{len(all_results)}")

if successful:
    print("\n" + "-" * 80)
    print("成功した実験の結果:")
    print("-" * 80)
    print(f"{'方程式ID':<15} {'次元':<6} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
    print("-" * 80)
    for r in successful:
        print(f"{r['equation_id']:<15} "
              f"{r['dimension']:<6} "
              f"{r['metrics']['rmse']:<12.6f} "
              f"{r['metrics']['mae']:<12.6f} "
              f"{r['metrics']['r2']:<10.6f}")
    print("-" * 80)
    
    # 統計
    rmse_values = [r['metrics']['rmse'] for r in successful]
    mae_values = [r['metrics']['mae'] for r in successful]
    r2_values = [r['metrics']['r2'] for r in successful]
    
    print(f"\n統計:")
    print(f"  RMSE - 平均: {np.mean(rmse_values):.6f}, 中央値: {np.median(rmse_values):.6f}, 標準偏差: {np.std(rmse_values):.6f}")
    print(f"  MAE  - 平均: {np.mean(mae_values):.6f}, 中央値: {np.median(mae_values):.6f}, 標準偏差: {np.std(mae_values):.6f}")
    print(f"  R²   - 平均: {np.mean(r2_values):.6f}, 中央値: {np.median(r2_values):.6f}, 標準偏差: {np.std(r2_values):.6f}")

if failed:
    print("\n" + "-" * 80)
    print("失敗した実験:")
    print("-" * 80)
    for r in failed:
        print(f"{r['equation_id']}: {r['status']}")
        if 'error' in r:
            print(f"  エラー: {r['error']}")

print("=" * 80)

# %% [markdown]
# ### 5.3 結果の可視化

# %%
if successful:
    # DataFrameに変換
    df_results = pd.DataFrame([
        {
            'equation': r['equation_id'],
            'dimension': r['dimension'],
            'rmse': r['metrics']['rmse'],
            'mae': r['metrics']['mae'],
            'r2': r['metrics']['r2']
        }
        for r in successful
    ])
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RMSE
    axes[0, 0].bar(df_results['equation'], df_results['rmse'])
    axes[0, 0].set_xlabel('方程式ID')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('方程式ごとのRMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].bar(df_results['equation'], df_results['mae'])
    axes[0, 1].set_xlabel('方程式ID')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('方程式ごとのMAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # R²
    axes[1, 0].bar(df_results['equation'], df_results['r2'])
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='R²=0.95')
    axes[1, 0].set_xlabel('方程式ID')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('方程式ごとのR²スコア')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 次元数 vs RMSE
    axes[1, 1].scatter(df_results['dimension'], df_results['rmse'], s=100, alpha=0.6)
    for idx, row in df_results.iterrows():
        axes[1, 1].annotate(row['equation'], (row['dimension'], row['rmse']), 
                           fontsize=8, alpha=0.7)
    axes[1, 1].set_xlabel('入力次元数')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('入力次元数とRMSEの関係')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 相関行列
    print("\n相関行列:")
    print(df_results[['dimension', 'rmse', 'mae', 'r2']].corr())

# %% [markdown]
# ### 5.4 結果の保存

# %%
# JSONファイルに保存
output_file = "feynman_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n結果を {output_file} に保存しました。")

# CSVファイルにも保存（成功した実験のみ）
if successful:
    csv_file = "feynman_results.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"成功した実験の結果を {csv_file} に保存しました。")

# %% [markdown]
# ## 6. まとめ
# 
# このノートブックでは、以下を実施しました：
# 
# 1. **単一方程式でのデモ実験**: I.6.2を使用した詳細な実験
# 2. **全方程式での一括実験**: 実装済みの全Feynman方程式での精度検証
# 3. **結果の可視化と分析**: RMSE、MAE、R²スコアの比較
# 4. **結果の保存**: JSON形式とCSV形式での保存
# 
# ### 主要な知見
# 
# - GP-KANは各方程式のKAN shapeに応じた柔軟なモデル構築が可能
# - 統一されたハイパーパラメータでも良好な性能を達成
# - 入力次元数と精度の関係を確認
# 
# ### 今後の改善点
# 
# - 方程式ごとのハイパーパラメータ最適化
# - より多くのFeynman方程式の実装
# - 深層モデル（より多層のKAN shape）での実験

# %%
print("\n" + "=" * 80)
print("全ての実験が完了しました！")
print("=" * 80)
