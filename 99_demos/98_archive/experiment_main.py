"""
GP-KAN実験用メインノートブック

このnotebookでは、モジュール化されたGP-KAN実装を使用して
ベンチマーク関数の学習と評価を行います。
"""

# %% [markdown]
# # GP-KAN with Modularized Implementation
# 
# このnotebookでは、モジュール化されたGP-KAN実装を使用してベンチマーク関数の近似実験を行います。
# 
# ## 目次
# 1. 環境設定とライブラリのインポート
# 2. 実験設定の読み込み
# 3. データセットの作成
# 4. モデルの構築
# 5. モデルの学習
# 6. 結果の評価と可視化

# %% [markdown]
# ## 1. 環境設定とライブラリのインポート

# %%
import sys
import warnings
from pathlib import Path

import torch
import gpytorch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# モジュールのインポート
from gpkan_modules import (
    GPKAN,
    set_seed,
    create_dataset,
    train_gpkan,
    evaluate_model,
    plot_learning_curves,
    plot_prediction_results_2d,
    visualize_gp_activation_functions,
)
from configs import ExperimentConfig, TrainingConfig, ModelConfig, DataConfig
from benchmarks import BBOBSphere

print("PyTorch version:", torch.__version__)
print("GPytorch version:", gpytorch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %% [markdown]
# ## 2. 実験設定の読み込み
# 
# ハイパーパラメータとベンチマーク関数の設定を読み込みます。

# %%
# 実験設定の作成
config = ExperimentConfig(
    training=TrainingConfig(
        learning_rate=0.1,
        batch_size=256,
        num_epochs=1000,
        scheduler_type="cosine",
        random_seed=42
    ),
    model=ModelConfig(
        architecture=[2, 1],
        num_inducing=16,
        inducing_range=(-1.5, 1.5)
    ),
    data=DataConfig(
        num_samples=1000,
        input_range=(-2.0, 2.0),
        benchmark_name="sphere",
        benchmark_dimension=2,
        benchmark_x_opt=[0.0, 0.0],
        benchmark_f_opt=10.0
    )
)

# 設定内容の表示
config.print_config()

# 再現性のための乱数シード設定
set_seed(config.training.random_seed)

# %% [markdown]
# ## 3. データセットの作成
# 
# ベンチマーク関数からデータセットを生成します。

# %%
# ベンチマーク関数の作成
benchmark_func = BBOBSphere(
    dimension=config.data.benchmark_dimension,
    x_opt=config.data.benchmark_x_opt,
    f_opt=config.data.benchmark_f_opt
)

print(benchmark_func.info())
print()

# データセットの作成
train_loader, val_loader, test_loader, X_test, y_test = create_dataset(
    target_function=benchmark_func,
    num_samples=config.data.num_samples,
    input_range=config.data.input_range,
    input_dim=config.data.benchmark_dimension,
    train_ratio=config.training.train_ratio,
    val_ratio=config.training.val_ratio,
    batch_size=config.training.batch_size,
    random_seed=config.training.random_seed
)

# %% [markdown]
# ## 4. モデルの構築
# 
# GP-KANモデルを構築します。

# %%
# モデルの作成
model = GPKAN(
    layer_sizes=config.model.architecture,
    num_inducing=config.model.num_inducing,
    inducing_range=config.model.inducing_range
)

model.print_model_info()

# パラメータ数の表示
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ## 5. モデルの学習
# 
# GP-KANモデルを学習します。

# %%
print("=== GP-KANモデルの学習を開始 ===\n")

train_losses, val_losses = train_gpkan(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=config.training.num_epochs,
    learning_rate=config.training.learning_rate,
    scheduler_type=config.training.scheduler_type,
    eta_min_ratio=config.training.eta_min_ratio
)

# %% [markdown]
# ### 5.1 学習曲線の可視化

# %%
plot_learning_curves(train_losses, val_losses)

# %% [markdown]
# ## 6. 結果の評価と可視化
# 
# テストデータでモデルを評価し、結果を可視化します。

# %% [markdown]
# ### 6.1 テストデータでの評価

# %%
results = evaluate_model(
    model=model,
    test_loader=test_loader,
    y_test=y_test
)

# %% [markdown]
# ### 6.2 予測結果の詳細可視化

# %%
plot_prediction_results_2d(
    model=model,
    target_function=benchmark_func,
    X_test=X_test,
    y_test=y_test,
    predictions=results['predictions'],
    variances=results['variances'],
    r2_score=results['r2'],
    rmse=results['rmse'],
    input_range=config.data.input_range,
    x_opt=config.data.benchmark_x_opt,
    f_opt=config.data.benchmark_f_opt
)

# %% [markdown]
# ### 6.3 GP活性化関数の可視化

# %%
visualize_gp_activation_functions(
    model=model,
    input_range=config.data.input_range,
    num_points=100
)

# %% [markdown]
# ## まとめ
# 
# この実験では、モジュール化されたGP-KAN実装を使用して以下を行いました：
# 
# 1. ベンチマーク関数（BBOB Sphere）のデータセット作成
# 2. GP-KANモデルの構築と学習
# 3. テストデータでの性能評価
# 4. 予測結果とGP活性化関数の可視化
# 
# モジュール化により、異なるベンチマーク関数や設定での実験が容易になりました。

# %%
