"""
Training Utilities: 学習関連のユーティリティ関数

データセット作成、学習ループ、評価などの機能を提供します。
"""

import random
import time
from typing import Tuple, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gpkan_modules import GPKAN


def set_seed(seed: int = 42):
    """
    完全な再現性のための乱数シード設定
    
    Args:
        seed: 乱数シード
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_dataset(
    target_function: Callable,
    num_samples: int,
    input_range: Tuple[float, float],
    input_dim: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 128,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    ベンチマーク関数からデータセットを作成
    
    Args:
        target_function: 目標関数
        num_samples: サンプル数
        input_range: 入力範囲 (min, max)
        input_dim: 入力次元数
        train_ratio: 学習データの割合
        val_ratio: 検証データの割合
        batch_size: バッチサイズ
        random_seed: 乱数シード
    
    Returns:
        (train_loader, val_loader, test_loader, X_test, y_test)
    """
    # 入力点をランダムサンプリング
    X = np.random.uniform(input_range[0], input_range[1], (num_samples, input_dim))
    y = target_function(X)
    
    # PyTorchテンソルに変換
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    # データ分割
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    X_val = X_tensor[train_size:train_size + val_size]
    y_val = y_tensor[train_size:train_size + val_size]
    X_test = X_tensor[train_size + val_size:]
    y_test = y_tensor[train_size + val_size:]
    
    # データセットとDataLoader作成
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    def worker_init_fn(worker_id):
        """DataLoaderワーカープロセス用の乱数シード設定"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    generator = torch.Generator().manual_seed(random_seed)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, worker_init_fn=worker_init_fn, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, worker_init_fn=worker_init_fn, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, worker_init_fn=worker_init_fn, generator=generator
    )
    
    print(f"=== データセット作成完了 ===")
    print(f"学習データ: {len(train_dataset)} samples")
    print(f"検証データ: {len(val_dataset)} samples")
    print(f"テストデータ: {len(test_dataset)} samples")
    print(f"入力次元: {X_train.shape[1]}")
    print(f"関数値範囲: [{y.min():.3f}, {y.max():.3f}]")
    
    return train_loader, val_loader, test_loader, X_test, y_test


def train_gpkan(
    model: GPKAN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    scheduler_type: str = "cosine",
    eta_min_ratio: float = 0.01
) -> Tuple[list, list]:
    """
    GP-KANモデルの学習
    
    Args:
        model: GP-KANモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        num_epochs: エポック数
        learning_rate: 学習率
        scheduler_type: スケジューラタイプ ("cosine" or "plateau")
        eta_min_ratio: CosineAnnealing用の最小学習率比率
    
    Returns:
        (train_losses, val_losses) 学習・検証損失のリスト
    """
    model.train_mode()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # スケジューラの設定
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate * eta_min_ratio
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # === 訓練フェーズ ===
        model.train_mode()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = model.get_total_loss(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # === 検証フェーズ ===
        model.eval_mode()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                loss = model.get_total_loss(batch_x, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # 学習率調整
        if scheduler_type == "cosine":
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        
        # 進捗表示
        if (epoch + 1) % 25 == 0:
            if scheduler_type == "cosine":
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{num_epochs} | " +
                  f"Train Loss: {avg_train_loss:.4f} | " +
                  f"Val Loss: {avg_val_loss:.4f} | " +
                  f"LR: {current_lr:.6f}")
    
    training_time = time.time() - start_time
    print(f"\n学習完了! 所要時間: {training_time:.2f}秒")
    print(f"最終 Train Loss: {train_losses[-1]:.4f}")
    print(f"最終 Val Loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


def evaluate_model(
    model: GPKAN,
    test_loader: DataLoader,
    y_test: torch.Tensor
) -> dict:
    """
    テストデータでモデルを評価
    
    Args:
        model: 学習済みGP-KANモデル
        test_loader: テストデータローダー
        y_test: テストラベル
    
    Returns:
        評価指標を含む辞書
    """
    model.eval_mode()
    
    # テストデータでの予測（バッチ処理）
    test_means = []
    test_vars = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            test_mean, test_var = model.predict(batch_x)
            test_means.append(test_mean)
            test_vars.append(test_var)
    
    test_mean = torch.cat(test_means, dim=0)
    test_var = torch.cat(test_vars, dim=0)
    
    # 評価指標の計算
    test_rmse = torch.sqrt(torch.mean((test_mean - y_test.squeeze()) ** 2))
    test_mae = torch.mean(torch.abs(test_mean - y_test.squeeze()))
    test_r2 = 1 - torch.sum((y_test.squeeze() - test_mean) ** 2) / \
              torch.sum((y_test.squeeze() - y_test.mean()) ** 2)
    
    results = {
        'rmse': test_rmse.item(),
        'mae': test_mae.item(),
        'r2': test_r2.item(),
        'mean_variance': test_var.mean().item(),
        'predictions': test_mean,
        'variances': test_var
    }
    
    print("=" * 60)
    print("GP-KAN テスト性能評価")
    print("=" * 60)
    print(f"RMSE (Root Mean Squared Error): {results['rmse']:.6f}")
    print(f"MAE (Mean Absolute Error):      {results['mae']:.6f}")
    print(f"R² Score:                       {results['r2']:.6f}")
    print(f"平均予測分散:                    {results['mean_variance']:.6f}")
    print("=" * 60)
    
    return results
