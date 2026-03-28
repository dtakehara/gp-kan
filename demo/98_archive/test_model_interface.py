"""
モデル比較テスト: GPKAN, SparseGP, DeepGP

3つのモデルが同じインターフェースで動作することを確認するテストスクリプト
"""

import sys

sys.path.append(".")

import time

import numpy as np
import torch
from gpkan_modules import GPKAN, DeepGP, SparseGP, set_seed

# 乱数シード設定
set_seed(42)

print("=" * 80)
print("モデルインターフェーステスト: GPKAN, SparseGP, DeepGP")
print("=" * 80)

# テストデータの生成
print("\n1. テストデータの生成")
print("-" * 80)
num_samples = 100
X = torch.randn(num_samples, 2)  # 2次元入力
y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)  # 簡単な2次関数

print(f"入力データ形状: {X.shape}")
print(f"出力データ形状: {y.shape}")

# モデルの定義
print("\n2. モデルの作成")
print("-" * 80)

layer_sizes = [2, 1]  # 2入力 → 1出力
num_inducing = 16

models = {
    "GPKAN": GPKAN(layer_sizes=layer_sizes, num_inducing=num_inducing),
    "SparseGP": SparseGP(layer_sizes=layer_sizes, num_inducing=num_inducing),
    "DeepGP": DeepGP(layer_sizes=layer_sizes, num_inducing=num_inducing),
}

# 各モデルの情報を表示
for name, model in models.items():
    print(f"\n{name}:")
    model.print_model_info()

# インターフェーステスト
print("\n3. インターフェーステスト")
print("-" * 80)

for name, _ in models.items():
    print(f"\n{name}:")

    # 各テストで新しいモデルインスタンスを作成
    if name == "GPKAN":
        model = GPKAN(layer_sizes=layer_sizes, num_inducing=num_inducing)
    elif name == "SparseGP":
        model = SparseGP(layer_sizes=layer_sizes, num_inducing=num_inducing)
    else:  # DeepGP
        model = DeepGP(layer_sizes=layer_sizes, num_inducing=num_inducing)

    # train_modeのテスト
    try:
        model.train_mode()
        print("  ✓ train_mode() - OK")
    except Exception as e:
        print(f"  ✗ train_mode() - FAILED: {e}")

    # forwardのテスト
    try:
        output = model.forward(X[:10])
        print(f"  ✓ forward() - OK (output shape: {output.shape})")
    except Exception as e:
        print(f"  ✗ forward() - FAILED: {e}")

    # get_total_lossのテスト
    try:
        loss = model.get_total_loss(X[:10], y[:10])
        print(f"  ✓ get_total_loss() - OK (loss: {loss.item():.4f})")
    except Exception as e:
        print(f"  ✗ get_total_loss() - FAILED: {e}")

    # eval_modeのテスト
    try:
        model.eval_mode()
        print("  ✓ eval_mode() - OK")
    except Exception as e:
        print(f"  ✗ eval_mode() - FAILED: {e}")

    # predictのテスト
    try:
        pred_mean, pred_var = model.predict(X[:10])
        print(
            f"  ✓ predict() - OK (mean shape: {pred_mean.shape}, var shape: {pred_var.shape})"
        )
    except Exception as e:
        print(f"  ✗ predict() - FAILED: {e}")

# 簡易学習テスト
print("\n4. 簡易学習テスト（5エポック）")
print("-" * 80)

train_X = X[:80]
train_y = y[:80]
test_X = X[80:]
test_y = y[80:]

training_times = {}

for name, _ in models.items():
    print(f"\n{name}:")

    # 新しいモデルインスタンスを作成
    if name == "GPKAN":
        model = GPKAN(layer_sizes=layer_sizes, num_inducing=num_inducing)
    elif name == "SparseGP":
        model = SparseGP(layer_sizes=layer_sizes, num_inducing=num_inducing)
    else:  # DeepGP
        model = DeepGP(layer_sizes=layer_sizes, num_inducing=num_inducing)

    model.train_mode()

    # オプティマイザの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 学習ループ
    losses = []
    epoch_times = []
    train_start = time.perf_counter()

    for epoch in range(5):
        epoch_start = time.perf_counter()
        optimizer.zero_grad()
        loss = model.get_total_loss(train_X, train_y)
        loss.backward()
        optimizer.step()
        epoch_end = time.perf_counter()

        losses.append(loss.item())
        epoch_times.append(epoch_end - epoch_start)

    train_end = time.perf_counter()
    total_time = train_end - train_start
    training_times[name] = total_time

    print(f"  学習完了: 初期損失={losses[0]:.4f}, 最終損失={losses[-1]:.4f}")
    print(
        f"  学習時間: 合計={total_time:.4f}秒, 平均={np.mean(epoch_times):.4f}秒/エポック"
    )

    # テストデータでの予測
    model.eval_mode()
    with torch.no_grad():
        pred_mean, pred_var = model.predict(test_X)
        mse = torch.mean((pred_mean - test_y.squeeze()) ** 2).item()

    print(f"  テストMSE: {mse:.4f}")

# 学習時間サマリー
print("\n5. 学習時間サマリー")
print("-" * 80)
fastest = min(training_times, key=training_times.get)
for name, t in training_times.items():
    ratio = t / training_times[fastest]
    print(f"  {name}: {t:.4f}秒 (x{ratio:.2f})")

print("\n" + "=" * 80)
print("すべてのテスト完了！")
print("=" * 80)
