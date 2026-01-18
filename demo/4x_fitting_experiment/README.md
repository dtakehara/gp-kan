# GP-KAN モジュール化実装

このディレクトリには、`temp_31_3_fitting_bbob_gpkan_v2.ipynb` をモジュール化した実装が含まれています。

## 📁 ディレクトリ構造

```
4x_fitting_experiment/
├── README.md                    # このファイル（プロジェクト説明）
├── experiment_main.py           # メイン実験スクリプト
├── gpkan_modules/              # GP-KANコアモジュール
│   ├── __init__.py
│   ├── gpkan_node.py           # GPKANNode: 1次元GPノード
│   ├── gpkan_layer.py          # GPKANLayer: エッジベースレイヤー
│   ├── gpkan_model.py          # GPKAN: 完全なネットワークモデル
│   ├── training_utils.py       # 学習・評価ユーティリティ
│   └── visualization_utils.py  # 可視化ユーティリティ
├── configs/                    # 実験設定
│   ├── __init__.py
│   └── experiment_config.py    # ハイパーパラメータ設定
└── benchmarks/                 # ベンチマーク関数
    ├── __init__.py
    └── bbob_functions.py       # BBOB関数群
```

## 🎯 モジュール化の方針

元のノートブックから以下の2つを切り離しました：

### 1. ハイパーパラメータ (`configs/`)
- **学習設定**: 学習率、バッチサイズ、エポック数など
- **モデル設定**: アーキテクチャ、誘導点数など
- **データ設定**: サンプル数、入力範囲など

### 2. ベンチマーク関数 (`benchmarks/`)
- **BBOB Sphere関数**: 凸最適化ベンチマーク
- 他のベンチマーク関数も追加可能

## 📦 モジュール詳細

### `gpkan_modules/` - コアモジュール

#### `gpkan_node.py` - GPKANNode
1次元ガウス過程ノードの実装。

**主要機能**:
- 変分推論を使った学習可能なGP
- RBFカーネルとスケーリング
- 誘導点の学習

**使用例**:
```python
from gpkan_modules import GPKANNode
import torch

inducing_points = torch.linspace(-2, 2, 10).reshape(-1, 1)
gp_node = GPKANNode(inducing_points)
```

#### `gpkan_layer.py` - GPKANLayer
エッジベースのレイヤー処理。

**主要機能**:
- 各入力次元から各出力次元へのGPエッジを作成
- 合計出力に対する損失計算（NLL + KL）
- 全エッジの寄与を総和

**使用例**:
```python
from gpkan_modules import GPKANLayer

layer = GPKANLayer(
    input_size=2,
    output_size=1,
    num_inducing=16,
    inducing_range=(-1.5, 1.5)
)
```

#### `gpkan_model.py` - GPKAN
完全なネットワークモデル。

**主要機能**:
- 複数レイヤーの積み重ね
- 予測（平均と分散）
- 全レイヤーの損失計算

**使用例**:
```python
from gpkan_modules import GPKAN

model = GPKAN(
    layer_sizes=[2, 1],
    num_inducing=16,
    inducing_range=(-1.5, 1.5)
)
model.print_model_info()
```

#### `training_utils.py` - 学習ユーティリティ
データセット作成、学習ループ、評価などの機能。

**主要関数**:
- `set_seed(seed)`: 乱数シード設定
- `create_dataset(...)`: データセット作成
- `train_gpkan(...)`: 学習ループ
- `evaluate_model(...)`: モデル評価

**使用例**:
```python
from gpkan_modules import create_dataset, train_gpkan, evaluate_model

# データセット作成
train_loader, val_loader, test_loader, X_test, y_test = create_dataset(
    target_function=benchmark_func,
    num_samples=1000,
    input_range=(-2.0, 2.0),
    input_dim=2,
    batch_size=256
)

# 学習
train_losses, val_losses = train_gpkan(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=1000,
    learning_rate=0.1
)

# 評価
results = evaluate_model(model, test_loader, y_test)
```

#### `visualization_utils.py` - 可視化ユーティリティ
学習曲線、予測結果、GP活性化関数などの可視化。

**主要関数**:
- `plot_learning_curves(...)`: 学習曲線
- `plot_prediction_results_2d(...)`: 2D関数の予測結果
- `visualize_gp_activation_functions(...)`: GP活性化関数

**使用例**:
```python
from gpkan_modules import (
    plot_learning_curves,
    plot_prediction_results_2d,
    visualize_gp_activation_functions
)

# 学習曲線
plot_learning_curves(train_losses, val_losses)

# 予測結果
plot_prediction_results_2d(
    model=model,
    target_function=benchmark_func,
    X_test=X_test,
    y_test=y_test,
    predictions=results['predictions'],
    variances=results['variances'],
    r2_score=results['r2'],
    rmse=results['rmse']
)

# GP活性化関数
visualize_gp_activation_functions(model, input_range=(-2.0, 2.0))
```

### `configs/` - 設定モジュール

#### `experiment_config.py` - 実験設定
datclassを使った型安全な設定管理。

**設定クラス**:
- `TrainingConfig`: 学習設定
- `ModelConfig`: モデル設定
- `DataConfig`: データ設定
- `ExperimentConfig`: 統合設定

**使用例**:
```python
from configs import ExperimentConfig, TrainingConfig, ModelConfig, DataConfig

config = ExperimentConfig(
    training=TrainingConfig(
        learning_rate=0.1,
        batch_size=256,
        num_epochs=1000
    ),
    model=ModelConfig(
        architecture=[2, 1],
        num_inducing=16
    ),
    data=DataConfig(
        num_samples=1000,
        benchmark_name="sphere"
    )
)

config.print_config()
```

### `benchmarks/` - ベンチマーク関数

#### `bbob_functions.py` - BBOB関数群
Black-Box Optimization Benchmarking 関数の実装。

**現在利用可能**:
- `BBOBSphere`: Sphere関数 f₁(x) = ||x - x_opt||² + f_opt

**使用例**:
```python
from benchmarks import BBOBSphere
import numpy as np

# Sphere関数の作成
sphere = BBOBSphere(
    dimension=2,
    x_opt=[0.0, 0.0],
    f_opt=10.0
)

# 関数評価
x = np.array([[1.0, 1.0], [2.0, 2.0]])
y = sphere(x)
print(y)  # [12. 18.]

# 情報表示
print(sphere.info())
```

## 🚀 使い方

### 基本的な実験の実行

```python
# 必要なモジュールをインポート
from gpkan_modules import GPKAN, set_seed, create_dataset, train_gpkan, evaluate_model
from configs import ExperimentConfig
from benchmarks import BBOBSphere

# 1. 乱数シード設定
set_seed(42)

# 2. 設定の作成
config = ExperimentConfig()
config.print_config()

# 3. ベンチマーク関数の作成
benchmark = BBOBSphere(dimension=2, x_opt=[0, 0], f_opt=10.0)

# 4. データセット作成
train_loader, val_loader, test_loader, X_test, y_test = create_dataset(
    target_function=benchmark,
    num_samples=config.data.num_samples,
    input_range=config.data.input_range,
    input_dim=2,
    batch_size=config.training.batch_size
)

# 5. モデル作成
model = GPKAN(
    layer_sizes=config.model.architecture,
    num_inducing=config.model.num_inducing,
    inducing_range=config.model.inducing_range
)

# 6. 学習
train_losses, val_losses = train_gpkan(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=config.training.num_epochs,
    learning_rate=config.training.learning_rate
)

# 7. 評価
results = evaluate_model(model, test_loader, y_test)
```

### カスタマイズ例

#### 異なるハイパーパラメータで実験

```python
from configs import ExperimentConfig, TrainingConfig, ModelConfig

custom_config = ExperimentConfig(
    training=TrainingConfig(
        learning_rate=0.05,      # 学習率変更
        batch_size=128,          # バッチサイズ変更
        num_epochs=2000          # エポック数増加
    ),
    model=ModelConfig(
        architecture=[2, 4, 1],  # 中間層追加
        num_inducing=32          # 誘導点数増加
    )
)
```

#### 新しいベンチマーク関数の追加

`benchmarks/bbob_functions.py` に新しいクラスを追加:

```python
class BBOBRosenbrock:
    """Rosenbrock関数: f₂(x) = sum(100*(x[i+1] - x[i]²)² + (x[i] - 1)²)"""
    
    def __init__(self, dimension: int = 2, f_opt: float = 0.0):
        self.dimension = dimension
        self.f_opt = f_opt
    
    def __call__(self, x):
        # 実装...
        pass
```

## 📊 実験結果の保存

モデルや結果を保存する例:

```python
import torch

# モデルの保存
torch.save(model.state_dict(), 'model_weights.pt')

# 学習履歴の保存
import json
with open('training_history.json', 'w') as f:
    json.dump({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config.__dict__
    }, f, indent=2)

# モデルの読み込み
model = GPKAN(layer_sizes=[2, 1], num_inducing=16)
model.load_state_dict(torch.load('model_weights.pt'))
```

## 🔧 依存関係

```
torch>=1.9.0
gpytorch>=1.6.0
numpy>=1.19.0
matplotlib>=3.3.0
japanize-matplotlib>=1.1.0
```

## 📝 元ノートブックとの対応

| 元ノートブックのセクション | モジュール化後の場所 |
|------------------------|------------------|
| ハイパーパラメータ設定 | `configs/experiment_config.py` |
| GPKANNode クラス | `gpkan_modules/gpkan_node.py` |
| GPKANLayer クラス | `gpkan_modules/gpkan_layer.py` |
| GPKAN クラス | `gpkan_modules/gpkan_model.py` |
| ベンチマーク関数定義 | `benchmarks/bbob_functions.py` |
| データセット作成 | `gpkan_modules/training_utils.py` の `create_dataset()` |
| 学習ループ | `gpkan_modules/training_utils.py` の `train_gpkan()` |
| 評価 | `gpkan_modules/training_utils.py` の `evaluate_model()` |
| 可視化関数群 | `gpkan_modules/visualization_utils.py` |

## 💡 利点

1. **再利用性**: 各モジュールを他のプロジェクトで再利用可能
2. **保守性**: 各機能が独立しているため、バグ修正や改善が容易
3. **拡張性**: 新しいベンチマーク関数やモデル構造の追加が簡単
4. **テスト性**: 個別のモジュールに対してユニットテストが可能
5. **可読性**: コードの構造が明確で理解しやすい
6. **設定管理**: ハイパーパラメータがdataclassで型安全に管理される

## 🔍 次のステップ

1. **他のベンチマーク関数の追加**: BBOB f₂〜f₂₄ の実装
2. **ユニットテストの作成**: 各モジュールのテスト
3. **実験結果の自動ログ**: Weights & Biases 等との連携
4. **複数実験の並列実行**: 異なる設定での比較実験
5. **モデルの最適化**: より効率的な実装の検討

## 📧 参考

元のノートブック: `demo/3x_COCO_fitting_evaluation/temp_31_3_fitting_bbob_gpkan_v2.ipynb`
