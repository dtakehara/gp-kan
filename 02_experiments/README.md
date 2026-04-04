# experiments

モジュール化された GP-KAN 実験実装。`demo/` のノートブック群とは独立して利用できます。

## ディレクトリ構成

```
experiments/
└── gpkan_modules/          GP-KAN の実装本体
    ├── base_model.py       抽象基底クラス
    ├── gpkan_model.py      GPKANモデル
    ├── sparse_gp_model.py  SparseGPモデル
    ├── deep_gp_model.py    DeepGPモデル
    ├── gpkan_layer.py      GPKANレイヤー
    ├── gpkan_node.py       GPKANノード
    ├── training_utils.py   学習ユーティリティ（create_dataset, train_gpkan, evaluate_model）
    ├── visualization_utils.py  可視化ユーティリティ
    └── __init__.py
```

## モデル

3つのGaussian Processモデルを、共通の抽象基底クラスを用いて実装。

```
BaseGPModel (抽象基底クラス)
├── GPKAN      (エッジベースGP)
├── SparseGP   (変分GP)
└── DeepGP     (多層GP)
```

| モデル | 適用場面 | 利点 | 計算量 |
|--------|---------|------|--------|
| **GPKAN** | 小規模・解釈性重視 | 活性化関数の可視化 | 中 |
| **SparseGP** | 大規模データ | 高速・省メモリ | 低 |
| **DeepGP** | 複雑な非線形問題 | 表現力が高い | 高 |

## 共通インターフェース

| メソッド | 説明 | 戻り値 |
|---------|------|--------|
| `forward(x)` | 順伝播 | 予測値 |
| `predict(x)` | 予測（不確実性付き） | (平均, 分散) |
| `get_total_loss(x, y)` | 損失計算 | 損失値 |
| `train_mode()` | 学習モード設定 | - |
| `eval_mode()` | 評価モード設定 | - |
| `print_model_info()` | モデル情報表示 | - |

## 使用例

```python
from gpkan_modules import GPKAN, SparseGP, DeepGP
from gpkan_modules import set_seed, create_dataset, train_gpkan, evaluate_model

set_seed(42)

# データセット作成
train_loader, val_loader, test_loader, X_test, y_test = create_dataset(
    target_function=my_func,
    num_samples=1000,
    input_range=(-2.0, 2.0),
    input_dim=2,
    batch_size=256
)

# モデル作成・学習・評価
model = GPKAN(layer_sizes=[2, 1], num_inducing=16)

model.train_mode()
train_losses, val_losses = train_gpkan(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=1000,
    learning_rate=0.1
)

model.eval_mode()
results = evaluate_model(model, test_loader, y_test)
mean, var = model.predict(X_test)
```
