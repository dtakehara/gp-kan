# Feynman Dataset 精度検証

このディレクトリには、Feynman Datasetを使用したGP-KANの精度検証スクリプトが含まれています。

## 🎯 実装済みのFeynman方程式

以下の10個のFeynman方程式を実装済み：

| 方程式ID | 式 | 変数 | KAN Shape |
|---------|---|------|-----------|
| I.6.2 | exp(-θ²/(2σ²)) / √(2πσ²) | θ, σ | [2,2,1,1] |
| I.6.2b | exp(-(θ-θ₁)²/(2σ²)) / √(2πσ²) | θ, θ₁, σ | [3,2,2,1,1] |
| I.9.18 | (G*m₁*m₂)/(r₁-r₂)² | G, m₁, m₂, r₁, r₂ | [5,4,2,2,1,1] |
| I.12.11 | q(Eⱼ + Bvssinθ) | q, Eⱼ, B, v, s, θ | [6,2,2,1] |
| I.13.12 | Gm₁m₂(1/r₂ - 1/r₁) | G, m₁, m₂, r₁, r₂ | [5,2,1] |
| I.15.3x | (x-u*t)/√(1-(u/c)²) | x, u, t, c | [4,2,1] |
| I.16.6 | (u+v)/(1+u*v/c²) | u, v, c | [3,2,2,2,1] |
| I.18.4 | (m₁*r₁ + m₂*r₂)/(m₁ + m₂) | m₁, r₁, m₂, r₂ | [4,2,2,2,1,1] |
| I.26.2 | arcsin(n*sin(θ₂)) | n, θ₂ | [2,2,2,2,1,1] |
| I.27.6 | 1/(γ-1) * p*V | γ, p, V | [3,2,1] |

## 📁 新規追加ファイル

### `benchmarks/feynman_functions.py`
Feynman方程式の実装モジュール。

**主要クラス:**
- `FeynmanEquation`: 基底クラス
- `FeynmanI6_2`, `FeynmanI6_2b`, etc.: 各方程式の実装

**主要関数:**
- `get_feynman_equation(equation_id)`: IDから方程式を取得
- `list_available_equations()`: 利用可能な方程式リストを取得

### `validate_feynman.py`
Feynman Dataset精度検証スクリプト。

**主要機能:**
- 各方程式のKAN shapeに基づいたモデル構築
- 自動的なデータ生成と学習
- 複数方程式の一括実験
- 結果のJSON出力

## 🚀 使い方

### 基本的な使い方（全方程式を実験）

```bash
cd /Users/daisuke.takehara/Desktop/workdir_gpkan/gp-kan/demo/4x_fitting_experiment
python validate_feynman.py
```

### 特定の方程式のみ実験

```bash
python validate_feynman.py --equations I.6.2 I.9.18 I.26.2
```

### 詳細出力を抑制

```bash
python validate_feynman.py --quiet
```

### 結果ファイル名を指定

```bash
python validate_feynman.py --output my_results.json
```

### コマンドラインオプション

- `--equations`: 実験する方程式ID（スペース区切り）
- `--output`: 結果を保存するJSONファイル名（デフォルト: `feynman_results.json`）
- `--quiet`: 詳細出力を抑制

## 📊 出力形式

### コンソール出力

```
================================================================================
Feynman Dataset 精度検証スクリプト
================================================================================
PyTorch version: 2.x.x
GPytorch version: 1.x.x
...

実験開始: I.6.2
================================================================================
Feynman I.6.2
  Variables: θ, σ
  KAN shape: [2, 2, 1, 1]
  Formula: exp(-θ²/(2σ²)) / √(2πσ²)
...

================================================================================
全実験完了
================================================================================
成功: 10/10
失敗: 0/10

--------------------------------------------------------------------------------
成功した実験の結果:
--------------------------------------------------------------------------------
方程式ID         RMSE         MAE          R²        
--------------------------------------------------------------------------------
I.6.2           0.000123     0.000098     0.999876
...
```

### JSON出力（`feynman_results.json`）

```json
[
  {
    "equation_id": "I.6.2",
    "status": "success",
    "dimension": 2,
    "kan_shape": [2, 2, 1, 1],
    "num_samples": 2000,
    "num_epochs": 500,
    "learning_rate": 0.1,
    "batch_size": 256,
    "num_inducing": 16,
    "metrics": {
      "rmse": 0.000123,
      "mae": 0.000098,
      "r2": 0.999876,
      "mean_variance": 0.000001
    },
    "training": {
      "final_train_loss": 0.0012,
      "final_val_loss": 0.0013,
      "min_val_loss": 0.0011
    }
  },
  ...
]
```

## ⚙️ ハイパーパラメータ

全方程式で共通のハイパーパラメータを使用：

```python
TrainingConfig(
    learning_rate=0.1,
    batch_size=256,
    num_epochs=500,
    scheduler_type="cosine",
    random_seed=42
)

ModelConfig(
    architecture=[...],  # 各方程式のKAN shapeを使用
    num_inducing=16,
    inducing_range=(-1.5, 1.5)
)

DataConfig(
    num_samples=2000,
    input_range=(-2.0, 2.0)  # 方程式ごとに調整される場合あり
)
```

## 🔧 カスタマイズ

### 新しいFeynman方程式の追加

`benchmarks/feynman_functions.py` に新しいクラスを追加：

```python
class FeynmanI30_3(FeynmanEquation):
    """新しい方程式"""
    
    def __init__(self):
        super().__init__("I.30.3", ["x", "y", "z"], [3, 2, 1])
    
    def __call__(self, x: Union[np.ndarray, list]) -> Union[np.ndarray, float]:
        # 実装
        pass
    
    def info(self) -> str:
        return "方程式の説明"

# レジストリに追加
FEYNMAN_EQUATIONS["I.30.3"] = FeynmanI30_3
```

### 入力範囲のカスタマイズ

`validate_feynman.py` の `generate_input_range()` 関数を編集：

```python
def generate_input_range(equation_id: str, dimension: int) -> tuple:
    if equation_id == "I.30.3":
        return (0.0, 10.0)  # カスタム範囲
    # ...
```

### ハイパーパラメータの変更

`validate_feynman.py` の `run_all_experiments()` 内の `base_config` を編集。

## 📈 結果の分析

結果JSONファイルを読み込んで分析：

```python
import json
import pandas as pd

# 結果の読み込み
with open('feynman_results.json', 'r') as f:
    results = json.load(f)

# DataFrameに変換
df = pd.DataFrame([
    {
        'equation': r['equation_id'],
        'rmse': r['metrics']['rmse'],
        'r2': r['metrics']['r2'],
        'dimension': r['dimension']
    }
    for r in results if r['status'] == 'success'
])

# 統計表示
print(df.describe())

# 相関分析
print(df.corr())
```

## 🎨 可視化

個別の方程式について詳細な可視化を行う場合：

```python
from validate_feynman import run_single_experiment
from gpkan_modules import visualize_gp_activation_functions
from configs import ExperimentConfig

# 実験実行
result = run_single_experiment("I.6.2", ExperimentConfig())

# モデルの可視化（実験中にモデルを保存する必要あり）
# visualize_gp_activation_functions(model, input_range=(-2, 2))
```

## 💡 注意点

1. **数値安定性**: 一部の方程式では `1e-10` を加算して0除算を防止
2. **入力範囲**: 方程式によって適切な入力範囲が異なる（特に三角関数や有理関数）
3. **KAN Shape**: 元論文の構造を忠実に再現
4. **計算時間**: 全方程式の実験は数時間かかる可能性あり

## 🔍 トラブルシューティング

### エラー: "Unknown equation ID"
→ `list_available_equations()` で利用可能な方程式を確認

### エラー: "data_error"
→ 入力範囲が不適切な可能性。`generate_input_range()` を調整

### エラー: "training_error"
→ ハイパーパラメータ（学習率、エポック数）を調整

### 低いR²スコア
→ エポック数を増やす、誘導点数を増やす、学習率を調整

## 📚 参考文献

- AI Feynman: A physics-inspired method for symbolic regression (Udrescu & Tegmark, 2020)
- The Feynman Lectures on Physics

## 次のステップ

1. **より多くの方程式の追加**: 画像の残りの方程式を実装
2. **ハイパーパラメータチューニング**: 方程式ごとに最適化
3. **アンサンブル学習**: 複数モデルの平均
4. **転移学習**: 類似した方程式間での知識転移
