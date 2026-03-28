# GP-KAN Modules: シンプルで統一されたインターフェース

3つのGaussian Processモデルを、共通の抽象基底クラスを用いて実装。

## アーキテクチャ

```
BaseGPModel (抽象基底クラス)
├── GPKAN (エッジベースGP)
├── SparseGP (変分GP)
└── DeepGP (多層GP)
```

## モデル

### 1. GPKAN
エッジベースのGPネットワーク。各入力-出力ノード間にGPエッジを配置。

### 2. SparseGP
誘導点を用いた効率的な変分GP。大規模データに対応。

### 3. DeepGP
複数のGP層を積み重ねた深層モデル。階層的な表現学習。

## 使用方法

```python
from gpkan_modules import GPKAN, SparseGP, DeepGP

# すべて同じインターフェース
model = GPKAN(layer_sizes=[2, 1], num_inducing=16)
# または
model = SparseGP(layer_sizes=[2, 1], num_inducing=16)
# または
model = DeepGP(layer_sizes=[2, 1], num_inducing=16)

# 学習
model.train_mode()
loss = model.get_total_loss(X, y)
loss.backward()

# 予測
model.eval_mode()
mean, var = model.predict(X_test)

# 情報表示
model.print_model_info()
```

## 共通インターフェース (BaseGPModel)

すべてのモデルは以下を実装：

| メソッド | 説明 | 戻り値 |
|---------|------|--------|
| `forward(x)` | 順伝播 | 予測値 |
| `predict(x)` | 予測（不確実性付き） | (平均, 分散) |
| `get_total_loss(x, y)` | 損失計算 | 損失値 |
| `train_mode()` | 学習モード設定 | - |
| `eval_mode()` | 評価モード設定 | - |
| `print_model_info()` | モデル情報表示 | - |

## コード比較

### リファクタリング前
- gpkan_model.py: 176行
- sparse_gp_model.py: 229行
- deep_gp_model.py: 322行
- **合計: 727行**

### リファクタリング後
- base_model.py: 109行 (新規)
- gpkan_model.py: 103行 (**-41%**)
- sparse_gp_model.py: 105行 (**-54%**)
- deep_gp_model.py: 158行 (**-51%**)
- **合計: 475行** (**-35%削減**)

## ファイル構成

```
gpkan_modules/
├── base_model.py           # 抽象基底クラス
├── gpkan_model.py          # GPKANモデル
├── sparse_gp_model.py      # SparseGPモデル
├── deep_gp_model.py        # DeepGPモデル
├── gpkan_layer.py          # GPKANレイヤー
├── gpkan_node.py           # GPKANノード
├── training_utils.py       # 学習ユーティリティ
├── visualization_utils.py  # 可視化ユーティリティ
└── __init__.py            # パッケージ初期化
```

## 改善点

### 1. 共通機能の統合
- `train_mode()`, `eval_mode()` を基底クラスに統合
- 形状調整ロジックをヘルパー関数化
- モデル情報表示の共通化

### 2. コードの簡潔化
- 冗長なdocstringを削減
- 重複コードを削除
- ネストを減らして可読性向上

### 3. 保守性の向上
- 単一責任原則に従った設計
- インターフェースの一貫性保証
- 拡張が容易な構造

## テスト

```bash
python test_model_interface.py
```

すべてのモデルで以下を確認：
- ✓ インターフェース互換性
- ✓ 学習・予測動作
- ✓ 出力形状の一貫性

## モデル選択ガイド

| モデル | 適用場面 | 利点 | 計算量 |
|--------|---------|------|--------|
| **GPKAN** | 小規模・解釈性重視 | 活性化関数の可視化 | 中 |
| **SparseGP** | 大規模データ | 高速・省メモリ | 低 |
| **DeepGP** | 複雑な非線形問題 | 表現力が高い | 高 |
