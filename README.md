# GP-KAN: Gaussian Process Kolmogorov-Arnold Network

「予測値」だけでなく「その予測の信頼度」も同時に出力できるニューラルネットワークの研究実装です。

---

## 概要

GP-KAN は、KAN（Kolmogorov-Arnold Network）の各活性化関数をガウス過程（GP）でモデル化したニューラルネットワークです。ネットワーク内部を流れる情報を「値」ではなく「確率分布（平均 + 分散のペア）」として扱うため、入力の不確実性が層を超えて伝播し、最終的な出力でも予測値と不確実性がセットで得られます。

```
入力 x → (平均, 分散) → [GP Layer 1] → (平均, 分散) → [GP Layer 2] → 出力 (予測値, 不確実性)
```

---

## 環境構築

```bash
git clone <repo-url>
cd gpkan
pip install -r requirements.txt
```

主な依存パッケージ: `torch`, `gpytorch`, `numpy`, `scipy`, `matplotlib`

---

## コード構成

```
gpkan/
├── lib/                        コアライブラリ
│   ├── gp_dist.py              ガウス分布を表すデータ構造（平均・分散のペア）
│   ├── layer.py                GP ニューロン層の実装
│   ├── conv.py                 GP ベースの畳み込み層
│   ├── model.py                モデルの組み立て・保存・読み込み
│   └── activations.py          ガウス分布のまま処理できる活性化関数群
├── simulations/                基礎的な動作確認ノートブック
└── experiments/                モジュール化された実験実装
    └── gpkan_modules/          GP-KAN の実装本体
```

---

## experiments の詳細

### gpkan_modules — モデル実装

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

すべてのモデルは以下の共通インターフェースを実装:

| メソッド | 説明 | 戻り値 |
|---------|------|--------|
| `forward(x)` | 順伝播 | 予測値 |
| `predict(x)` | 予測（不確実性付き） | (平均, 分散) |
| `get_total_loss(x, y)` | 損失計算 | 損失値 |
| `train_mode()` | 学習モード設定 | - |
| `eval_mode()` | 評価モード設定 | - |
| `print_model_info()` | モデル情報表示 | - |

```python
from gpkan_modules import GPKAN, SparseGP, DeepGP

model = GPKAN(layer_sizes=[2, 1], num_inducing=16)
model.train_mode()
loss = model.get_total_loss(X, y)
loss.backward()

model.eval_mode()
mean, var = model.predict(X_test)
```

ファイル構成:

```
gpkan_modules/
├── base_model.py           抽象基底クラス
├── gpkan_model.py          GPKANモデル
├── sparse_gp_model.py      SparseGPモデル
├── deep_gp_model.py        DeepGPモデル
├── gpkan_layer.py          GPKANレイヤー
├── gpkan_node.py           GPKANノード
├── training_utils.py       学習ユーティリティ（create_dataset, train_gpkan, evaluate_model）
├── visualization_utils.py  可視化ユーティリティ（学習曲線・予測結果・GP活性化関数）
└── __init__.py
```

---

## 用語説明

| 用語 | 説明 |
|---|---|
| ガウス過程 (GP) | 関数を確率的に表現する手法。「予測値 ± 不確かさ」をセットで扱える |
| 誘導点 (Inducing Points) | GP の計算量を削減する近似手法。代表点を少数選んで GP を近似する |
| 変分推論 | 厳密な計算が難しいベイズ推論を、扱いやすい分布で近似して解く手法 |
| BBOB / Feynman | 性能評価用のベンチマーク。それぞれ最適化テスト関数群・物理方程式群からなる |

---

## ライセンス

GPL-3.0
