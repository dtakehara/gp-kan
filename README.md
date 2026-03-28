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

## 最初に読むノートブック

### 1. 単一ニューロンの動作確認
[simulations/gp_neuron_expr.ipynb](simulations/gp_neuron_expr.ipynb)

GP ニューロンが入力分布を受け取り、出力分布を計算する過程を検証しています。理論の基礎を手を動かして確認するのに最適です。

### 2. MNIST 分類デモ
[simulations/MNIST/mnist.ipynb](simulations/MNIST/mnist.ipynb)

GP-KAN を使った手書き数字分類のデモです。モデルが実際に動く様子を確認できます。

### 3. 関数フィッティング実験
[demo/4x_fitting_experiment/](demo/4x_fitting_experiment/)

物理方程式（Feynman データセット）や最適化ベンチマーク（BBOB）への当てはめ実験です。コードが整理されており、実装の参照にも適しています。

---

## 実験の流れ

実験は番号順に段階を踏んで進んでいます。

| ディレクトリ | 内容 |
|---|---|
| [demo/0x](demo/0x_kernel_rank_plot/), [1x](demo/1x_kernel_rank_analysis/) | GP の計算で使うカーネル行列の数値安定性を解析 |
| [demo/2x](demo/2x_variational_learning/) | 変分推論の実装。ベースライン → Sparse GP → Deep GP → GPyTorch と段階的に比較 |
| [demo/3x](demo/3x_COCO_fitting_evaluation/) | BBOB ベンチマークで SparseGP / DeepGP / GP-KAN の近似精度を比較 |
| [demo/4x](demo/4x_fitting_experiment/) | コードをモジュール化し、物理方程式・ベンチマーク関数で性能を体系的に評価 |

---

## コード構成

```
gpkan/
├── lib/                    コアライブラリ
│   ├── gp_dist.py          ガウス分布を表すデータ構造（平均・分散のペア）
│   ├── layer.py            GP ニューロン層の実装
│   ├── conv.py             GP ベースの畳み込み層
│   ├── model.py            モデルの組み立て・保存・読み込み
│   └── activations.py      ガウス分布のまま処理できる活性化関数群
├── simulations/            基礎的な動作確認ノートブック
└── demo/                   実験ノートブック群（番号順に発展）
    └── 4x_fitting_experiment/  最新のモジュール化された実装
        ├── gpkan_modules/      GP-KAN の実装本体
        ├── benchmarks/         ベンチマーク関数の定義
        └── configs/            実験設定の管理
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
