# Sparse GP の変分ベイズ法：数式と実装の対応

## 1. 問題設定

観測データ $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ に対して、GPの周辺尤度 $p(\mathbf{y} | \mathbf{X})$ を直接計算すると $O(N^3)$ のコストがかかる。これを回避するために **誘導点 (inducing points)** $\mathbf{Z} = \{\mathbf{z}_m\}_{m=1}^M$ ($M \ll N$) と、対応する関数値 $\mathbf{u} = f(\mathbf{Z})$ を導入する。

## 2. 変分下界 (ELBO)

真の事後分布 $p(\mathbf{u} | \mathbf{y})$ を変分分布 $q(\mathbf{u})$ で近似する。対数周辺尤度に対する変分下界は：

$$\log p(\mathbf{y}) \geq \underbrace{\sum_{i=1}^N \mathbb{E}_{q(f_i)} \left[ \log p(y_i | f_i) \right]}_{\text{Expected Log Likelihood}} - \underbrace{\mathrm{KL}\left[ q(\mathbf{u}) \| p(\mathbf{u}) \right]}_{\text{KL正則化項}}$$

ここで $q(f_i) = \int p(f_i | \mathbf{u}) q(\mathbf{u}) \, d\mathbf{u}$ である。

## 3. 各項の具体形

### 変分分布（`CholeskyVariationalDistribution`）

$$q(\mathbf{u}) = \mathcal{N}(\mathbf{u} | \mathbf{m}, \mathbf{S}), \quad \mathbf{S} = \mathbf{L}\mathbf{L}^\top$$

$\mathbf{m}$（変分平均）と $\mathbf{L}$（Cholesky因子）が学習パラメータ。

### 条件付き分布（`VariationalStrategy`）

$$q(f_i) = \mathcal{N}\left( \mu_i, \sigma_i^2 \right)$$

$$\mu_i = K_{\mathbf{x}_i, \mathbf{Z}} K_{\mathbf{Z},\mathbf{Z}}^{-1} \mathbf{m}$$

$$\sigma_i^2 = K_{\mathbf{x}_i, \mathbf{x}_i} - K_{\mathbf{x}_i, \mathbf{Z}} K_{\mathbf{Z},\mathbf{Z}}^{-1} \left( K_{\mathbf{Z},\mathbf{Z}} - \mathbf{S} \right) K_{\mathbf{Z},\mathbf{Z}}^{-1} K_{\mathbf{Z}, \mathbf{x}_i}$$

### 尤度（`GaussianLikelihood`）

$$p(y_i | f_i) = \mathcal{N}(y_i | f_i, \sigma_n^2)$$

$\sigma_n^2$ は観測ノイズ分散（学習パラメータ）。

### KL項

2つの多変量正規分布間のKLダイバージェンス：

$$\mathrm{KL}[q(\mathbf{u}) \| p(\mathbf{u})] = \frac{1}{2} \left[ \mathrm{tr}(K_{\mathbf{Z},\mathbf{Z}}^{-1} \mathbf{S}) + \mathbf{m}^\top K_{\mathbf{Z},\mathbf{Z}}^{-1} \mathbf{m} - M + \log \frac{|K_{\mathbf{Z},\mathbf{Z}}|}{|\mathbf{S}|} \right]$$

## 4. 実装との対応

| 数式 | GPytorch 実装 | コード箇所 |
|---|---|---|
| $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{LL}^\top)$ | `CholeskyVariationalDistribution` | `sparse_gp_model.py` L26-27 |
| $p(f_* \| \mathbf{u})$ による予測分布の導出 | `VariationalStrategy` | `sparse_gp_model.py` L29-34 |
| $\mathbf{Z}$（誘導点、学習可能） | `learn_inducing_locations=True` | `sparse_gp_model.py` L33 |
| $k(\cdot, \cdot) = \sigma_f^2 \cdot k_{\mathrm{RBF}}(\cdot, \cdot)$ | `ScaleKernel(RBFKernel(ard_num_dims=...))` | `sparse_gp_model.py` L38 |
| $p(y \| f) = \mathcal{N}(y \| f, \sigma_n^2)$ | `GaussianLikelihood` | `sparse_gp_model.py` L85 |
| $\mathrm{ELBO} = \sum \mathbb{E}[\log p(y_i \| f_i)] - \mathrm{KL}$ | `VariationalELBO(likelihood, model, num_data=N)` | `sparse_gp_model.py` L100 |
| 損失 $= -\mathrm{ELBO}$ の最小化 | `-self.mll(self.model(x), y)` | `sparse_gp_model.py` L101 |

## 5. 学習パラメータ一覧

| パラメータ | 記号 | 意味 |
|---|---|---|
| `variational_distribution.variational_mean` | $\mathbf{m}$ | 変分平均 |
| `variational_distribution.chol_variational_covar` | $\mathbf{L}$ | 変分共分散のCholesky因子 |
| `variational_strategy.inducing_points` | $\mathbf{Z}$ | 誘導点位置 |
| `covar_module.outputscale` | $\sigma_f^2$ | カーネル出力スケール |
| `covar_module.base_kernel.lengthscale` | $\ell$ | RBFの長さスケール (ARD) |
| `mean_module.constant` | $\mu_0$ | 平均関数の定数 |
| `likelihood.noise` | $\sigma_n^2$ | 観測ノイズ分散 |

## 6. 補足

- この実装は **Titsias (2009) / Hensman et al. (2013)** の Sparse Variational GP の標準的な定式化に従っている。
- `VariationalELBO` が ELBO を計算し、その負値を最小化することで変分パラメータ・カーネルハイパーパラメータ・誘導点位置を同時に最適化する。
- `num_data=x.size(0)` はミニバッチ学習時にELBOのスケーリング補正に使われる。フルバッチ学習であれば現在の実装で問題ない。ミニバッチ学習に切り替える場合は、ここに **全データ数 N** を渡す必要がある。

## 参考文献

- Titsias, M. (2009). "Variational Learning of Inducing Variables in Sparse Gaussian Processes." AISTATS.
- Hensman, J., Fusi, N., & Lawrence, N. D. (2013). "Gaussian Processes for Big Data." UAI.
