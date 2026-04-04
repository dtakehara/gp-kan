"""
Configuration: 実験設定ファイル

GP-KANモデルの学習に関するハイパーパラメータを定義します。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    """学習設定"""
    
    # 学習パラメータ
    learning_rate: float = 0.1
    batch_size: int = 256
    num_epochs: int = 1000
    
    # スケジューラ設定
    scheduler_type: str = "cosine"  # "cosine" or "plateau"
    eta_min_ratio: float = 0.01     # CosineAnnealing用: 最小学習率の比率
    
    # データ分割
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 再現性
    random_seed: int = 42
    
    def __post_init__(self):
        """設定の整合性チェック"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "データ分割比率の合計が1.0である必要があります"
        assert 0 < self.eta_min_ratio < 1.0, \
            "eta_min_ratioは0と1の間である必要があります"


@dataclass
class ModelConfig:
    """モデル設定"""
    
    # ネットワーク構造
    architecture: list = None  # 例: [2, 1] は 2入力→1出力
    
    # GP設定
    num_inducing: int = 16
    inducing_range: Tuple[float, float] = (-1.5, 1.5)
    
    def __post_init__(self):
        """デフォルト値の設定"""
        if self.architecture is None:
            self.architecture = [2, 1]
        
        assert len(self.architecture) >= 2, \
            "architectureは少なくとも入力層と出力層の2つが必要です"
        assert self.num_inducing > 0, \
            "num_inducingは正の整数である必要があります"
        assert self.inducing_range[0] < self.inducing_range[1], \
            "inducing_rangeは (最小値, 最大値) の形式である必要があります"


@dataclass
class DataConfig:
    """データ設定"""
    
    # データ生成
    num_samples: int = 1000
    input_range: Tuple[float, float] = (-2.0, 2.0)
    
    # ベンチマーク関数設定
    benchmark_name: str = "sphere"  # "sphere" など
    benchmark_dimension: int = 2
    benchmark_x_opt: list = None  # 最適解位置（Noneの場合は原点）
    benchmark_f_opt: float = 10.0  # 最適値
    
    def __post_init__(self):
        """デフォルト値の設定"""
        if self.benchmark_x_opt is None:
            self.benchmark_x_opt = [0.0] * self.benchmark_dimension
        
        assert self.num_samples > 0, \
            "num_samplesは正の整数である必要があります"
        assert self.input_range[0] < self.input_range[1], \
            "input_rangeは (最小値, 最大値) の形式である必要があります"
        assert len(self.benchmark_x_opt) == self.benchmark_dimension, \
            "benchmark_x_optの次元数がbenchmark_dimensionと一致しません"


@dataclass
class ExperimentConfig:
    """実験全体の設定を統合"""
    
    training: TrainingConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        """デフォルトインスタンスの作成"""
        if self.training is None:
            self.training = TrainingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
    
    def print_config(self):
        """設定内容を表示"""
        print("=" * 60)
        print("実験設定")
        print("=" * 60)
        
        print("\n[学習設定]")
        print(f"  学習率: {self.training.learning_rate}")
        print(f"  バッチサイズ: {self.training.batch_size}")
        print(f"  エポック数: {self.training.num_epochs}")
        print(f"  スケジューラ: {self.training.scheduler_type}")
        print(f"  乱数シード: {self.training.random_seed}")
        
        print("\n[モデル設定]")
        print(f"  アーキテクチャ: {self.model.architecture}")
        print(f"  誘導点数: {self.model.num_inducing}")
        print(f"  誘導点範囲: {self.model.inducing_range}")
        
        print("\n[データ設定]")
        print(f"  サンプル数: {self.data.num_samples}")
        print(f"  入力範囲: {self.data.input_range}")
        print(f"  ベンチマーク関数: {self.data.benchmark_name}")
        print(f"  関数次元: {self.data.benchmark_dimension}")
        print(f"  最適解: {self.data.benchmark_x_opt}")
        print(f"  最適値: {self.data.benchmark_f_opt}")
        
        print("=" * 60)


# デフォルト設定の作成（インポート時に使用可能）
default_config = ExperimentConfig()
