"""
Feynman Dataset精度検証スクリプト

Feynman Equationsを使用してGP-KANの精度を検証します。
各方程式のKAN shapeに基づいてモデルを構築し、学習・評価を行います。
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import gpytorch

warnings.filterwarnings("ignore")

# モジュールのインポート
from gpkan_modules import (
    GPKAN,
    set_seed,
    create_dataset,
    train_gpkan,
    evaluate_model,
)
from configs import ExperimentConfig, TrainingConfig, ModelConfig, DataConfig
from benchmarks import get_feynman_equation, list_available_equations

print("=" * 80)
print("Feynman Dataset 精度検証スクリプト")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"GPytorch version: {gpytorch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 80)


def generate_input_range(equation_id: str, dimension: int) -> tuple:
    """
    各方程式に適した入力範囲を生成
    
    Args:
        equation_id: 方程式ID
        dimension: 入力次元数
    
    Returns:
        (min, max) の入力範囲
    """
    # 方程式ごとにカスタマイズ可能
    if "I.6.2" in equation_id:
        return (-2.0, 2.0)  # ガウス分布系
    elif "I.26.2" in equation_id:
        return (0.1, 2.0)   # 三角関数系
    elif "I.27.6" in equation_id:
        return (1.1, 3.0)   # γが1より大きい必要がある
    else:
        return (-2.0, 2.0)  # デフォルト


def run_single_experiment(
    equation_id: str,
    config: ExperimentConfig,
    verbose: bool = True
) -> Dict:
    """
    単一のFeynman方程式に対して実験を実行
    
    Args:
        equation_id: 方程式ID (例: "I.6.2")
        config: 実験設定
        verbose: 詳細出力フラグ
    
    Returns:
        実験結果の辞書
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"実験開始: {equation_id}")
        print("=" * 80)
    
    # Feynman方程式の取得
    feynman_eq = get_feynman_equation(equation_id)
    
    if verbose:
        print(f"\n{feynman_eq.info()}\n")
    
    # 入力範囲の設定
    input_range = generate_input_range(equation_id, feynman_eq.dimension)
    
    # モデル設定の更新（KAN shapeを使用）
    config.model.architecture = feynman_eq.kan_shape
    config.data.num_samples = 2000  # Feynman用にサンプル数増加
    
    if verbose:
        print(f"入力次元: {feynman_eq.dimension}")
        print(f"KAN shape: {feynman_eq.kan_shape}")
        print(f"入力範囲: {input_range}")
        print(f"サンプル数: {config.data.num_samples}\n")
    
    # データセットの作成
    try:
        train_loader, val_loader, test_loader, X_test, y_test = create_dataset(
            target_function=feynman_eq,
            num_samples=config.data.num_samples,
            input_range=input_range,
            input_dim=feynman_eq.dimension,
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            batch_size=config.training.batch_size,
            random_seed=config.training.random_seed
        )
    except Exception as e:
        print(f"データセット作成エラー: {e}")
        return {
            "equation_id": equation_id,
            "status": "data_error",
            "error": str(e)
        }
    
    # モデルの作成
    try:
        model = GPKAN(
            layer_sizes=config.model.architecture,
            num_inducing=config.model.num_inducing,
            inducing_range=config.model.inducing_range
        )
        
        if verbose:
            model.print_model_info()
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}\n")
    except Exception as e:
        print(f"モデル作成エラー: {e}")
        return {
            "equation_id": equation_id,
            "status": "model_error",
            "error": str(e)
        }
    
    # 学習
    try:
        if verbose:
            print("=" * 80)
            print("学習開始")
            print("=" * 80)
        
        train_losses, val_losses = train_gpkan(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            scheduler_type=config.training.scheduler_type,
            eta_min_ratio=config.training.eta_min_ratio
        )
    except Exception as e:
        print(f"学習エラー: {e}")
        return {
            "equation_id": equation_id,
            "status": "training_error",
            "error": str(e)
        }
    
    # 評価
    try:
        if verbose:
            print("\n" + "=" * 80)
            print("評価開始")
            print("=" * 80)
        
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            y_test=y_test
        )
        
        # 結果の整理
        experiment_result = {
            "equation_id": equation_id,
            "status": "success",
            "dimension": feynman_eq.dimension,
            "kan_shape": feynman_eq.kan_shape,
            "num_samples": config.data.num_samples,
            "num_epochs": config.training.num_epochs,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "num_inducing": config.model.num_inducing,
            "metrics": {
                "rmse": results['rmse'],
                "mae": results['mae'],
                "r2": results['r2'],
                "mean_variance": results['mean_variance']
            },
            "training": {
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "min_val_loss": min(val_losses)
            }
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"実験完了: {equation_id}")
            print("=" * 80)
            print(f"RMSE: {results['rmse']:.6f}")
            print(f"MAE: {results['mae']:.6f}")
            print(f"R²: {results['r2']:.6f}")
            print("=" * 80)
        
        return experiment_result
        
    except Exception as e:
        print(f"評価エラー: {e}")
        return {
            "equation_id": equation_id,
            "status": "evaluation_error",
            "error": str(e)
        }


def run_all_experiments(
    equation_ids: List[str] = None,
    output_file: str = "feynman_results.json",
    verbose: bool = True
) -> List[Dict]:
    """
    複数のFeynman方程式に対して実験を実行
    
    Args:
        equation_ids: 実験する方程式IDのリスト（Noneの場合は全て）
        output_file: 結果を保存するJSONファイル
        verbose: 詳細出力フラグ
    
    Returns:
        実験結果のリスト
    """
    # 実験する方程式のリストを取得
    if equation_ids is None:
        equation_ids = list_available_equations()
    
    print(f"\n実験対象: {len(equation_ids)} 個の方程式")
    print(f"方程式: {', '.join(equation_ids)}\n")
    
    # 共通の実験設定
    base_config = ExperimentConfig(
        training=TrainingConfig(
            learning_rate=0.1,
            batch_size=256,
            num_epochs=500,  # Feynman用に調整
            scheduler_type="cosine",
            random_seed=42
        ),
        model=ModelConfig(
            architecture=[2, 1],  # 各方程式で上書きされる
            num_inducing=16,
            inducing_range=(-1.5, 1.5)
        ),
        data=DataConfig(
            num_samples=2000,
            input_range=(-2.0, 2.0)
        )
    )
    
    # 乱数シード設定
    set_seed(base_config.training.random_seed)
    
    # 全実験の結果を保存
    all_results = []
    
    # 各方程式に対して実験を実行
    for i, eq_id in enumerate(equation_ids, 1):
        print(f"\n{'#' * 80}")
        print(f"進捗: {i}/{len(equation_ids)}")
        print(f"{'#' * 80}")
        
        result = run_single_experiment(eq_id, base_config, verbose=verbose)
        all_results.append(result)
        
        # 中間結果を保存
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        if verbose and result['status'] == 'success':
            print(f"\n✓ {eq_id}: RMSE={result['metrics']['rmse']:.6f}, R²={result['metrics']['r2']:.6f}")
    
    # 最終結果のサマリー
    print("\n" + "=" * 80)
    print("全実験完了")
    print("=" * 80)
    
    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] != 'success']
    
    print(f"成功: {len(successful)}/{len(all_results)}")
    print(f"失敗: {len(failed)}/{len(all_results)}")
    
    if successful:
        print("\n" + "-" * 80)
        print("成功した実験の結果:")
        print("-" * 80)
        print(f"{'方程式ID':<15} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        print("-" * 80)
        for r in successful:
            print(f"{r['equation_id']:<15} "
                  f"{r['metrics']['rmse']:<12.6f} "
                  f"{r['metrics']['mae']:<12.6f} "
                  f"{r['metrics']['r2']:<10.6f}")
        print("-" * 80)
        
        # 統計
        rmse_values = [r['metrics']['rmse'] for r in successful]
        r2_values = [r['metrics']['r2'] for r in successful]
        
        print(f"\nRMSE - 平均: {np.mean(rmse_values):.6f}, 中央値: {np.median(rmse_values):.6f}")
        print(f"R²   - 平均: {np.mean(r2_values):.6f}, 中央値: {np.median(r2_values):.6f}")
    
    if failed:
        print("\n" + "-" * 80)
        print("失敗した実験:")
        print("-" * 80)
        for r in failed:
            print(f"{r['equation_id']}: {r['status']}")
            if 'error' in r:
                print(f"  エラー: {r['error']}")
    
    print(f"\n結果を {output_file} に保存しました。")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # コマンドライン引数の処理（オプション）
    import argparse
    
    parser = argparse.ArgumentParser(description="Feynman Dataset精度検証")
    parser.add_argument(
        "--equations",
        nargs="+",
        default=None,
        help="実験する方程式ID（例: I.6.2 I.9.18）"
    )
    parser.add_argument(
        "--output",
        default="feynman_results.json",
        help="結果を保存するファイル名"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="詳細出力を抑制"
    )
    
    args = parser.parse_args()
    
    # 実験実行
    results = run_all_experiments(
        equation_ids=args.equations,
        output_file=args.output,
        verbose=not args.quiet
    )
    
    print(f"\n全ての実験が完了しました！")
