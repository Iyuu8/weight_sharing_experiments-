"""
Master Runner for Weight-Sharing Research Experiments.
Executes:
1. Gradual Difficulty Benchmarks:
   - Level 1: 5-Arm Spiral (Geometric Depth Separation)
   - Level 2: 4x4 Checkerboard (Combinatorial Depth Separation)
   - Level 3: CIFAR-100 (Natural Image Hierarchical Depth Separation)
2. Study 1: Accuracy vs Original Network Size (Target Depth T)
3. Study 2: Accuracy vs Weight-Sharing Network Size (Bank Size n)
4. Comprehensive Publication-Quality Plots and Tabular Reports
"""

import sys
import os
import time
import argparse
import torch
import pandas as pd

import models
import datasets
import experiments
import plot_results

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Weight Sharing AI Experiments")
    parser.add_argument('--suite', type=str, default='all',
                        choices=['all', 'benchmarks', 'studies', 'cifar', 'synthetic'],
                        help="Suite to run: 'all', 'benchmarks', 'studies', 'cifar', 'synthetic'")
    parser.add_argument('--cifar-epochs', type=int, default=12,
                        help="Number of epochs for CIFAR-100 models (default: 12)")
    parser.add_argument('--synthetic-epochs', type=int, default=300,
                        help="Number of epochs for synthetic 2D models (default: 300)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use ('cuda' or 'cpu')")
    return parser.parse_args()


def build_synthetic_models(in_dim, out_dim, H=64, depth=6):
    """Builds the 10 model variants for synthetic tasks."""
    return {
        "Deep-Unshared": models.MLPDeepUnshared(in_dim, out_dim, H=H, depth=depth),
        "Shallow": models.MLPShallow(in_dim, out_dim, H=H),
        "Shared-Single": models.MLPSharedSingle(in_dim, out_dim, H=H, depth=depth, inject_input=True),
        "Shared-2-Cyclic": models.MLPSharedCyclic(in_dim, out_dim, H=H, n_layers=2, depth=depth, inject_input=True),
        "Shared-3-Cyclic": models.MLPSharedCyclic(in_dim, out_dim, H=H, n_layers=3, depth=depth, inject_input=True),
        "Internal-Router-Soft": models.MLPInternalRouter(in_dim, out_dim, H=H, n_bank=3, depth=depth, hard=False),
        "Internal-Router-Hard": models.MLPInternalRouter(in_dim, out_dim, H=H, n_bank=3, depth=depth, hard=True),
        "External-Controller-Soft": models.MLPExternalController(in_dim, out_dim, H=H, n_bank=3, depth=depth, hard=False),
        "External-Controller-Hard": models.MLPExternalController(in_dim, out_dim, H=H, n_bank=3, depth=depth, hard=True),
        "Adaptive-Depth-Controller": models.MLPAdaptiveDepthController(in_dim, out_dim, H=H, max_depth=depth),
    }


def build_cifar_models(num_classes=100, channels=48, depth=6):
    """Builds the 10 model variants for CIFAR-100 benchmark."""
    return {
        "Deep-Unshared": models.ConvDeepUnshared(num_classes=num_classes, channels=channels, depth=depth),
        "Shallow": models.ConvShallow(num_classes=num_classes, channels=channels),
        "Shared-Single": models.ConvSharedSingle(num_classes=num_classes, channels=channels, depth=depth),
        "Shared-2-Cyclic": models.ConvSharedCyclic(num_classes=num_classes, channels=channels, n_layers=2, depth=depth),
        "Shared-3-Cyclic": models.ConvSharedCyclic(num_classes=num_classes, channels=channels, n_layers=3, depth=depth),
        "Internal-Router-Soft": models.ConvInternalRouter(num_classes=num_classes, channels=channels, n_bank=3, depth=depth, hard=False),
        "Internal-Router-Hard": models.ConvInternalRouter(num_classes=num_classes, channels=channels, n_bank=3, depth=depth, hard=True),
        "External-Controller-Soft": models.ConvExternalController(num_classes=num_classes, channels=channels, n_bank=3, depth=depth, hard=False),
        "External-Controller-Hard": models.ConvExternalController(num_classes=num_classes, channels=channels, n_bank=3, depth=depth, hard=True),
        "Adaptive-Depth-Controller": models.ConvAdaptiveDepthController(num_classes=num_classes, channels=channels, max_depth=depth),
    }


def main():
    args = parse_args()
    device = args.device
    print("=" * 80, flush=True)
    print("        WEIGHT SHARING & DYNAMIC ROUTING EXPERIMENT SUITE", flush=True)
    print("=" * 80, flush=True)
    print(f"PyTorch Version: {torch.__version__}", flush=True)
    print(f"Device: {device}", flush=True)
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Suite Mode: {args.suite}", flush=True)
    print(f"Synthetic Epochs: {args.synthetic_epochs}", flush=True)
    print(f"CIFAR-100 Epochs: {args.cifar_epochs}", flush=True)
    print("=" * 80, flush=True)

    t_total_start = time.time()
    all_benchmark_results = {}

    # ------------------------------------------------------------------------
    # 1. Benchmark Level 1: 5-Arm Spiral (Geometric Depth Separation)
    # ------------------------------------------------------------------------
    if args.suite in ['all', 'benchmarks', 'synthetic']:
        print("\n>>> Level 1: 5-Arm Spiral Classification Benchmark <<<", flush=True)
        train_data, test_data, in_d, out_d = datasets.get_spiral_data(device=device)
        spiral_models = build_synthetic_models(in_d, out_d, H=64, depth=6)
        df_spiral, _ = experiments.run_task_benchmark(
            task_name="Level 1 - 5-Arm Spiral",
            model_dict=spiral_models,
            train_data=train_data,
            test_data=test_data,
            epochs=args.synthetic_epochs,
            device=device,
            is_conv=False,
            output_dir=RESULTS_DIR
        )
        all_benchmark_results["Level 1 - Spiral"] = df_spiral

    # ------------------------------------------------------------------------
    # 2. Benchmark Level 2: 4x4 Checkerboard (Combinatorial Depth Separation)
    # ------------------------------------------------------------------------
    if args.suite in ['all', 'benchmarks', 'synthetic']:
        print("\n>>> Level 2: 4x4 Checkerboard Classification Benchmark <<<", flush=True)
        train_data, test_data, in_d, out_d = datasets.get_checkerboard_data(device=device)
        checker_models = build_synthetic_models(in_d, out_d, H=64, depth=6)
        df_checker, _ = experiments.run_task_benchmark(
            task_name="Level 2 - 4x4 Checkerboard",
            model_dict=checker_models,
            train_data=train_data,
            test_data=test_data,
            epochs=args.synthetic_epochs,
            device=device,
            is_conv=False,
            output_dir=RESULTS_DIR
        )
        all_benchmark_results["Level 2 - Checkerboard"] = df_checker

    # ------------------------------------------------------------------------
    # 3. Size Studies
    # ------------------------------------------------------------------------
    if args.suite in ['all', 'studies', 'synthetic']:
        # Study 1: Accuracy vs Original Network Size (Target Depth T)
        df_depth = experiments.run_size_study_original_depth(
            depths=(2, 4, 6, 8, 10),
            epochs=args.synthetic_epochs,
            device=device,
            output_dir=RESULTS_DIR
        )

        # Study 2: Accuracy vs Weight-Sharing Network Size (Bank Size n)
        df_bank = experiments.run_size_study_weight_sharing_bank(
            bank_sizes=(1, 2, 3, 4, 6),
            target_depth=6,
            epochs=args.synthetic_epochs,
            device=device,
            output_dir=RESULTS_DIR
        )

    # ------------------------------------------------------------------------
    # 4. Benchmark Level 3: CIFAR-100 (Natural Image Hierarchical Depth)
    # ------------------------------------------------------------------------
    if args.suite in ['all', 'benchmarks', 'cifar']:
        print("\n>>> Level 3: CIFAR-100 Fine-Grained Image Benchmark <<<", flush=True)
        cifar_dataset, in_d, out_d = datasets.get_cifar100_fast(batch_size=256, device=device)
        cifar_models = build_cifar_models(num_classes=out_d, channels=48, depth=6)
        df_cifar, _ = experiments.run_task_benchmark(
            task_name="Level 3 - CIFAR-100",
            model_dict=cifar_models,
            train_data=cifar_dataset,
            test_data=None,
            epochs=args.cifar_epochs,
            device=device,
            is_conv=True,
            output_dir=RESULTS_DIR
        )
        all_benchmark_results["Level 3 - CIFAR-100"] = df_cifar

    # Multi-task summary plot
    summary_dict = {}
    p1 = os.path.join(RESULTS_DIR, "benchmark_level_1_-_5-arm_spiral.csv")
    p2 = os.path.join(RESULTS_DIR, "benchmark_level_2_-_4x4_checkerboard.csv")
    p3 = os.path.join(RESULTS_DIR, "benchmark_level_3_-_cifar-100.csv")
    if os.path.exists(p1):
        summary_dict["Level 1 - Spiral"] = pd.read_csv(p1)
    if os.path.exists(p2):
        summary_dict["Level 2 - Checkerboard"] = pd.read_csv(p2)
    if os.path.exists(p3):
        summary_dict["Level 3 - CIFAR-100"] = pd.read_csv(p3)

    if len(summary_dict) > 1:
        plot_results.plot_cross_task_summary(summary_dict, output_dir=RESULTS_DIR)

    total_time = time.time() - t_total_start
    print("\n" + "=" * 80, flush=True)
    print(f"EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES", flush=True)
    print(f"All figures, CSV tables, and Markdown reports saved in '{RESULTS_DIR}/'", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
