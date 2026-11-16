"""
Plotting and Visualization Module for Weight Sharing Experiments.
Generates publication-quality figures:
1. Benchmark comparisons across difficulty levels
2. Accuracy vs Parameter Count (Pareto frontier)
3. Study of Original Network Size (Target Depth T)
4. Study of Weight-Sharing Network Size (Bank Size n)
5. Accuracy deltas vs Baselines (Deep-Unshared Reference & Shallow Control in percentage points)
6. Training and Convergence Dynamics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_task_benchmark(df, task_name, output_dir='results'):
    """
    Plots benchmark results for a single task:
    Left: Test Accuracy Bar Chart (Baselines vs Weight Sharing Variants)
    Right: Accuracy vs Parameter Count (Pareto Frontier)
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = []
    for model in df['Model']:
        if 'Deep-Unshared' in model:
            colors.append('#2b2b2b') # Dark charcoal reference
        elif 'Shallow' in model:
            colors.append('#e74c3c') # Red for shallow control
        elif 'Single' in model:
            colors.append('#3498db') # Blue
        elif 'Cyclic' in model:
            colors.append('#2ecc71') # Green
        elif 'Internal' in model:
            colors.append('#9b59b6') # Purple
        elif 'External' in model:
            colors.append('#e67e22') # Orange
        elif 'Adaptive' in model:
            colors.append('#1abc9c') # Teal
        else:
            colors.append('#95a5a6')

    # Panel 1: Bar chart of Accuracy
    ax = axes[0]
    bars = ax.barh(df['Model'], df['Test Acc (%)'], color=colors, edgecolor='none', height=0.65)
    ax.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{task_name} — Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, max(df['Test Acc (%)']) * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.8, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold', color='#333333')

    # Panel 2: Pareto Frontier (Accuracy vs Params)
    ax2 = axes[1]
    
    # Custom annotation offsets to prevent label collision between nearby models
    stagger_offsets = {
        'Shallow': (-10, -14),
        'Shared-Single': (8, 6),
        'Adaptive-Depth-Controller': (8, -12),
        'Adaptive-Depth': (8, -12),
        'Shared-2-Cyclic': (8, 6),
        'Shared-3-Cyclic': (8, 6),
        'Internal-Router-Soft': (8, 8),
        'Internal-Router-Hard': (8, -12),
        'External-Controller-Soft': (8, 8),
        'External-Controller-Hard': (8, -14),
        'Deep-Unshared': (-10, 10),
    }

    for idx, row in df.iterrows():
        model = row['Model']
        params = row['Params']
        acc = row['Test Acc (%)']
        color = colors[idx]
        marker = 's' if 'Baseline' in model or 'Unshared' in model or 'Shallow' in model else 'o'
        size = 140 if marker == 's' else 110
        ax2.scatter(params, acc, color=color, s=size, edgecolors='black', linewidth=1.2, zorder=4)
        
        dx, dy = stagger_offsets.get(model, (8, 4))
        ha_align = 'right' if dx < 0 else 'left'
        ax2.annotate(model, (params, acc), textcoords="offset points", 
                     xytext=(dx, dy), ha=ha_align, fontsize=8.5, fontweight='semibold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8, linewidth=0.5))

    ax2.set_xlabel('Trainable Parameters', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{task_name} — Accuracy vs Parameter Efficiency', fontsize=14, fontweight='bold', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Extra margin for annotations
    p_min, p_max = df['Params'].min(), df['Params'].max()
    ax2.set_xlim(max(0, p_min * 0.7), p_max * 1.15)
    a_min, a_max = df['Test Acc (%)'].min(), df['Test Acc (%)'].max()
    ax2.set_ylim(max(0, a_min - 5), min(105, a_max + 6))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'benchmark_{task_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved benchmark plot: {plot_path}")


def plot_cross_task_summary(all_results_dict, output_dir='results'):
    """
    Plots a multi-panel comparison across all tasks in explicit order:
    Level 1 (Spiral) -> Level 2 (Checkerboard) -> Level 3 (CIFAR-100).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def task_order_key(t):
        if '1' in t or 'Spiral' in t:
            return 1
        elif '2' in t or 'Checkerboard' in t:
            return 2
        elif '3' in t or 'CIFAR' in t:
            return 3
        return 99

    tasks = sorted(all_results_dict.keys(), key=task_order_key)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 7), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        df = all_results_dict[task]
        colors = ['#2b2b2b' if 'Deep-Unshared' in m else '#e74c3c' if 'Shallow' in m else '#3498db' for m in df['Model']]
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['Test Acc (%)'], color=colors, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Model'], fontsize=10)
        ax.set_xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(task, fontsize=13, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(0, max(df['Test Acc (%)']) * 1.15)
        for i, val in enumerate(df['Test Acc (%)']):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Weight Sharing Performance Across Tasks with Increasing Difficulty', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'cross_task_benchmarks_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-task summary plot: {summary_path}")


def plot_size_study_depth(df_depth, output_dir='results'):
    """
    Plots the study of accuracy and parameters as a function of
    Original Network Size (Target Depth T).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Accuracy vs Target Depth T
    ax1 = axes[0]
    palette = {'Deep-Unshared': '#2b2b2b', 'Shallow': '#e74c3c', 'Shared-Single': '#3498db',
               'Shared-2-Cyclic': '#2ecc71', 'Adaptive-Depth': '#1abc9c'}
    
    for model_name, group in df_depth.groupby('Model'):
        color = palette.get(model_name, '#95a5a6')
        linestyle = '--' if 'Shallow' in model_name else '-'
        marker = 's' if 'Baseline' in model_name or 'Unshared' in model_name else 'o'
        ax1.plot(group['Target Depth (T)'], group['Test Acc (%)'], marker=marker,
                 linewidth=2.5, markersize=8, label=model_name, color=color, linestyle=linestyle)

    ax1.set_xlabel('Target Depth $T$ (Effective Layers Applied)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Study 1: Accuracy vs Original Network Depth ($T$)', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(frameon=True, facecolor='white', loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Parameter Count vs Target Depth T
    ax2 = axes[1]
    for model_name, group in df_depth.groupby('Model'):
        color = palette.get(model_name, '#95a5a6')
        linestyle = '--' if 'Shallow' in model_name else '-'
        marker = 's' if 'Baseline' in model_name or 'Unshared' in model_name else 'o'
        ax2.plot(group['Target Depth (T)'], group['Params'], marker=marker,
                 linewidth=2.5, markersize=8, label=model_name, color=color, linestyle=linestyle)

    ax2.set_xlabel('Target Depth $T$ (Effective Layers Applied)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Trainable Parameters (Count)', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Scaling vs Original Network Depth ($T$)', fontsize=14, fontweight='bold', pad=12)
    ax2.legend(frameon=True, facecolor='white', loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    depth_plot_path = os.path.join(output_dir, 'study_original_network_size_depth.png')
    plt.savefig(depth_plot_path, dpi=300)
    plt.close()
    print(f"Saved depth study plot: {depth_plot_path}")


def plot_size_study_bank(df_bank, output_dir='results'):
    """
    Plots the study of accuracy and compression as a function of
    Weight Sharing Network Size (Bank Size n).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Accuracy vs Bank Size n
    ax1 = axes[0]
    ax1.plot(df_bank['Bank Size (n)'], df_bank['Test Acc (%)'], marker='o',
             color='#3498db', linewidth=2.5, markersize=9, label='Shared-N-Cyclic')
    
    # Highlight reference and shallow
    ceiling_col = 'Ceiling Acc' if 'Ceiling Acc' in df_bank.columns else 'Deep Ref Acc'
    if ceiling_col in df_bank.columns and len(df_bank[ceiling_col]) > 0:
        ceiling_acc = df_bank[ceiling_col].iloc[0]
        ax1.axhline(ceiling_acc, color='#2b2b2b', linestyle='--', linewidth=2, label=f'Deep-Unshared Reference ({ceiling_acc:.1f}%)')
    
    if 'Shallow Acc' in df_bank.columns and len(df_bank['Shallow Acc']) > 0:
        shallow_acc = df_bank['Shallow Acc'].iloc[0]
        ax1.axhline(shallow_acc, color='#e74c3c', linestyle=':', linewidth=2, label=f'Shallow Control ({shallow_acc:.1f}%)')

    ax1.set_xlabel('Weight Sharing Bank Size $n$ (Unique Shared Layers)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Study 2: Accuracy vs Weight-Sharing Bank Size ($n$)', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(frameon=True, facecolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Accuracy vs Parameter Compression Ratio
    ax2 = axes[1]
    scatter = ax2.scatter(df_bank['Compression Ratio'], df_bank['Test Acc (%)'],
                          c=df_bank['Bank Size (n)'], cmap='viridis', s=160, edgecolors='black', linewidth=1.2, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Bank Size $n$', fontsize=11, fontweight='bold')

    # Extend x-limits so n=1 label is never clipped
    max_comp = df_bank['Compression Ratio'].max()
    ax2.set_xlim(0.8, max_comp * 1.35)

    for _, row in df_bank.iterrows():
        p_int = int(row['Params'])
        ax2.annotate(f"n={int(row['Bank Size (n)'])} ({p_int:,} params)",
                     (row['Compression Ratio'], row['Test Acc (%)']),
                     textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight='semibold')

    ax2.set_xlabel('Parameter Compression Ratio vs Deep Reference', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Retention vs Parameter Compression', fontsize=14, fontweight='bold', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    bank_plot_path = os.path.join(output_dir, 'study_weight_sharing_size_bank.png')
    plt.savefig(bank_plot_path, dpi=300)
    plt.close()
    print(f"Saved bank size study plot: {bank_plot_path}")


def plot_baseline_contrasts(df, task_name, output_dir='results'):
    """
    Plots the accuracy delta of each weight-sharing model contrasted against:
    1) Deep-Unshared Reference (Full-Capacity Baseline)
    2) Shallow Control (Parameter-Matched Lower Bound)
    Reported in percentage points (pp).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_variants = df[~df['Model'].isin(['Deep-Unshared', 'Shallow'])].copy()
    if len(df_variants) == 0:
        return

    ceiling_acc = df[df['Model'] == 'Deep-Unshared']['Test Acc (%)'].values[0]
    shallow_acc = df[df['Model'] == 'Shallow']['Test Acc (%)'].values[0]

    df_variants['vs_ceiling'] = df_variants['Test Acc (%)'] - ceiling_acc
    df_variants['vs_shallow'] = df_variants['Test Acc (%)'] - shallow_acc

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: vs Deep-Unshared Reference
    ax1 = axes[0]
    colors1 = ['#2ecc71' if x >= 0 else '#e67e22' if x > -3.0 else '#e74c3c' for x in df_variants['vs_ceiling']]
    bars1 = ax1.barh(df_variants['Model'], df_variants['vs_ceiling'], color=colors1, height=0.6)
    ax1.axvline(0, color='#2b2b2b', linestyle='--', linewidth=1.5, label='Deep Reference (0.0 pp)')
    ax1.set_xlabel('Difference vs Deep-Unshared Reference (percentage points, pp)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{task_name} — Accuracy vs Deep Reference', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='lower left' if df_variants['vs_ceiling'].mean() < 0 else 'upper left')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    # Set x limits with margin to prevent label clipping/overlap
    min_c = df_variants['vs_ceiling'].min()
    max_c = df_variants['vs_ceiling'].max()
    margin_c = max(abs(min_c), abs(max_c), 5.0) * 0.25
    ax1.set_xlim(min(min_c - margin_c, -1.0), max(max_c + margin_c, 1.0))

    for bar in bars1:
        w = bar.get_width()
        if w >= 0:
            ax1.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:+.1f} pp',
                     va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            ax1.text(w - 0.3, bar.get_y() + bar.get_height()/2, f'{w:+.1f} pp',
                     va='center', ha='right', fontsize=9, fontweight='bold')

    # Panel 2: vs Shallow Control
    ax2 = axes[1]
    colors2 = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_variants['vs_shallow']]
    bars2 = ax2.barh(df_variants['Model'], df_variants['vs_shallow'], color=colors2, height=0.6)
    ax2.axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Shallow Control (0.0 pp)')
    ax2.set_xlabel('Advantage over Shallow Control (percentage points, pp)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{task_name} — Gain over Parameter-Matched Shallow Control', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='lower right' if df_variants['vs_shallow'].mean() > 0 else 'upper right')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    min_s = df_variants['vs_shallow'].min()
    max_s = df_variants['vs_shallow'].max()
    margin_s = max(abs(min_s), abs(max_s), 5.0) * 0.25
    ax2.set_xlim(min(min_s - margin_s, -1.0), max(max_s + margin_s, 1.0))

    for bar in bars2:
        w = bar.get_width()
        if w >= 0:
            ax2.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:+.1f} pp',
                     va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            ax2.text(w - 0.3, bar.get_y() + bar.get_height()/2, f'{w:+.1f} pp',
                     va='center', ha='right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    contrast_plot_path = os.path.join(output_dir, f'baseline_contrasts_{task_name.lower().replace(" ", "_")}.png')
    plt.savefig(contrast_plot_path, dpi=300)
    plt.close()
    print(f"Saved baseline contrast plot: {contrast_plot_path}")


def plot_training_histories(histories, task_name, output_dir='results'):
    """
    Plots training loss and test accuracy curves across epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Train Loss
    ax1 = axes[0]
    for model_name, hist in histories.items():
        if 'train_loss' in hist and len(hist['train_loss']) > 0:
            epochs = range(1, len(hist['train_loss']) + 1)
            ax1.plot(epochs, hist['train_loss'], label=model_name, linewidth=1.8)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title(f'{task_name} — Training Loss Dynamics', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8, loc='upper right')

    # Test Accuracy
    ax2 = axes[1]
    for model_name, hist in histories.items():
        if 'test_acc' in hist and len(hist['test_acc']) > 0:
            epochs = range(1, len(hist['test_acc']) + 1)
            ax2.plot(epochs, hist['test_acc'], label=model_name, linewidth=1.8)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{task_name} — Test Accuracy Progression', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    hist_path = os.path.join(output_dir, f'training_dynamics_{task_name.lower().replace(" ", "_")}.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved training dynamics plot: {hist_path}")


def replot_all_from_csv(results_dir='results'):
    """
    Utility function to regenerate all benchmark, baseline contrast,
    size study, and cross-task summary plots directly from existing CSV logs.
    """
    p1 = os.path.join(results_dir, "benchmark_level_1_-_5-arm_spiral.csv")
    p2 = os.path.join(results_dir, "benchmark_level_2_-_4x4_checkerboard.csv")
    p3 = os.path.join(results_dir, "benchmark_level_3_-_cifar-100.csv")
    pd_depth = os.path.join(results_dir, "study_original_network_size_depth.csv")
    pd_bank = os.path.join(results_dir, "study_weight_sharing_size_bank.csv")

    all_dict = {}

    if os.path.exists(p1):
        df1 = pd.read_csv(p1)
        plot_task_benchmark(df1, "Level 1 - 5-Arm Spiral", output_dir=results_dir)
        plot_baseline_contrasts(df1, "Level 1 - 5-Arm Spiral", output_dir=results_dir)
        all_dict["Level 1 - Spiral"] = df1

    if os.path.exists(p2):
        df2 = pd.read_csv(p2)
        plot_task_benchmark(df2, "Level 2 - 4x4 Checkerboard", output_dir=results_dir)
        plot_baseline_contrasts(df2, "Level 2 - 4x4 Checkerboard", output_dir=results_dir)
        all_dict["Level 2 - Checkerboard"] = df2

    if os.path.exists(p3):
        df3 = pd.read_csv(p3)
        plot_task_benchmark(df3, "Level 3 - CIFAR-100", output_dir=results_dir)
        plot_baseline_contrasts(df3, "Level 3 - CIFAR-100", output_dir=results_dir)
        all_dict["Level 3 - CIFAR-100"] = df3

    if len(all_dict) > 1:
        plot_cross_task_summary(all_dict, output_dir=results_dir)

    if os.path.exists(pd_depth):
        df_depth = pd.read_csv(pd_depth)
        plot_size_study_depth(df_depth, output_dir=results_dir)

    if os.path.exists(pd_bank):
        df_bank = pd.read_csv(pd_bank)
        plot_size_study_bank(df_bank, output_dir=results_dir)

    print("Replotting completed successfully.")


if __name__ == '__main__':
    replot_all_from_csv()
