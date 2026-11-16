"""
Experiment Execution Engine for Weight-Sharing Research.
Optimized for high-speed CUDA execution:
- Full-batch tensor computation for synthetic 2D benchmarks (300 epochs in ~0.3s)
- Fast GPU-resident batching with AMP for CIFAR-100 image benchmark (10 epochs in ~30s per model)
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

import models
import datasets
import plot_results

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 1. Training & Evaluation Engine
# ============================================================================

def train_and_eval_model(name, model, train_data, test_data=None, epochs=300, lr=1e-3,
                         weight_decay=1e-4, device='cuda', is_conv=False, verbose=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = (device == 'cuda' and is_conv)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    params = models.count_parameters(model)
    history = {'train_loss': [], 'test_acc': []}
    best_acc = 0.0

    if verbose:
        print(f"\n--> Training [{name}] (Params: {params:,} | Epochs: {epochs} | Device: {device})", flush=True)

    t_start = time.time()

    # Path 1: FastCIFAR100 GPU-resident batcher
    if isinstance(train_data, datasets.FastCIFAR100):
        cifar = train_data
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for inputs, targets in cifar.get_train_batches():
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = _compute_loss(model, outputs, targets, criterion, device)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = _compute_loss(model, outputs, targets, criterion, device)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(1, n_batches)

            # Fast GPU Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in cifar.get_test_batches():
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)

                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    _, preds = logits.max(1)
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()

            test_acc = 100.0 * correct / max(1, total)
            if test_acc > best_acc:
                best_acc = test_acc

            history['train_loss'].append(avg_loss)
            history['test_acc'].append(test_acc)

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                print(f"    Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)", flush=True)

    # Path 2: Full-batch tensor data (for synthetic 2D datasets)
    elif isinstance(train_data, tuple):
        X_train, y_train = train_data
        X_test, y_test = test_data
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = _compute_loss(model, outputs, y_train, criterion, device)
            loss.backward()
            optimizer.step()

            avg_loss = loss.item()

            # Eval
            model.eval()
            with torch.no_grad():
                test_out = model(X_test)
                if isinstance(test_out, tuple):
                    test_logits = test_out[0]
                else:
                    test_logits = test_out
                preds = test_logits.argmax(dim=-1)
                test_acc = 100.0 * (preds == y_test).float().mean().item()

            if test_acc > best_acc:
                best_acc = test_acc

            history['train_loss'].append(avg_loss)
            history['test_acc'].append(test_acc)

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                print(f"    Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)", flush=True)

    elapsed = time.time() - t_start
    if verbose:
        print(f"    Completed in {elapsed:.2f}s | Best Test Acc: {best_acc:.2f}%", flush=True)

    return best_acc, history, params


def _compute_loss(model, outputs, targets, criterion, device):
    """Handles standard loss as well as load-balancing & ponder auxiliary losses."""
    if isinstance(outputs, tuple):
        logits = outputs[0]
        if isinstance(model, (models.MLPExternalController, models.ConvExternalController)):
            plan_logits = outputs[1]
            task_loss = criterion(logits, targets)
            bal_loss = models.compute_load_balance_loss(plan_logits, len(model.bank))
            return task_loss + 0.5 * bal_loss
        elif isinstance(model, (models.MLPAdaptiveDepthController, models.ConvAdaptiveDepthController)):
            halt_at = outputs[1]
            task_loss = criterion(logits, targets)
            steps = (halt_at * torch.arange(1, halt_at.shape[1] + 1, device=device)).sum(1).mean()
            return task_loss + 0.01 * steps
        else:
            return criterion(logits, targets)
    else:
        return criterion(outputs, targets)


# ============================================================================
# 2. Benchmark Suite Runner
# ============================================================================

def run_task_benchmark(task_name, model_dict, train_data, test_data=None, epochs=300,
                       device='cuda', is_conv=False, output_dir=RESULTS_DIR):
    print(f"\n{'='*70}\nSTARTING BENCHMARK: {task_name}\n{'='*70}", flush=True)
    results = []
    histories = {}

    for name, model in model_dict.items():
        acc, hist, params = train_and_eval_model(
            name=name,
            model=model,
            train_data=train_data,
            test_data=test_data,
            epochs=epochs,
            device=device,
            is_conv=is_conv,
            verbose=True
        )
        histories[name] = hist
        results.append({
            'Model': name,
            'Params': params,
            'Test Acc (%)': acc
        })

    df = pd.DataFrame(results)

    # Calculate baseline references
    ceiling_row = df[df['Model'] == 'Deep-Unshared']
    ceiling_params = ceiling_row['Params'].values[0] if len(ceiling_row) > 0 else df['Params'].max()
    ceiling_acc = ceiling_row['Test Acc (%)'].values[0] if len(ceiling_row) > 0 else df['Test Acc (%)'].max()

    shallow_row = df[df['Model'] == 'Shallow']
    shallow_acc = shallow_row['Test Acc (%)'].values[0] if len(shallow_row) > 0 else 0.0

    df['Compression Ratio'] = ceiling_params / df['Params']
    df['Compression'] = df['Compression Ratio'].apply(lambda x: f"{x:.2f}x")
    df['vs Deep Reference (pp)'] = (df['Test Acc (%)'] - ceiling_acc).apply(lambda x: f"{x:+.2f} pp")
    df['vs Shallow Control (pp)'] = (df['Test Acc (%)'] - shallow_acc).apply(lambda x: f"{x:+.2f} pp")

    # Save results to CSV and Markdown
    csv_path = os.path.join(output_dir, f'benchmark_{task_name.lower().replace(" ", "_")}.csv')
    md_path = os.path.join(output_dir, f'benchmark_{task_name.lower().replace(" ", "_")}.md')
    df.to_csv(csv_path, index=False)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Results — {task_name}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"\n--- {task_name} Results ---", flush=True)
    print(df.to_markdown(index=False), flush=True)

    # Generate dedicated plots
    plot_results.plot_task_benchmark(df, task_name, output_dir=output_dir)
    plot_results.plot_baseline_contrasts(df, task_name, output_dir=output_dir)
    plot_results.plot_training_histories(histories, task_name, output_dir=output_dir)

    return df, histories


# ============================================================================
# 3. Size Study 1: Accuracy vs Original Network Size (Target Depth T)
# ============================================================================

def run_size_study_original_depth(depths=(2, 4, 6, 8, 10), epochs=300, device='cuda', output_dir=RESULTS_DIR):
    print(f"\n{'='*70}\nSTUDY 1: Accuracy vs Original Network Size (Target Depth T)\n{'='*70}", flush=True)
    train_data, test_data, in_dim, out_dim = datasets.get_spiral_data(device=device)

    records = []
    for depth in depths:
        print(f"\n>>> Evaluating Target Depth T = {depth} <<<", flush=True)
        candidates = {
            'Deep-Unshared': models.MLPDeepUnshared(in_dim, out_dim, H=64, depth=depth),
            'Shallow': models.MLPShallow(in_dim, out_dim, H=64),
            'Shared-Single': models.MLPSharedSingle(in_dim, out_dim, H=64, depth=depth, inject_input=True),
            'Shared-2-Cyclic': models.MLPSharedCyclic(in_dim, out_dim, H=64, n_layers=2, depth=depth, inject_input=True),
            'Adaptive-Depth': models.MLPAdaptiveDepthController(in_dim, out_dim, H=64, max_depth=depth),
        }

        for model_name, model in candidates.items():
            acc, _, params = train_and_eval_model(
                name=f"{model_name}-T{depth}",
                model=model,
                train_data=train_data,
                test_data=test_data,
                epochs=epochs,
                device=device,
                is_conv=False,
                verbose=False
            )
            records.append({
                'Target Depth (T)': depth,
                'Model': model_name,
                'Params': params,
                'Test Acc (%)': acc
            })
            print(f"    T={depth:2d} | {model_name:<18} | Params: {params:6,d} | Acc: {acc:.2f}%", flush=True)

    df_depth = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'study_original_network_size_depth.csv')
    df_depth.to_csv(csv_path, index=False)
    
    with open(os.path.join(output_dir, 'study_original_network_size_depth.md'), 'w', encoding='utf-8') as f:
        f.write("# Study 1: Accuracy & Parameter Scaling vs Original Network Depth (T)\n\n")
        f.write(df_depth.to_markdown(index=False))
        f.write("\n")

    plot_results.plot_size_study_depth(df_depth, output_dir=output_dir)
    return df_depth


# ============================================================================
# 4. Size Study 2: Accuracy vs Weight Sharing Network Size (Bank Size n)
# ============================================================================

def run_size_study_weight_sharing_bank(bank_sizes=(1, 2, 3, 4, 6), target_depth=6, epochs=300, device='cuda', output_dir=RESULTS_DIR):
    print(f"\n{'='*70}\nSTUDY 2: Accuracy vs Weight-Sharing Network Size (Bank Size n, Target Depth T={target_depth})\n{'='*70}", flush=True)
    train_data, test_data, in_dim, out_dim = datasets.get_spiral_data(device=device)

    # Get ceiling and shallow references
    ceiling_model = models.MLPDeepUnshared(in_dim, out_dim, H=64, depth=target_depth)
    ceiling_acc, _, ceiling_params = train_and_eval_model(
        'Deep-Unshared-Ceiling', ceiling_model, train_data, test_data, epochs=epochs, device=device, verbose=False
    )
    shallow_model = models.MLPShallow(in_dim, out_dim, H=64)
    shallow_acc, _, shallow_params = train_and_eval_model(
        'Shallow-Control', shallow_model, train_data, test_data, epochs=epochs, device=device, verbose=False
    )

    records = []
    for n in bank_sizes:
        if n == target_depth:
            acc = ceiling_acc
            params = ceiling_params
        elif n == 1:
            m = models.MLPSharedSingle(in_dim, out_dim, H=64, depth=target_depth, inject_input=True)
            acc, _, params = train_and_eval_model(f"Shared-Bank-n{n}", m, train_data, test_data, epochs=epochs, device=device, verbose=False)
        else:
            m = models.MLPSharedCyclic(in_dim, out_dim, H=64, n_layers=n, depth=target_depth, inject_input=True)
            acc, _, params = train_and_eval_model(f"Shared-Bank-n{n}", m, train_data, test_data, epochs=epochs, device=device, verbose=False)

        comp_ratio = ceiling_params / params
        records.append({
            'Bank Size (n)': n,
            'Params': params,
            'Compression Ratio': comp_ratio,
            'Test Acc (%)': acc,
            'Ceiling Acc': ceiling_acc,
            'Shallow Acc': shallow_acc,
            'Delta vs Ceiling': acc - ceiling_acc,
            'Delta vs Shallow': acc - shallow_acc
        })
        print(f"    Bank Size n={n:2d} | Params: {params:6,d} ({comp_ratio:.2f}x compression) | Acc: {acc:.2f}% (vs Ceiling: {acc - ceiling_acc:+.2f}%)", flush=True)

    df_bank = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'study_weight_sharing_size_bank.csv')
    df_bank.to_csv(csv_path, index=False)

    with open(os.path.join(output_dir, 'study_weight_sharing_size_bank.md'), 'w', encoding='utf-8') as f:
        f.write("# Study 2: Accuracy vs Weight-Sharing Bank Size (n)\n\n")
        f.write(df_bank.to_markdown(index=False))
        f.write("\n")

    plot_results.plot_size_study_bank(df_bank, output_dir=output_dir)
    return df_bank
