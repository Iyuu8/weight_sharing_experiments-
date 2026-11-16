# Empirical Evaluation of Depth-Wise Weight Sharing and Learned Layer Routing

**Author:** AI Research Experiment  
**Status:** Completed Experiment & Benchmark Report  
**Framework:** PyTorch 2.6.0 on CUDA  

---

## Abstract

We investigate whether a deep neural network can be replaced by a compact architecture that reuses a small bank of layers across depth (weight sharing applied cyclically), preserving the accuracy of a full unshared deep network while drastically reducing total parameters. We further evaluate whether learned layer selection—implemented via internal state-conditioned routers or external input-conditioned controllers (soft mixture and hard discrete variants), as well as adaptive depth controllers—provides any empirical advantage over static cyclic reuse. Across three difficulty tiers (a 5-arm synthetic spiral, a $4 \times 4$ checkerboard, and CIFAR-100), results reveal a nuanced and domain-dependent picture. On the 2-D spiral task, cyclic weight sharing achieves parity or exceeds the unshared deep reference (up to 100.0% vs. 95.3%, a gain of +4.7 percentage points [pp]), with a single shared layer matching the parameter-matched shallow control while outperforming it by +13.9 pp (99.6% vs. 85.7%). However, this synthetic advantage is heavily driven by optimization instability in the unshared deep baseline, which completely collapses to random guess (20.0%) at depth $T=10$. On the $4 \times 4$ checkerboard, the unshared deep network is superior (95.4%), with the best shared model (`Shared-2-Cyclic`) achieving 94.8% ($-0.6$ pp) at $2.93\times$ compression. On CIFAR-100, the weight-sharing hypothesis fails completely: all weight-shared variants obtain 10.2% to 15.6% test accuracy, falling 17.8 to 23.1 pp below the deep reference (33.32%) and 9.0 to 14.4 pp below the parameter-matched shallow baseline (24.58%). Across all benchmarks, learned routers and controllers add parameters without delivering consistent gains over fixed cyclic scheduling. We conclude that while cyclic weight reuse confers inductive regularization that prevents gradient collapse in low-dimensional synthetic settings, it does not transfer to natural image classification under the evaluated configuration.

---

## 1. Research Question

Modern deep neural networks derive representational power from the composition of successive nonlinear transformations. However, stacking independent layers incurs a linear parameter cost $O(T)$ with depth $T$. This raises two central research questions:

1. **Parameter Reuse Hypothesis:** Can an effective depth of $T$ hidden layer applications be realized by iteratively applying a small bank of $n \ll T$ shared layers (cyclic weight sharing) such that task accuracy matches the unshared deep ceiling at a fraction of the parameter cost?
2. **Learned Routing Hypothesis:** Does equipping the network with a learned dynamic mechanism (an internal router conditioned on intermediate activations, an external controller predicting layer plans from input, or an adaptive halting controller) improve performance over deterministic, static cyclic reuse?

---

## 2. Models Compared

Let $T$ denote the number of hidden-layer applications (effective computational depth) and $n$ denote the number of unique layers maintained in the parameter bank.

| Model | Type | Bank Size ($n$) | Description |
| :--- | :--- | :---: | :--- |
| **`Deep-Unshared`** | Reference Baseline | $T$ | $T$ distinct hidden layers applied sequentially. Serves as the full-capacity unshared reference ("deep ceiling"). |
| **`Shallow`** | Control Baseline | $1$ | One hidden layer applied once ($T=1$). Strictly parameter-matched to `Shared-Single` and `Adaptive-Depth-Controller`. |
| **`Shared-Single`** | Static Weight-Sharing | $1$ | A single hidden layer applied recurrently for $T$ steps ($B_0 \to B_0 \to \dots \to B_0$). |
| **`Shared-2-Cyclic`** | Static Weight-Sharing | $2$ | Bank of $n=2$ distinct layers applied in alternating sequence: $B_0 \to B_1 \to B_0 \to B_1 \dots$ for $T$ steps. |
| **`Shared-3-Cyclic`** | Static Weight-Sharing | $3$ | Bank of $n=3$ distinct layers applied in rotating sequence: $B_0 \to B_1 \to B_2 \to B_0 \dots$ for $T$ steps. |
| **`Internal-Router-Soft`** | Dynamic Routing | $3$ | Bank of $n=3$ layers. At each step $t$, intermediate hidden state $h_t$ predicts routing weights; layer outputs are combined via a continuous softmax mixture. |
| **`Internal-Router-Hard`** | Dynamic Routing | $3$ | Same bank and routing mechanism as Soft, but uses discrete layer selection via Gumbel-Softmax ($k = \arg\max$). |
| **`External-Controller-Soft`** | Dynamic Planning | $3$ | Bank of $n=3$ layers. An auxiliary controller network inspects input $x$ globally and predicts a soft routing matrix $W \in \mathbb{R}^{T \times n}$ prior to layer execution. |
| **`External-Controller-Hard`** | Dynamic Planning | $3$ | Same global controller architecture, but enforces discrete single-layer selection at each step $t$. |
| **`Adaptive-Depth-Controller`** | Dynamic Halting | $1$ | Single shared layer ($n=1$) coupled with a halting controller that predicts an early termination / ponder distribution over steps $t \in \{1, \dots, T\}$. |

---

## 3. Parameter Accounting & Inferred Architectures

### 3.1 Mathematical Parameter Formulas (2-D Synthetic Tasks)

For the 2-D input synthetic benchmarks ($d_{\text{in}} = 2$), the parameter counts across all variants match a dense feed-forward architecture of width $H = 64$. 

For an $n$-block shared bank on a $C$-class problem:
$$P_{\text{shared}}(n) = \underbrace{(d_{\text{in}} \cdot H + H)}_{\text{Input Projection}} + n \cdot \underbrace{(H \cdot H + H)}_{\text{Shared Bank Layer}} + \underbrace{(H \cdot C + C)}_{\text{Output Head}}$$

Substituting $d_{\text{in}} = 2$ and $H = 64$:
$$P_{\text{shared}}(n) = (2 \cdot 64 + 64) + n \cdot (64 \cdot 64 + 64) + (64 \cdot C + C) = 192 + 4160\,n + (65\,C)$$

For the unshared deep reference with $T$ distinct layers:
$$P_{\text{deep}}(T) = 192 + 4160\,T + (65\,C)$$

The parameter compression ratio $\rho$ relative to the unshared deep reference is defined as:
$$\rho = \frac{P_{\text{deep}}(T)}{P_{\text{model}}}$$

### 3.2 Exact Counts by Task

1. **Level 1 (5-Arm Spiral, $C = 5$ classes):**
   - Output layer: $64 \cdot 5 + 5 = 325$ parameters.
   - `Shallow` ($n=1, T=1$): $192 + 4160(1) + 325 = 4,677$ parameters.
   - `Shared-Single` ($n=1, T=6$): $192 + 4160(1) + 325 = 4,677$ parameters ($\rho = 5.45\times$).
   - `Shared-2-Cyclic` ($n=2, T=6$): $192 + 4160(2) + 325 = 8,837$ parameters ($\rho = 2.88\times$).
   - `Shared-3-Cyclic` ($n=3, T=6$): $192 + 4160(3) + 325 = 12,997$ parameters ($\rho = 1.96\times$).
   - `Deep-Unshared` ($T=6$): $192 + 4160(6) + 325 = 25,477$ parameters ($\rho = 1.00\times$).

2. **Level 2 ($4 \times 4$ Checkerboard, $C = 2$ classes):**
   - Output layer: $64 \cdot 2 + 2 = 130$ parameters.
   - `Shallow` ($n=1, T=1$): $192 + 4160(1) + 130 = 4,482$ parameters.
   - `Shared-Single` ($n=1, T=6$): $4,482$ parameters ($\rho = 5.64\times$).
   - `Shared-2-Cyclic` ($n=2, T=6$): $8,642$ parameters ($\rho = 2.93\times$).
   - `Shared-3-Cyclic` ($n=3, T=6$): $12,802$ parameters ($\rho = 1.97\times$).
   - `Deep-Unshared` ($T=6$): $25,282$ parameters ($\rho = 1.00\times$).

3. **Controller and Router Overhead:**
   - `Internal-Router` adds a routing linear layer ($H \to n$), adding $+195$ parameters on Spiral and Checkerboard, and $+147$ on CIFAR-100.
   - `External-Controller` adds a dedicated controller network, adding $+354$ parameters on Spiral and Checkerboard, and $+13,522$ on CIFAR-100.
   - `Adaptive-Depth-Controller` adds a halting predictor parameterized as $48 + 17T$ parameters over `Shared-Single` ($82, 116, 150, 184, 218$ extra parameters for $T = 2, 4, 6, 8, 10$ respectively).

4. **Architectural Equivalence at Boundary Points:**
   - At $T = 2$, `Shared-2-Cyclic` has $n = 2$ distinct layers applied once each, making it **architecturally identical to `Deep-Unshared` at $T = 2$** (both $8,837$ parameters).
   - In Study 2 at target depth $T = 6$, setting bank size $n = 6$ yields an unshared network with $T = 6$ independent layers ($25,477$ parameters), identical to `Deep-Unshared`.
   - Any empirical performance differences between these identical pairs reflect run-to-run initialization and optimization noise.

5. **Level 3 (CIFAR-100):**
   - Each hidden block comprises $41,664$ parameters (a 2-layer convolutional block with $C=48$ channels, $3 \times 3$ kernels, and batch normalization).
   - `Deep-Unshared` ($T=6$): $256,276$ parameters.
   - `Shallow` ($T=1$): $47,956$ parameters.
   - `Shared-Single` ($n=1, T=6$): $47,956$ parameters ($\rho = 5.34\times$).
   - `Shared-3-Cyclic` ($n=3, T=6$): $131,284$ parameters ($\rho = 1.95\times$).

---

## 4. Main Benchmark Results ($T = 6$)

Table 1 summarizes test classification accuracy (%) across all 10 architectures on the three benchmark tasks at target depth $T = 6$. All differences are reported in **percentage points (pp)** relative to the reference models.

**Table 1: Main Benchmark Test Accuracy (%) across Tasks at $T = 6$**

| Model | Params (L1 / L2 / L3) | L1 Spiral (5-class) | L2 Checkerboard (2-class) | L3 CIFAR-100 (100-class) |
| :--- | :---: | :---: | :---: | :---: |
| **`Deep-Unshared`** *(Reference)* | 25,477 / 25,282 / 256,276 | 95.30% | **95.40%** | **33.32%** |
| **`Shallow`** *(Control)* | 4,677 / 4,482 / 47,956 | 85.70% | 91.70% | 24.58% |
| **`Shared-Single`** | 4,677 / 4,482 / 47,956 | 99.60% | 93.10% | 11.63% |
| **`Shared-2-Cyclic`** | 8,837 / 8,642 / 89,620 | 99.90% | 94.80% | 11.23% |
| **`Shared-3-Cyclic`** | 12,997 / 12,802 / 131,284 | **100.00%** | 94.30% | 15.01% |
| **`Internal-Router-Soft`** | 13,192 / 12,997 / 131,431 | 99.70% | 94.70% | 14.84% |
| **`Internal-Router-Hard`** | 13,192 / 12,997 / 131,431 | 99.50% | 94.40% | 15.42% |
| **`External-Controller-Soft`**| 13,351 / 13,156 / 144,806 | 97.20% | 94.10% | 15.56% |
| **`External-Controller-Hard`**| 13,351 / 13,156 / 144,806 | 99.70% | 92.80% | 13.44% |
| **`Adaptive-Depth-Controller`**| 4,827 / 4,632 / 60,698 | 99.60% | 93.30% | 10.22% |

### 4.1 Detailed Findings per Benchmark Level

#### Level 1: 5-Arm Spiral (Geometric Depth Separation)
- **Weight Sharing Outperforms the Reference:** Every weight-shared variant (97.20% to 100.00%) outperforms `Deep-Unshared` (95.30%) by $+1.90$ to $+4.70$ pp, and outperforms `Shallow` (85.70%) by $+11.50$ to $+14.30$ pp.
- **Single-Layer Efficiency:** `Shared-Single` achieves 99.60% using the exact same parameter count ($4,677$) as `Shallow` (85.70%), an improvement of **$+13.90$ pp**.
- **Reference Underperformance:** The "deep ceiling" is not an empirical upper bound on this task; the unshared deep network is actively harder to optimize and underperforms the shared variants.
- **Outliers:** The only notably weaker weight-shared variant is `External-Controller-Soft` (97.20%), which lags `Shared-3-Cyclic` (100.00%) by $-2.80$ pp.

#### Level 2: $4 \times 4$ Checkerboard (Combinatorial Depth Separation)
- **Deep Reference is Best:** `Deep-Unshared` achieves the highest score (95.40%).
- **Competitive Cyclic Sharing:** `Shared-2-Cyclic` comes closest at 94.80% ($-0.60$ pp vs. Reference) at $2.93\times$ compression, followed by `Internal-Router-Soft` at 94.70% ($-0.70$ pp).
- **Narrow Dynamic Range:** The gap between `Shallow` (91.70%) and `Deep-Unshared` (95.40%) is only $3.70$ pp. `Shared-Single` (93.10%) and `Adaptive-Depth-Controller` (93.30%) surpass `Shallow` by only $+1.40$ pp and $+1.60$ pp, leaving insufficient dynamic range to resolve structural superiority beyond run-to-run noise.

#### Level 3: CIFAR-100 (The Method Fails)
- **Severe Degradation:** `Deep-Unshared` reaches 33.32% and `Shallow` reaches 24.58%. All weight-shared models score between 10.22% and 15.56%.
- **Negative Gain vs. Shallow Control:** Every weight-shared model falls **17.76 to 23.10 pp below `Deep-Unshared`** and **9.02 to 14.36 pp below the parameter-matched `Shallow` network**.
- **Repeated Application is Harmful:** `Shared-Single` and `Shallow` possess the exact same parameter footprint ($47,956$), yet `Shared-Single` scores 11.63% while `Shallow` scores 24.58% ($-12.95$ pp). Applying the same convolutional transformation repeatedly degrades image representations.
- **Worst Performer:** `Adaptive-Depth-Controller` performs worst at 10.22% ($-14.36$ pp vs. Shallow).
- **Learning Sanity:** Chance performance on 100 balanced classes is 1.00%. The models learn features, but sub-optimally.

### 4.2 Cross-Variant Observations

1. **No Benefit from Routers and Controllers:** Learned internal routers and external controllers add parameter overhead without consistent accuracy gains over deterministic cyclic reuse (`Shared-3-Cyclic`):
   - Spiral: `Shared-3-Cyclic` (100.00%) beats all router and controller variants (97.20% to 99.70%).
   - Checkerboard: `Shared-3-Cyclic` scores 94.30%; routers score 94.40% to 94.70% (+0.1 to +0.4 pp, within run-to-run noise).
   - CIFAR-100: `Shared-3-Cyclic` scores 15.01%; controllers score 13.44% to 15.56% (mixed).
2. **Soft vs. Hard Routing is Inconclusive:**
   - Spiral: `External-Controller-Hard` (99.70%) strongly beats `Soft` (97.20%).
   - Checkerboard: `External-Controller-Soft` (94.10%) beats `Hard` (92.80%).
   - CIFAR-100: `Soft` beats `Hard` for External Controller (15.56% vs. 13.44%), but `Hard` beats `Soft` for Internal Router (15.42% vs. 14.84%).
3. **Adaptive-Depth Adds No Value Over Shared-Single:**
   - Level 1: `Adaptive-Depth` scores 99.60% vs. `Shared-Single` 99.60% ($0.00$ pp).
   - Level 2: `Adaptive-Depth` scores 93.30% vs. `Shared-Single` 93.10% ($+0.20$ pp).
   - Level 3: `Adaptive-Depth` scores 10.22% vs. `Shared-Single` 11.63% ($-1.41$ pp).
   - Halting distribution prediction provides no measurable benefit over fixed-depth recurrent iteration.

---

## 5. Ablation Studies (Spiral Task)

### 5.1 Study 1: Accuracy vs. Original Network Depth ($T$)

To evaluate depth scaling, models were trained across target depths $T \in \{2, 4, 6, 8, 10\}$ on the 5-arm spiral dataset.

**Table 2: Test Accuracy (%) and Parameter Count Across Target Depths ($T$)**

| Target Depth ($T$) | `Deep-Unshared` | `Shallow` (Control) | `Shared-Single` | `Shared-2-Cyclic` | `Adaptive-Depth` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$T = 2$** | 99.00% ($8,837$) | 91.80% ($4,677$) | 98.20% ($4,677$) | 98.40% ($8,837$) | 97.70% ($4,759$) |
| **$T = 4$** | 100.00% ($17,157$) | 87.40% ($4,677$) | 99.90% ($4,677$) | 99.70% ($8,837$) | 99.70% ($4,793$) |
| **$T = 6$** | 95.80% ($25,477$) | 85.70% ($4,677$) | 99.80% ($4,677$) | 99.90% ($8,837$) | 92.30% ($4,827$) |
| **$T = 8$** | 91.50% ($33,797$) | 87.70% ($4,677$) | 98.70% ($4,677$) | 98.60% ($8,837$) | 99.10% ($4,861$) |
| **$T = 10$** | **20.00%** ($42,117$) | 91.40% ($4,677$) | **95.20%** ($4,677$) | **96.60%** ($8,837$) | **97.90%** ($4,895$) |

#### Key Findings from Study 1:
- **Catastrophic Collapse of Unshared Deep Network:** `Deep-Unshared` peaks at $T = 4$ (100.00%), degrades past $T = 6$ (95.80% $\to$ 91.50%), and **collapses to exactly 20.00% at $T = 10$**. For a 5-class balanced problem, 20.00% corresponds precisely to random chance ($\frac{1}{5}$). This represents a complete optimization failure (vanishing/exploding gradients during backpropagation through 10 unnormalized dense layers), rather than overfitting.
- **Robustness of Weight-Shared Models:** All weight-shared models maintain robust convergence across depth ($95.20\%$ to $97.90\%$ at $T=10$). Parameter reuse acts as a structural stabilizer, enforcing consistent state representations and mitigating gradient pathology.
- **Parameter Divergence:** At $T = 10$, `Deep-Unshared` requires $42,117$ parameters, whereas `Shared-Single` maintains $4,677$ parameters ($9.0\times$ compression) while remaining functional.

### 5.2 Study 2: Accuracy vs. Bank Size ($n$) at Fixed Depth ($T = 6$)

To determine the minimum bank size needed to match unshared capacity, bank sizes $n \in \{1, 2, 3, 4, 6\}$ were evaluated at fixed depth $T = 6$ on the spiral dataset. (Note: $n = 5$ was not evaluated; $n = 6$ is architecturally identical to the unshared deep network).

**Table 3: Accuracy vs. Bank Size ($n$) for `Shared-N-Cyclic` ($T = 6$)**

| Bank Size ($n$) | Parameters | Compression ($\rho$) | Test Acc (%) | $\Delta$ vs. Deep Ref (pp) | $\Delta$ vs. Shallow (pp) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$n = 1$** | 4,677 | $5.45\times$ | 97.40% | $-0.70$ pp | $+10.20$ pp |
| **$n = 2$** | **8,837** | **$2.88\times$** | **99.60%** | **$+1.50$ pp** | **$+12.40$ pp** |
| **$n = 3$** | 12,997 | $1.96\times$ | 99.70% | $+1.60$ pp | $+12.50$ pp |
| **$n = 4$** | 17,157 | $1.48\times$ | 99.90% | $+1.80$ pp | $+12.70$ pp |
| **$n = 6$** *(Unshared)* | 25,477 | $1.00\times$ | 98.10% | $0.00$ pp | $+10.90$ pp |

*(Reference in this run: Deep Reference 98.10%, Shallow Control 87.20%).*

#### Key Findings from Study 2:
- **The $n = 2$ "Elbow Point":** The transition from $n = 1$ to $n = 2$ produces the steepest accuracy gain ($97.40\% \to 99.60\%$, $+2.20$ pp), reaching within $0.3$ pp of the maximum observed score while preserving nearly $3\times$ compression.
- **Plateau Beyond $n = 2$:** Increasing bank size to $n = 3$ ($99.70\%$) and $n = 4$ ($99.90\%$) yields diminishing returns ($+0.1$ to $+0.2$ pp).
- **Noise at $n = 6$:** The $n = 6$ row is architecturally identical to `Deep-Unshared` ($T=6$), yet scores 98.10% compared to 95.30% in the benchmark and 95.80% in Study 1, demonstrating substantial optimization variance.

---

## 6. Critical Analysis & Methodological Caveats

Rigorous analysis requires addressing six substantial methodological caveats:

### 6.1 Substantial Run-to-Run Noise Without Multiple Random Seeds
All reported figures originate from single experimental runs without cross-seed replication. Strong internal evidence demonstrates that run-to-run variance is on the order of 2 to 7 pp:
1. `Shallow` has identical architecture ($4,677$ params) in every row of Study 1, yet scores vary from **85.70% to 91.80% across rows (a 6.10 pp spread)**. It scores 85.70% in Benchmark Level 1 and 87.20% in Study 2.
2. `Deep-Unshared` at $T = 6$ ($25,477$ params) scores **95.30%** in Table 1, **95.80%** in Study 1, and **98.10%** in Study 2 (a **2.80 pp spread** across identical configurations).
3. `Adaptive-Depth-Controller` at $T = 6$ ($4,827$ params) scores **99.60%** in Table 1 but **92.30%** in Study 1 (a **7.30 pp spread**).
4. `Shared-2-Cyclic` and `Deep-Unshared` at $T = 2$ are architecturally identical ($8,837$ params), yet score **98.40% vs. 99.00%** ($0.60$ pp difference).

> **Conclusion on Significance:** Any observed accuracy differences below approximately **$2.0$ to $3.0$ pp** (e.g., Soft vs. Hard routing differences, or cyclic vs. router comparisons) cannot be considered statistically significant.

### 6.2 The "Deep Ceiling" is an Unreliable Baseline
In prior files, `Deep-Unshared` was termed the "deep ceiling." This terminology is scientifically misleading:
- In Level 1, `Deep-Unshared` underperforms every weight-shared variant.
- In Study 1, `Deep-Unshared` degrades past $T=4$ and collapses at $T=10$.
- It exhibits up to $2.80$ pp variance across runs.
It is an *unshared deep reference model*, not an empirical performance ceiling.

### 6.3 Asymmetry in the "Parameter-Matched Shallow Control"
The `Shallow` network ($4,677$ params on Spiral; $4,482$ on Checkerboard; $47,956$ on CIFAR-100) is parameter-matched **only to `Shared-Single` and `Adaptive-Depth-Controller`**.
- `Shared-2-Cyclic` has $1.89\times$ more parameters than `Shallow`.
- `Shared-3-Cyclic`, `Internal-Router`, and `External-Controller` have $2.74\times$ to $2.93\times$ more parameters than `Shallow`.
Therefore, stating that `Shared-2-Cyclic` or `Shared-3-Cyclic` exhibits a "$+14$ pp advantage over the parameter-matched shallow baseline" is conflating depth-reuse with raw parameter capacity. A fair evaluation requires width-matched shallow baselines at $\sim 8.8\text{k}$ and $\sim 13.0\text{k}$ parameters.

### 6.4 Single Dataset Family for Positive Findings
The positive results for weight sharing are confined exclusively to low-dimensional synthetic 2-D coordinates (primarily the 5-arm spiral). On the combinatorial checkerboard, gains are marginal ($+1.40$ pp over shallow). On natural images (CIFAR-100), the approach fails completely. The hypothesis that weight sharing generalizes across domains is refuted by the CIFAR-100 data.

### 6.5 Deep Baseline Optimization Failure vs. Intrinsic Shared Superiority
The superiority of weight sharing on the spiral task is primarily attributable to the **fragility of the unshared deep baseline**, rather than unique representational power of weight reuse. Unshared dense feedforward networks without normalization or residual connections are notoriously difficult to optimize at depths $T \ge 6$. A modernized deep baseline (equipped with LayerNorm/BatchNorm, residual shortcuts, and learning rate warmup) would likely resolve the $T=10$ collapse, potentially eliminating the margin credited to weight sharing.

### 6.6 Unresolved Failure Mode on CIFAR-100
Why does cyclic weight sharing fail catastrophically on CIFAR-100? We identify four candidate, untested hypotheses:
1. **Receptive Field and Semantic Hierarchy Mismatch:** Visual hierarchies require qualitative transitions from local edge detectors (small receptive fields, low abstraction) to global semantic part detectors. Forcing depth steps to share identical convolutional weights enforces an identical filter scale, which may be fundamentally incompatible with multi-scale visual abstraction.
2. **Batch Normalization Distribution Shift:** When a convolutional layer with running mean and variance statistics is reused across steps $t \in \{1, \dots, T\}$, the feature distribution changes dramatically with depth. Running statistics calculated across all depth steps or fixed to a single layer become mismatched with step-specific activations.
3. **Severe Under-Training:** Vision models trained for 10–12 epochs with complex recurrent dependencies may simply be under-converged relative to feed-forward baselines.
4. **Capacity Bottleneck:** A single block ($47\text{k}$ params) may lack sufficient capacity to learn a general transformation valid across multiple depth steps.

---

## 7. Unspecified Details & Reporting Transparency

To maintain scientific integrity and avoid inventing unrecorded parameters, we explicitly document the experimental details that were not formally specified or standardized across runs:

- **Random Seeds & Replications:** All experiments represent single-seed runs ($N=1$). Exact PRNG seeds for model weight initialization and data shuffling are unrecorded across comparative runs.
- **Optimization Hyperparameters:** While synthetic models trained for 300 epochs and CIFAR-100 models trained for 10 epochs using Adam, the exact learning rate schedule, weight decay, momentum parameters, gradient clipping thresholds, and batch sizes across all individual model runs were not logged in a centralized specification.
- **Architectural Micro-Design:** Activation functions (ReLU assumed), presence/absence of residual skip connections and input injection ($h_0$) across synthetic baselines, and exact normalizations were inferred from source code rather than an explicit protocol.
- **Routing Mechanisms:** The mathematical formulation of the router and controller gating distributions, temperature schedules ($\tau$) for Gumbel-Softmax, auxiliary load-balancing losses, and gradient estimators for discrete routing choices were not formally ablated.
- **CIFAR-100 Vision Block:** The exact channel topology, stride progression, and data augmentation pipeline were inferred from implementation files.

---

## 8. Figures and Artifacts Index

All generated figures, tabular CSV metrics, and benchmark logs reside in `results/`:

1. **Benchmark Performance & Pareto Frontiers:**
   - Level 1 Spiral: `results/benchmark_level_1_-_5-arm_spiral.png`
   - Level 2 Checkerboard: `results/benchmark_level_2_-_4x4_checkerboard.png`
   - Level 3 CIFAR-100: `results/benchmark_level_3_-_cifar-100.png`
2. **Baseline Contrasts (Deltas in Percentage Points):**
   - Level 1 Spiral: `results/baseline_contrasts_level_1_-_5-arm_spiral.png`
   - Level 2 Checkerboard: `results/baseline_contrasts_level_2_-_4x4_checkerboard.png`
   - Level 3 CIFAR-100: `results/baseline_contrasts_level_3_-_cifar-100.png`
3. **Cross-Task Benchmark Summary:**
   - Multi-panel comparison ordered Level 1 $\to$ Level 2 $\to$ Level 3: `results/cross_task_benchmarks_summary.png`
4. **Size Scaling Studies:**
   - Original Depth Scaling ($T \in [2, 10]$): `results/study_original_network_size_depth.png`
   - Bank Size Scaling ($n \in [1, 6]$): `results/study_weight_sharing_size_bank.png`
5. **Training Dynamics:**
   - Convergence loss and accuracy curves: `results/training_dynamics_level_{1,2,3}_*.png`

---

## 9. Conclusion

1. **Synthetic Low-Dimensional Tasks:** Cyclic weight sharing with a small bank ($n = 2$) matches or exceeds the accuracy of an unshared deep reference network on low-dimensional geometric problems at $\sim 2.9\times$ parameter compression. Reusing layers provides strong inductive regularization that prevents optimization collapse at extreme depths ($T = 10$).
2. **Learned Routing:** Dynamic routers (soft mixture or hard Gumbel-Softmax) and global external controllers provide no consistent accuracy benefit over static, deterministic cyclic reuse, while adding parameters and computational complexity.
3. **Domain Transfer Failure:** Weight sharing fails completely on natural image classification (CIFAR-100), where every weight-shared architecture falls significantly below both the unshared deep reference ($-18$ to $-23$ pp) and the parameter-matched shallow baseline ($-9$ to $-14$ pp). Repeated application of a single convolutional block is actively deleterious.
4. **Methodological Limits:** Current conclusions are constrained by single-seed runs, substantial optimization noise, the absence of wider shallow controls, and an unoptimized deep baseline.

---

## 10. Recommended Next Experiments

To transition these exploratory findings into publication-ready research, future work should prioritize:

1. **Multi-Seed Replication:** Execute all benchmark configurations across at least 5 distinct random seeds, reporting metrics as $\text{mean} \pm \text{standard deviation}$ with two-tailed Welch's $t$-tests.
2. **Width-Matched Shallow Controls:** Introduce wider shallow baselines parameter-matched to `Shared-2-Cyclic` ($\sim 8.8\text{k}$ params) and `Shared-3-Cyclic` ($\sim 13.0\text{k}$ params) to rigorously isolate depth-via-reuse from raw capacity.
3. **Modernized Deep Baseline:** Evaluate deep baselines with LayerNorm/BatchNorm, residual skip connections, and cosine learning rate schedules to verify whether weight sharing retains any advantage when the unshared baseline is properly stabilized.
4. **Diagnostic Analysis of the Vision Failure:**
   - Implement step-specific Batch Normalization layers (shared conv weights, but private running statistics per depth step $t$).
   - Train for extended schedules (50–100 epochs) with cosine decay.
   - Conduct per-step feature alignment and representation similarity analysis (e.g., Centered Kernel Alignment - CKA) across depth steps.
5. **Intermediate Benchmark Suite:** Evaluate tasks of intermediate complexity between 2-D coordinates and CIFAR-100 (e.g., MNIST, Fashion-MNIST, CIFAR-10, SVHN) to identify the precise dimensionality and spatial threshold where cyclic weight sharing breaks down.
