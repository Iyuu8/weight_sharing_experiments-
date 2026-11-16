"""
Weight-Reuse / Layer-Sharing Network Architectures
Implements both MLP (for 2D synthetic depth benchmarks) and Conv/ResNet
(for vision benchmarks such as CIFAR-100) variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. MLP Architectures (for 2D Synthetic Depth Separation Benchmarks)
# ============================================================================

class MLPShallow(nn.Module):
    """
    Shallow baseline: A single hidden layer applied once.
    Parameter count roughly matches SharedSingle — serves as the
    control to verify whether depth-via-reuse is superior to standard shallow capacity.
    """
    def __init__(self, in_dim=2, out_dim=2, H=64):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.layer = nn.Linear(H, H)
        self.out = nn.Linear(H, out_dim)

    def forward(self, x):
        h0 = F.relu(self.inp(x))
        h = F.relu(self.layer(h0))
        return self.out(h)


class MLPDeepUnshared(nn.Module):
    """
    Deep-Unshared (The Deep Reference Baseline):
    T distinct Linear(H, H) layers. Full parameter reference model every other variant is compared against.
    """
    def __init__(self, in_dim=2, out_dim=2, H=64, depth=6):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.layers = nn.ModuleList([nn.Linear(H, H) for _ in range(depth)])
        self.out = nn.Linear(H, out_dim)
        self.depth = depth

    def forward(self, x):
        h0 = F.relu(self.inp(x))
        h = h0
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.out(h)


class MLPSharedSingle(nn.Module):
    """
    Shared-Single with input injection:
    ONE Linear(H, H) applied T times recurrently: h = relu(W h + h0).
    """
    def __init__(self, in_dim=2, out_dim=2, H=64, depth=6, inject_input=True):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.shared = nn.Linear(H, H)
        self.out = nn.Linear(H, out_dim)
        self.depth = depth
        self.inject = inject_input

    def forward(self, x):
        h0 = F.relu(self.inp(x))
        h = h0
        for t in range(self.depth):
            if self.inject:
                h = F.relu(self.shared(h) + h0)
            else:
                h = F.relu(self.shared(h))
        return self.out(h)


class MLPSharedCyclic(nn.Module):
    """
    Shared-N-Cyclic:
    A small bank of n distinct layers applied in repeating cycle (0, 1, ..., n-1)
    until total depth T is reached.
    """
    def __init__(self, in_dim=2, out_dim=2, H=64, n_layers=3, depth=6, inject_input=True):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.bank = nn.ModuleList([nn.Linear(H, H) for _ in range(n_layers)])
        self.out = nn.Linear(H, out_dim)
        self.depth = depth
        self.n = n_layers
        self.inject = inject_input

    def forward(self, x):
        h0 = F.relu(self.inp(x))
        h = h0
        for t in range(self.depth):
            step_out = self.bank[t % self.n](h)
            h = F.relu(step_out + (h0 if self.inject else 0))
        return self.out(h)


class MLPInternalRouter(nn.Module):
    """
    Internal-Router:
    A bank of n candidate layers. At each step t, a router head reads the current
    hidden state h_t and routes soft (Softmax) or hard (Gumbel-Softmax ST).
    """
    def __init__(self, in_dim=2, out_dim=2, H=64, n_bank=3, depth=6, hard=False):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.bank = nn.ModuleList([nn.Linear(H, H) for _ in range(n_bank)])
        self.router = nn.Linear(H, n_bank)
        self.out = nn.Linear(H, out_dim)
        self.depth = depth
        self.hard = hard

    def forward(self, x):
        h0 = F.relu(self.inp(x))
        h = h0
        for t in range(self.depth):
            logits = self.router(h)
            if self.hard:
                w = F.gumbel_softmax(logits, tau=1.0, hard=True)
            else:
                w = F.softmax(logits, dim=-1)
            outs = torch.stack([layer(h) for layer in self.bank], dim=-1) # (B, H, n)
            mixed = (outs * w.unsqueeze(1)).sum(-1)
            h = F.relu(mixed + h0)
        return self.out(h)


class MLPRoutingController(nn.Module):
    def __init__(self, in_dim, depth, n_bank, ctrl_hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, ctrl_hidden),
            nn.ReLU(),
            nn.Linear(ctrl_hidden, depth * n_bank),
        )
        self.depth, self.n_bank = depth, n_bank

    def forward(self, x):
        return self.net(x).view(-1, self.depth, self.n_bank)


class MLPExternalController(nn.Module):
    """
    External-Controller:
    A small separate network reads raw x and outputs the routing plan for all T steps.
    """
    def __init__(self, in_dim=2, out_dim=2, H=64, n_bank=3, depth=6, hard=False, ctrl_hidden=16):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.bank = nn.ModuleList([nn.Linear(H, H) for _ in range(n_bank)])
        self.controller = MLPRoutingController(in_dim, depth, n_bank, ctrl_hidden)
        self.out = nn.Linear(H, out_dim)
        self.depth = depth
        self.hard = hard

    def forward(self, x):
        plan_logits = self.controller(x) # (B, T, n)
        h0 = F.relu(self.inp(x))
        h = h0
        for t in range(self.depth):
            logits_t = plan_logits[:, t, :]
            w = F.gumbel_softmax(logits_t, tau=1.0, hard=True) if self.hard else F.softmax(logits_t, -1)
            outs = torch.stack([layer(h) for layer in self.bank], dim=-1)
            h = F.relu((outs * w.unsqueeze(1)).sum(-1) + h0)
        return self.out(h), plan_logits


class MLPAdaptiveDepthController(nn.Module):
    """
    Adaptive-Depth-Controller:
    ONE shared layer, but a controller predicts per-step halting probability.
    Final output is probability-weighted blend of hidden states at each depth.
    """
    def __init__(self, in_dim=2, out_dim=2, H=64, max_depth=6, ctrl_hidden=16):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.shared = nn.Linear(H, H)
        self.halt_ctrl = nn.Sequential(
            nn.Linear(in_dim, ctrl_hidden),
            nn.ReLU(),
            nn.Linear(ctrl_hidden, max_depth)
        )
        self.out = nn.Linear(H, out_dim)
        self.max_depth = max_depth

    def forward(self, x):
        halt_logits = self.halt_ctrl(x)
        continue_prob = torch.sigmoid(halt_logits)
        keep = torch.cumprod(continue_prob, dim=1)
        halt_at = torch.zeros_like(keep)
        halt_at[:, 0] = 1 - continue_prob[:, 0]
        for t in range(1, self.max_depth):
            halt_at[:, t] = keep[:, t - 1] * (1 - continue_prob[:, t])
        halt_at[:, -1] = halt_at[:, -1] + keep[:, -1]

        h0 = F.relu(self.inp(x))
        h = h0
        weighted_out = 0
        for t in range(self.max_depth):
            h = F.relu(self.shared(h) + h0)
            weighted_out = weighted_out + halt_at[:, t:t+1] * h
        return self.out(weighted_out), halt_at


# ============================================================================
# 2. Conv / ResNet Architectures (for CIFAR-100 Benchmark)
# ============================================================================

class ConvBasicBlock(nn.Module):
    """Residual Conv Block: Conv-BN-ReLU-Conv-BN."""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


class ConvStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvHead(nn.Module):
    def __init__(self, channels=64, num_classes=100):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvShallow(nn.Module):
    """
    Shallow Conv Baseline:
    1 BasicBlock applied once. Matches the parameter count of ConvSharedSingle.
    """
    def __init__(self, num_classes=100, channels=64):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.block = ConvBasicBlock(channels)
        self.head = ConvHead(channels, num_classes)

    def forward(self, x):
        h0 = self.stem(x)
        h = F.relu(self.block(h0) + h0)
        return self.head(h)


class ConvDeepUnshared(nn.Module):
    """
    Deep Unshared Conv Baseline (The Deep Reference):
    T independent BasicBlocks with residual shortcuts and h0 injection.
    """
    def __init__(self, num_classes=100, channels=64, depth=6):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.layers = nn.ModuleList([ConvBasicBlock(channels) for _ in range(depth)])
        self.head = ConvHead(channels, num_classes)
        self.depth = depth

    def forward(self, x):
        h0 = self.stem(x)
        h = h0
        for layer in self.layers:
            h = F.relu(layer(h) + h + h0)
        return self.head(h)


class ConvSharedSingle(nn.Module):
    """
    Shared-Single Conv:
    ONE BasicBlock reused depth T times with residual shortcut + h0 injection.
    """
    def __init__(self, num_classes=100, channels=64, depth=6):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.shared = ConvBasicBlock(channels)
        self.head = ConvHead(channels, num_classes)
        self.depth = depth

    def forward(self, x):
        h0 = self.stem(x)
        h = h0
        for t in range(self.depth):
            h = F.relu(self.shared(h) + h + h0)
        return self.head(h)


class ConvSharedCyclic(nn.Module):
    """
    Shared-N-Cyclic Conv:
    A bank of n distinct BasicBlocks cycled repeatedly to depth T.
    """
    def __init__(self, num_classes=100, channels=64, n_layers=3, depth=6):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.bank = nn.ModuleList([ConvBasicBlock(channels) for _ in range(n_layers)])
        self.head = ConvHead(channels, num_classes)
        self.depth = depth
        self.n = n_layers

    def forward(self, x):
        h0 = self.stem(x)
        h = h0
        for t in range(self.depth):
            h = F.relu(self.bank[t % self.n](h) + h + h0)
        return self.head(h)


class ConvInternalRouter(nn.Module):
    """
    Internal Router Conv:
    Bank of n BasicBlocks. At each step, a router head reads the pooled hidden state
    and dynamically mixes or selects blocks (Softmax or Gumbel-Softmax ST).
    """
    def __init__(self, num_classes=100, channels=64, n_bank=3, depth=6, hard=False):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.bank = nn.ModuleList([ConvBasicBlock(channels) for _ in range(n_bank)])
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, n_bank)
        )
        self.head = ConvHead(channels, num_classes)
        self.depth = depth
        self.hard = hard

    def forward(self, x):
        h0 = self.stem(x)
        h = h0
        for t in range(self.depth):
            logits = self.router(h)
            if self.hard:
                w = F.gumbel_softmax(logits, tau=1.0, hard=True)
            else:
                w = F.softmax(logits, dim=-1)
            outs = torch.stack([layer(h) for layer in self.bank], dim=-1) # (B, C, H, W, n)
            w_reshaped = w.view(w.size(0), 1, 1, 1, w.size(1))
            mixed = (outs * w_reshaped).sum(-1)
            h = F.relu(mixed + h + h0)
        return self.head(h)


class ConvExternalPlanController(nn.Module):
    def __init__(self, in_channels, depth, n_bank, ctrl_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(in_channels * 8 * 8, ctrl_hidden),
            nn.ReLU(),
            nn.Linear(ctrl_hidden, depth * n_bank),
        )
        self.depth, self.n_bank = depth, n_bank

    def forward(self, x):
        return self.net(x).view(-1, self.depth, self.n_bank)


class ConvExternalController(nn.Module):
    """
    External Controller Conv:
    Separate controller network predicts routing plan for all T steps from raw input.
    """
    def __init__(self, num_classes=100, channels=64, n_bank=3, depth=6, hard=False):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.bank = nn.ModuleList([ConvBasicBlock(channels) for _ in range(n_bank)])
        self.controller = ConvExternalPlanController(3, depth, n_bank)
        self.head = ConvHead(channels, num_classes)
        self.depth = depth
        self.hard = hard

    def forward(self, x):
        plan_logits = self.controller(x) # (B, T, n)
        h0 = self.stem(x)
        h = h0
        for t in range(self.depth):
            logits_t = plan_logits[:, t, :]
            w = F.gumbel_softmax(logits_t, tau=1.0, hard=True) if self.hard else F.softmax(logits_t, -1)
            outs = torch.stack([layer(h) for layer in self.bank], dim=-1)
            w_reshaped = w.view(w.size(0), 1, 1, 1, w.size(1))
            h = F.relu((outs * w_reshaped).sum(-1) + h + h0)
        return self.head(h), plan_logits


class ConvAdaptiveDepthController(nn.Module):
    """
    Adaptive Depth Controller Conv:
    ONE shared BasicBlock, controller predicts halting distribution across steps.
    """
    def __init__(self, num_classes=100, channels=64, max_depth=6):
        super().__init__()
        self.stem = ConvStem(3, channels)
        self.shared = ConvBasicBlock(channels)
        self.halt_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, max_depth)
        )
        self.head = ConvHead(channels, num_classes)
        self.max_depth = max_depth

    def forward(self, x):
        halt_logits = self.halt_ctrl(x)
        continue_prob = torch.sigmoid(halt_logits)
        keep = torch.cumprod(continue_prob, dim=1)
        halt_at = torch.zeros_like(keep)
        halt_at[:, 0] = 1 - continue_prob[:, 0]
        for t in range(1, self.max_depth):
            halt_at[:, t] = keep[:, t - 1] * (1 - continue_prob[:, t])
        halt_at[:, -1] = halt_at[:, -1] + keep[:, -1]

        h0 = self.stem(x)
        h = h0
        weighted_out = 0
        for t in range(self.max_depth):
            h = F.relu(self.shared(h) + h + h0)
            halt_weight = halt_at[:, t:t+1].view(-1, 1, 1, 1)
            weighted_out = weighted_out + halt_weight * h
        return self.head(weighted_out), halt_at


# ============================================================================
# 3. Model Helpers and Parameter Counting
# ============================================================================

def count_parameters(model):
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_load_balance_loss(plan_logits, n_bank):
    """Standard MoE load-balancing auxiliary loss to prevent expert collapse."""
    probs = F.softmax(plan_logits, dim=-1)
    usage = probs.mean(dim=(0, 1))
    target = torch.full_like(usage, 1.0 / n_bank)
    return F.mse_loss(usage, target)
