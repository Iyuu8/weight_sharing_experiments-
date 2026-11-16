"""
Datasets with Gradual Increasing Difficulty of Training.
All tasks share the core property that solving them requires depth (depth separation).

Level 1: 5-Arm Spiral Classification (Geometric depth separation)
Level 2: 4x4 Checkerboard Classification with 3% Label Noise (Combinatorial depth separation)
Level 3: CIFAR-100 (Real-world natural image classification, 100 classes)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ============================================================================
# Level 1: 5-Arm Nested Spiral (Geometric Depth Separation)
# ============================================================================

def generate_spiral_data(n_points_per_arm=500, n_arms=5, noise=0.15, seed=42):
    np.random.seed(seed)
    X = []
    y = []
    for c in range(n_arms):
        r = np.linspace(0.05, 1.0, n_points_per_arm)
        theta = np.linspace(c * (2 * np.pi / n_arms), c * (2 * np.pi / n_arms) + 3.2 * np.pi, n_points_per_arm)
        theta = theta + np.random.randn(n_points_per_arm) * noise
        x1 = r * np.sin(theta)
        x2 = r * np.cos(theta)
        X.append(np.stack([x1, x2], axis=1))
        y.append(np.full(n_points_per_arm, c))
    
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    return X, y

def get_spiral_data(device='cuda', n_train_per_arm=500, n_test_per_arm=200):
    """
    Returns full-batch tensors on device for fast training.
    """
    X_train, y_train = generate_spiral_data(n_points_per_arm=n_train_per_arm, n_arms=5, noise=0.15, seed=42)
    X_test, y_test = generate_spiral_data(n_points_per_arm=n_test_per_arm, n_arms=5, noise=0.15, seed=1337)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    return (X_train_t, y_train_t), (X_test_t, y_test_t), 2, 5


# ============================================================================
# Level 2: 4x4 Checkerboard Classification with Label Noise (Combinatorial Depth)
# ============================================================================

def generate_checkerboard_data(n_samples=3000, grid_size=4, noise_ratio=0.03, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
    cell_x = np.floor((X[:, 0] + 1.0) / 2.0 * grid_size).astype(int)
    cell_y = np.floor((X[:, 1] + 1.0) / 2.0 * grid_size).astype(int)
    cell_x = np.clip(cell_x, 0, grid_size - 1)
    cell_y = np.clip(cell_y, 0, grid_size - 1)

    y = ((cell_x + cell_y) % 2).astype(np.int64)

    if noise_ratio > 0:
        n_flip = int(n_samples * noise_ratio)
        flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)
        y[flip_indices] = 1 - y[flip_indices]

    return X, y

def get_checkerboard_data(device='cuda', n_train=3000, n_test=1000):
    X_train, y_train = generate_checkerboard_data(n_samples=n_train, grid_size=4, noise_ratio=0.03, seed=42)
    X_test, y_test = generate_checkerboard_data(n_samples=n_test, grid_size=4, noise_ratio=0.03, seed=1337)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    return (X_train_t, y_train_t), (X_test_t, y_test_t), 2, 2


# ============================================================================
# Level 3: CIFAR-100 (Natural Image Classification, Hierarchical Depth Required)
# ============================================================================

class FastCIFAR100:
    """
    GPU-resident CIFAR-100 dataset.
    Preloads all 50k train and 10k test images into GPU memory once,
    performing in-place tensor augmentations (random crop + flip) on GPU
    for ultra-fast throughput without CPU/disk bottleneck.
    """
    def __init__(self, data_dir='./data', batch_size=256, device='cuda'):
        self.device = device
        self.batch_size = batch_size

        print("--> Preloading CIFAR-100 onto GPU memory for ultra-fast training...", flush=True)
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

        train_l = DataLoader(trainset, batch_size=50000, shuffle=False)
        test_l = DataLoader(testset, batch_size=10000, shuffle=False)

        x_train, y_train = next(iter(train_l))
        x_test, y_test = next(iter(test_l))

        norm_mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)

        self.x_train = ((x_train - norm_mean) / norm_std).to(device)
        self.y_train = y_train.to(device)
        self.x_test = ((x_test - norm_mean) / norm_std).to(device)
        self.y_test = y_test.to(device)
        self.num_train = self.x_train.size(0)
        self.num_test = self.x_test.size(0)
        print(f"    Loaded {self.num_train} train images and {self.num_test} test images into VRAM.", flush=True)

    def get_train_batches(self):
        perm = torch.randperm(self.num_train, device=self.device)
        for i in range(0, self.num_train, self.batch_size):
            idx = perm[i:i + self.batch_size]
            xb = self.x_train[idx]
            yb = self.y_train[idx]

            # GPU In-place Random Crop (pad 4, random crop 32x32)
            pad_xb = F.pad(xb, (4, 4, 4, 4), mode='reflect')
            ci = torch.randint(0, 9, (1,)).item()
            cj = torch.randint(0, 9, (1,)).item()
            aug_xb = pad_xb[:, :, ci:ci+32, cj:cj+32]

            # GPU Random Horizontal Flip
            if torch.rand(1).item() > 0.5:
                aug_xb = torch.flip(aug_xb, dims=[3])

            yield aug_xb, yb

    def get_test_batches(self):
        for i in range(0, self.num_test, self.batch_size):
            yield self.x_test[i:i + self.batch_size], self.y_test[i:i + self.batch_size]


def get_cifar100_fast(data_dir='./data', batch_size=256, device='cuda'):
    fast_cifar = FastCIFAR100(data_dir=data_dir, batch_size=batch_size, device=device)
    return fast_cifar, 3, 100
