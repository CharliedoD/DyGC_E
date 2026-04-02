"""
Kernel functions for MMD loss computation.
"""
import torch
from torch import nn


class RBF(nn.Module):
    """RBF (Radial Basis Function) kernel for MMD computation.
    
    Args:
        n_kernels: Number of kernels with different bandwidths.
        mul_factor: Multiplication factor for bandwidth scaling.
        bandwidth: Fixed bandwidth value (if None, computed from data).
    """
    def __init__(self, n_kernels: int = 5, mul_factor: float = 2.0, bandwidth: float = None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        flat_X = X.view(X.shape[0], -1)
        L2_distances_X = torch.cdist(flat_X, flat_X) ** 2
        K_X = -L2_distances_X[None, ...] / (
            self.get_bandwidth(L2_distances_X) * self.bandwidth_multipliers.to(L2_distances_X.device)[:, None, None]
        )
        K = torch.exp(K_X).sum(dim=0)
        return K


class RBF_eff(nn.Module):
    """Efficient RBF kernel with precomputed bandwidth.
    
    Used for large-scale MMD computation where bandwidth is precomputed
    to avoid repeated computation in the optimization loop.
    
    Args:
        n_kernels: Number of kernels with different bandwidths.
        mul_factor: Multiplication factor for bandwidth scaling.
    """
    def __init__(self, n_kernels: int = 5, mul_factor: float = 2.0):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)

    def forward(self, bandwidth, X, Y):
        """Compute kernel matrix between X and Y.
        
        Args:
            bandwidth: Precomputed bandwidth value.
            X: First input tensor of shape (N, T, D) or (N, D).
            Y: Second input tensor of shape (M, T, D) or (M, D).
            
        Returns:
            Kernel matrix of shape (N, M).
        """
        flat_X = X.view(X.shape[0], -1)
        flat_Y = Y.view(Y.shape[0], -1)
        
        L2_distances_F = torch.cdist(flat_X, flat_Y) ** 2
        K = -L2_distances_F[None, ...] / (
            bandwidth * self.bandwidth_multipliers.to(L2_distances_F.device)[:, None, None]
        )
        K = torch.exp(K).sum(dim=0)
        return K


class PoliKernel(nn.Module):
    """Polynomial kernel."""
    def __init__(self, constant_term: int = 1, degree: int = 2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        K = (torch.matmul(X, X.t()) + self.constant_term) ** self.degree
        return K


class LinearKernel(nn.Module):
    """Linear kernel."""
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        K = torch.matmul(X, X.t())
        return K


class LaplaceKernel(nn.Module):
    """Laplace kernel."""
    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] * (self.gammas)[:, None, None]).sum(dim=0)
