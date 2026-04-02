"""
Loss functions for graph condensation.
"""
import torch
from torch import nn
from .kernels import RBF, RBF_eff, LinearKernel, PoliKernel, LaplaceKernel


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss for distribution matching.
    
    Args:
        kernel_type: Type of kernel ('RBF', 'Lin', 'Poly', 'Lap').
    """
    def __init__(self, kernel_type: str = 'RBF'):
        super().__init__()
        if kernel_type == 'RBF':
            self.kernel = RBF()
        elif kernel_type == 'Lin':
            self.kernel = LinearKernel()
        elif kernel_type == 'Poly':
            self.kernel = PoliKernel()
        elif kernel_type == 'Lap':
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        """Compute MMD loss between X and Y.
        
        Args:
            X: Samples from first distribution of shape (N, ...).
            Y: Samples from second distribution of shape (M, ...).
            
        Returns:
            MMD loss value.
        """
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class MMDLoss_eff(nn.Module):
    """Efficient MMD loss with precomputed XX kernel.
    
    Used for large-scale computation where k(X, X) is precomputed
    to avoid repeated computation in the optimization loop.
    """
    def __init__(self):
        super().__init__()
        self.kernel = RBF_eff()
        self.computed_XX = []

    def forward(self, XX, bandwidth, X, Y):
        """Compute MMD loss with precomputed XX kernel.
        
        Args:
            XX: Precomputed mean of k(X, X).
            bandwidth: Precomputed bandwidth.
            X: Samples from first distribution.
            Y: Samples from second distribution.
            
        Returns:
            MMD loss value.
        """
        XY = self.kernel(bandwidth, X, Y).mean()
        YY = self.kernel(bandwidth, Y, Y).mean()
        MMD = XX - 2 * XY + YY
        return MMD
