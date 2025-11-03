"""
Utility functions for quantum optimization.

This module provides quantum-aware optimizers and helper functions
for training quantum neural networks.
"""

import sympy as sp
import torch
import torch.optim as optim
from pennylane import numpy as np


class OptimizerWrapper:
    """
    Wrapper for PyTorch optimizers to work with quantum cost functions.
    
    Provides a unified step(cost_fn, params) interface that handles both
    automatic differentiation and finite-difference gradients.
    """
    
    def __init__(self, lr=0.01):
        """
        Initialize the optimizer wrapper.
        
        Args:
            lr: Learning rate for parameter updates
        """
        self.lr = lr
        
    def step(self, cost_fn, params):
        """
        Perform one optimization step.
        
        Args:
            cost_fn: Cost function that takes parameters and returns loss
            params: Current parameter values (numpy array or tensor)
        
        Returns:
            Updated parameters as numpy array
        """
        # Convert parameters to tensor with gradient tracking
        if isinstance(params, np.ndarray):
            params_tensor = torch.tensor(params, requires_grad=True, dtype=torch.float64)
        else:
            params_tensor = torch.tensor(params, requires_grad=True, dtype=torch.float64)
        
        # Create Adam optimizer for this step
        optimizer = optim.Adam([params_tensor], lr=self.lr)
        
        # Compute cost with gradients
        optimizer.zero_grad()
        loss = cost_fn(params_tensor)
        
        # Use automatic differentiation if available
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        else:
            # Fallback to finite differences for gradient estimation
            # This is useful when gradients aren't available
            loss_val = float(loss)
            eps = 1e-5  # Perturbation step size
            grad = np.zeros_like(params)
            params_flat = params.flatten()
            
            # Compute gradients via finite differences
            for i in range(len(params_flat)):
                params_pert = params.copy()
                params_pert.flat[i] += eps
                loss_pert = float(cost_fn(torch.tensor(params_pert, dtype=torch.float64)))
                grad.flat[i] = (loss_pert - loss_val) / eps
            
            # Manual parameter update
            with torch.no_grad():
                params_tensor -= self.lr * torch.tensor(grad, dtype=params_tensor.dtype)
        
        return params_tensor.detach().numpy()


def quantum_optimizer(lr=0.01):
    """
    Create a quantum-aware optimizer.
    
    Uses symbolic simplification to demonstrate quantum-aware optimization
    techniques. In practice, this could be extended with quantum-specific
    optimization strategies.
    
    Args:
        lr: Learning rate for the optimizer
    
    Returns:
        OptimizerWrapper instance
    """
    # Symbolic simplification example - demonstrates quantum-aware approach
    x = sp.symbols('x')
    expr = sp.simplify(x**2 + x)
    print(f"Optimizer initialized - simplified expression: {expr}")
    
    return OptimizerWrapper(lr=lr)
