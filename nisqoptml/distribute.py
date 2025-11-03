"""
Distributed quantum circuit execution using MPI.

This module provides functions for distributing quantum circuit execution
across multiple nodes using MPI (Message Passing Interface). Supports both
distributed circuit execution and federated aggregation.
"""

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except (ImportError, RuntimeError):
    # Fallback when MPI is not available
    MPI_AVAILABLE = False
    MPI = None
    comm = None
    rank = 0
    size = 1

import numpy as np


def distribute_circuit(qnode, x, params, shots):
    """
    Distribute quantum circuit execution across multiple nodes.
    
    Splits the total number of shots across available MPI processes,
    executes locally, and aggregates results.
    
    Args:
        qnode: Quantum node (circuit) to execute
        x: Input features
        params: Circuit parameters
        shots: Total number of measurement shots
    
    Returns:
        Aggregated prediction from all nodes
    """
    if not MPI_AVAILABLE:
        # Fallback: execute on single node
        return qnode(x, params)
    
    # Distribute shots across nodes
    local_shots = shots // size
    
    # Execute locally
    local_pred = sum(qnode(x, params) for _ in range(local_shots)) / local_shots
    
    # Aggregate results from all nodes
    global_pred = comm.allreduce(local_pred, op=MPI.SUM) / size
    
    return global_pred


def federated_aggregate(params):
    """
    Aggregate model parameters in federated learning (FedAvg).
    
    Implements Federated Averaging algorithm to combine parameters
    from multiple nodes using MPI reduction operations.
    
    Args:
        params: Local model parameters (numpy array)
    
    Returns:
        Averaged parameters across all nodes
    """
    if not MPI_AVAILABLE:
        # Single-node mode: return parameters unchanged
        return params
    
    # Convert to numpy array for MPI operations
    params_np = np.array(params)
    global_params = np.zeros_like(params_np)
    
    # Sum parameters from all nodes
    comm.Allreduce(params_np, global_params, op=MPI.SUM)
    
    # Average across nodes (FedAvg)
    return global_params / size
