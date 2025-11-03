"""
Quantum Neural Network (QNN) implementation for NISQ devices.

This module implements a noise-resilient quantum neural network with support
for federated learning and differential privacy. Designed for use with NISQ
(Noisy Intermediate-Scale Quantum) devices.
"""

import pennylane as qml
from pennylane import numpy as np
import torch
from .mitigate import mitigate_apply
from .distribute import distribute_circuit, federated_aggregate
from .explain import sensitivity_analyzer
from .utils import quantum_optimizer


class QNN:
    """
    Quantum Neural Network for NISQ devices.
    
    Implements a variational quantum circuit with support for error mitigation,
    distributed execution, and federated learning with differential privacy.
    
    Args:
        layers: Number of variational layers in the circuit (default: 2)
        qubits: Number of qubits in the quantum circuit (default: 4)
        mitigation: Error mitigation method - 'none', 'auto', or 'zne' (default: 'none')
        distributed: Enable MPI-based distributed circuit execution (default: False)
        federated: Enable federated learning mode (default: False)
        dp_sigma: Standard deviation for Gaussian DP noise (default: 0.0)
    
    Example:
        >>> model = QNN(layers=2, qubits=4, mitigation='auto')
        >>> model.fit(X, y, epochs=10)
    """
    
    def __init__(self, layers=2, qubits=4, mitigation='none', distributed=False, federated=False, dp_sigma=0.0):
        self.layers = layers
        self.qubits = qubits
        self.mitigation = mitigation
        self.distributed = distributed
        self.federated = federated
        self.dp_sigma = dp_sigma  # Gaussian noise std dev for DP
        
        # Initialize quantum device
        self.dev = qml.device('default.qubit', wires=qubits)
        
        # Random initialization of variational parameters
        # Using uniform distribution over [0, 2π] as per common practice
        self.params = np.random.uniform(0, 2*np.pi, (layers, qubits))
        
        # Quantum-aware optimizer
        self.optimizer = quantum_optimizer()

    def circuit(self, x, params):
        """
        Construct the variational quantum circuit.
        
        Uses alternating rotation gates (RY, RZ) followed by entangling
        CNOT gates. This architecture is well-suited for NISQ devices.
        
        Args:
            x: Input features (classical data)
            params: Variational parameters to optimize
        
        Returns:
            List of expectation values for Pauli-Z operators
        """
        # Apply variational layers
        for i in range(self.layers):
            # Data encoding + parameterized rotations
            for j in range(self.qubits):
                qml.RY(x[j % len(x)], wires=j)  # Encode input data
                qml.RZ(params[i, j], wires=j)    # Parameterized rotation
            
            # Entangling layer - linear connectivity
            for j in range(self.qubits - 1):
                qml.CNOT(wires=[j, j+1])
        
        # Measure expectation values for all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

    def fit(self, X, y, epochs=10, shots=1000):
        """
        Train the quantum neural network.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples, n_outputs)
            epochs: Number of training epochs
            shots: Number of measurement shots for quantum circuit
        """
        # Create QNode with PyTorch interface for automatic differentiation
        qnode = qml.QNode(self.circuit, self.dev, interface='torch')
        
        def cost(params):
            """
            Compute the cost function (MSE loss).
            
            Handles both tensor and numpy inputs for flexibility.
            Supports distributed execution and error mitigation.
            """
            # Handle both tensor and numpy parameter inputs
            if isinstance(params, torch.Tensor):
                params_np = params.detach().numpy()
                params_tensor = params
            else:
                params_np = params
                params_tensor = torch.tensor(params, requires_grad=True, dtype=torch.float64)
            
            preds = []
            for x in X:
                if self.distributed:
                    # Use distributed execution if enabled
                    pred = distribute_circuit(qnode, x, params_np, shots)
                else:
                    # Standard single-node execution
                    pred = qnode(torch.tensor(x, dtype=torch.float64), params_tensor)
                
                # Apply error mitigation if specified
                if self.mitigation != 'none':
                    pred = mitigate_apply(pred, method=self.mitigation)
                
                preds.append(pred)
            
            # Stack predictions into tensor
            if isinstance(preds[0], torch.Tensor):
                pred_tensor = torch.stack(preds) if len(preds) > 1 else preds[0]
            else:
                pred_tensor = torch.tensor(preds, dtype=torch.float64)
            
            # Compute MSE loss
            y_tensor = torch.tensor(y, dtype=torch.float64)
            return ((pred_tensor - y_tensor)**2).mean()
        
        # Training loop
        for epoch in range(epochs):
            self.params = self.optimizer.step(cost, self.params)
        
        print("Training complete.")

    def federated_fit(self, local_X, local_y, rounds=5, local_epochs=2, shots=1000):
        """
        Federated learning training with optional differential privacy.
        
        Implements FedAvg algorithm with local training followed by global
        aggregation. Supports adding Gaussian noise for differential privacy.
        
        Args:
            local_X: Local training features
            local_y: Local training targets
            rounds: Number of federated learning rounds
            local_epochs: Number of local training epochs per round
            shots: Number of measurement shots
        
        Raises:
            ValueError: If federated mode is not enabled
        """
        if not self.federated:
            raise ValueError("Set federated=True in __init__ to use federated learning")
        
        qnode = qml.QNode(self.circuit, self.dev, interface='torch')
        
        def local_cost(params):
            """Compute local cost function for federated learning."""
            # Handle tensor/numpy conversion
            if isinstance(params, torch.Tensor):
                params_np = params.detach().numpy()
                params_tensor = params
            else:
                params_np = params
                params_tensor = torch.tensor(params, requires_grad=True, dtype=torch.float64)
            
            preds = []
            for x in local_X:
                pred = qnode(torch.tensor(x, dtype=torch.float64), params_tensor)
                
                # Apply mitigation if enabled
                if self.mitigation != 'none':
                    pred = mitigate_apply(pred, method=self.mitigation)
                
                preds.append(pred)
            
            # Stack predictions
            if isinstance(preds[0], torch.Tensor):
                pred_tensor = torch.stack(preds) if len(preds) > 1 else preds[0]
            else:
                pred_tensor = torch.tensor(preds, dtype=torch.float64)
            
            y_tensor = torch.tensor(local_y, dtype=torch.float64)
            return ((pred_tensor - y_tensor)**2).mean()
        
        # Federated learning rounds
        for round_num in range(rounds):
            # Local training phase
            for _ in range(local_epochs):
                self.params = self.optimizer.step(local_cost, self.params)
            
            # Add differential privacy noise if enabled
            if self.dp_sigma > 0:
                noise = np.random.normal(0, self.dp_sigma, self.params.shape)
                self.params += noise
            
            # Global aggregation (FedAvg)
            self.params = federated_aggregate(self.params)
            print(f"Round {round_num+1}/{rounds} complete.")
        
        print("Federated training complete.")

    def explain(self, noise_impact=False):
        """
        Generate sensitivity analysis visualization.
        
        Analyzes parameter sensitivity and creates a visualization plot.
        
        Args:
            noise_impact: If True, annotate high noise impact regions
        
        Returns:
            Message confirming plot generation
        """
        return sensitivity_analyzer(self.params, noise_impact)
