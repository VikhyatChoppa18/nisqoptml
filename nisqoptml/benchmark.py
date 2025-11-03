"""
Benchmarking utilities for NISQOptML.

This module provides functions to benchmark quantum neural networks
on quantum-encoded datasets like MNIST, measuring accuracy, shots,
and training time.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple
from .qnn import QNN


def encode_mnist_digit(digit: np.ndarray, qubits: int = 4) -> np.ndarray:
    """
    Encode MNIST digit (28x28) into quantum feature vector.
    
    Uses amplitude encoding or angle encoding to reduce dimensionality
    to match number of qubits.
    
    Args:
        digit: MNIST digit image (flattened or 2D)
        qubits: Number of qubits available
    
    Returns:
        Encoded feature vector of length qubits
    """
    # Flatten if needed
    if digit.ndim > 1:
        digit = digit.flatten()
    
    # Normalize to [0, 1]
    digit = digit / 255.0
    
    # Downsample/compress to qubits dimensions
    if len(digit) > qubits:
        # Use PCA-like approach or simple averaging
        step = len(digit) // qubits
        encoded = np.array([digit[i*step:(i+1)*step].mean() for i in range(qubits)])
    else:
        # Pad if needed
        encoded = np.pad(digit, (0, max(0, qubits - len(digit))), mode='constant')[:qubits]
    
    # Scale to [0, 2π] for angle encoding
    encoded = encoded * 2 * np.pi
    
    return encoded


def benchmark_model(
    model: QNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 10,
    shots: int = 1000,
    use_federated: bool = False,
    clients: int = 4
) -> Dict[str, float]:
    """
    Benchmark a QNN model and return performance metrics.
    
    Args:
        model: QNN model to benchmark
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        epochs: Number of training epochs
        shots: Number of quantum measurement shots
        use_federated: Whether to use federated training
        clients: Number of federated clients (if applicable)
    
    Returns:
        Dictionary with metrics: accuracy, shots_used, training_time
    """
    # Start timing
    start_time = time.time()
    
    # Encode data if needed (MNIST-like)
    if X_train.shape[1] != model.qubits:
        X_train_encoded = np.array([encode_mnist_digit(x, model.qubits) for x in X_train])
        X_test_encoded = np.array([encode_mnist_digit(x, model.qubits) for x in X_test])
    else:
        X_train_encoded = X_train
        X_test_encoded = X_test
    
    # Encode labels
    if y_train.ndim == 1:
        # One-hot encode if needed
        num_classes = len(np.unique(y_train))
        y_train_encoded = np.zeros((len(y_train), model.qubits))
        y_test_encoded = np.zeros((len(y_test), model.qubits))
        for i, label in enumerate(y_train):
            y_train_encoded[i, :min(num_classes, model.qubits)] = np.eye(num_classes)[label][:model.qubits]
        for i, label in enumerate(y_test):
            y_test_encoded[i, :min(num_classes, model.qubits)] = np.eye(num_classes)[label][:model.qubits]
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    if use_federated and model.federated:
        # Federated training
        # Split data across clients
        data_per_client = len(X_train_encoded) // clients
        local_X = X_train_encoded[:data_per_client]
        local_y = y_train_encoded[:data_per_client]
        
        # Train with federated learning
        model.federated_fit(local_X, local_y, rounds=rounds, local_epochs=2, shots=shots)
    else:
        # Standard training
        model.fit(X_train_encoded, y_train_encoded, epochs=epochs, shots=shots)
    
    # Training time
    training_time = time.time() - start_time
    
    # Evaluate accuracy using model's evaluate method
    accuracy = model.evaluate(X_test_encoded[:100], y_test_encoded[:100])
    
    return {
        'accuracy': accuracy,
        'shots_used': model.shots_used if hasattr(model, 'shots_used') else shots_used,
        'training_time': model.training_time if hasattr(model, 'training_time') else training_time
    }


def run_benchmark_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    qubits: int = 4,
    layers: int = 2
) -> Dict[str, Dict[str, float]]:
    """
    Run full benchmark suite comparing different configurations.
    
    Could return results for:
    - Baseline (no mitigation, no distributed)
    - Distributed mode
    - Federated mode
    - Federated + DP mode
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        qubits: Number of qubits
        layers: Number of layers
    
    Returns:
        Dictionary with benchmark results for each configuration
    """
    results = {}
    
    # Baseline
    print("Benchmarking Baseline...")
    baseline = QNN(layers=layers, qubits=qubits, mitigation='none')
    results['baseline'] = benchmark_model(baseline, X_train, y_train, X_test, y_test, epochs=10, shots=5000)
    
    # Distributed mode
    print("Benchmarking Distributed mode...")
    distributed = QNN(layers=layers, qubits=qubits, mitigation='auto', distributed=True)
    results['distributed'] = benchmark_model(distributed, X_train, y_train, X_test, y_test, epochs=10, shots=3750)
    
    # Federated mode
    print("Benchmarking Federated mode...")
    federated = QNN(layers=layers, qubits=qubits, mitigation='auto', federated=True, dp_sigma=0.0)
    results['federated'] = benchmark_model(federated, X_train, y_train, X_test, y_test, epochs=10, shots=4000, use_federated=True, clients=4)
    
    # Federated + DP
    print("Benchmarking Federated + DP mode...")
    federated_dp = QNN(layers=layers, qubits=qubits, mitigation='auto', federated=True, dp_sigma=0.01)
    results['federated_dp'] = benchmark_model(federated_dp, X_train, y_train, X_test, y_test, epochs=10, shots=4100, use_federated=True, clients=4)
    
    return results


def print_benchmark_table(results: Dict[str, Dict[str, float]]):
    """
    Print benchmark results in a formatted table.
    
    Args:
        results: Dictionary of benchmark results
    """
    print("\n" + "="*80)
    print("Benchmark Results")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'Shots':<12} {'Time (s)':<12}")
    print("-"*80)
    
    baseline = results.get('baseline', {})
    baseline_acc = baseline.get('accuracy', 0)
    
    for name, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        shots = metrics.get('shots_used', 0)
        time_val = metrics.get('training_time', 0)
        
        # Calculate improvements relative to baseline
        acc_improvement = acc - baseline_acc if baseline_acc > 0 else 0
        acc_str = f"{acc:.1f}%"
        if acc_improvement > 0:
            acc_str += f" (+{acc_improvement:.1f}%)"
        
        print(f"{name.capitalize():<30} {acc_str:<12} {int(shots):<12} {time_val:.1f}")
    
    print("="*80 + "\n")

