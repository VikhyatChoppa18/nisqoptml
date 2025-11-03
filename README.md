# NISQOptML

**NISQ-Optimized Machine Learning**: A noise-resilient Quantum Machine Learning library with Federated Learning and Differential Privacy support.

## Overview

This library provides a comprehensive framework for building and training quantum neural networks on NISQ (Noisy Intermediate-Scale Quantum) devices. It addresses key challenges in quantum machine learning including noise mitigation, distributed execution, and privacy-preserving federated learning.

## Key Features

- 🚀 **Quantum Neural Networks (QNN)** - Build and train variational quantum circuits
- 🛡️ **Noise Mitigation** - Automatic error mitigation and zero-noise extrapolation techniques
- 🌐 **Federated Learning** - Train models across distributed nodes with FedAvg
- 🔒 **Differential Privacy** - Add Gaussian noise for privacy-preserving training
- 📊 **Explainability** - Sensitivity analysis and parameter visualization tools
- ⚡ **Distributed Execution** - MPI-based distributed quantum circuit execution

## Installation

```bash
pip install nisqoptml
```

## Quick Start

### Basic QNN Training

```python
import nisqoptml as nq
import numpy as np

# Create a QNN model
model = nq.QNN(layers=2, qubits=4)

# Prepare training data
X = np.random.rand(10, 4)
y = np.random.rand(10, 4)

# Train the model
model.fit(X, y, epochs=10)
```

### With Error Mitigation

```python
# Enable automatic error mitigation
model = nq.QNN(layers=2, qubits=4, mitigation='auto')
model.fit(X, y, epochs=10)

# Or use zero-noise extrapolation
model = nq.QNN(layers=2, qubits=4, mitigation='zne')
model.fit(X, y, epochs=10)
```

### Federated Learning with Differential Privacy

```python
# Create federated QNN with DP noise
model = nq.QNN(
    layers=2, 
    qubits=4, 
    federated=True, 
    dp_sigma=0.01,  # DP noise standard deviation
    mitigation='auto'
)

# Train with federated learning
local_X = np.random.rand(5, 4)
local_y = np.random.rand(5, 4)
model.federated_fit(local_X, local_y, rounds=5, local_epochs=2)
```

### Distributed Execution

```python
# Enable distributed circuit execution (requires MPI)
model = nq.QNN(layers=2, qubits=4, distributed=True)
model.fit(X, y, epochs=10, shots=1000)
```

### Explainability

```python
# Analyze parameter sensitivity
model = nq.QNN(layers=2, qubits=4)
model.fit(X, y, epochs=10)

# Generate sensitivity analysis plot
result = model.explain(noise_impact=True)
print(result)  # Saves sensitivity.png
```

## API Reference

### QNN Class

```python
QNN(
    layers=2,              # Number of quantum layers
    qubits=4,              # Number of qubits
    mitigation='none',     # 'none', 'auto', or 'zne'
    distributed=False,     # Enable distributed execution
    federated=False,       # Enable federated learning
    dp_sigma=0.0          # DP noise standard deviation
)
```

### Methods

- **`fit(X, y, epochs=10, shots=1000)`** - Train the model on provided data
- **`federated_fit(local_X, local_y, rounds=5, local_epochs=2)`** - Federated training with DP support
- **`explain(noise_impact=False)`** - Generate sensitivity analysis visualization

## Requirements

- Python >= 3.8
- PennyLane >= 0.30
- Qiskit >= 1.0
- PyTorch >= 2.0
- NumPy, SymPy, Matplotlib
- MPI4Py (optional, for distributed execution)

## Architecture

The library implements a variational quantum circuit architecture with:
- **Data Encoding**: Rotation gates (RY) for classical data embedding
- **Variational Layers**: Parameterized rotations (RZ) with entangling gates (CNOT)
- **Measurement**: Pauli-Z expectation values for all qubits

## Error Mitigation

Two mitigation strategies are available:
1. **Auto**: Neural network-based error prediction and correction
2. **ZNE**: Zero-noise extrapolation using simple scaling factors

## Federated Learning

Implements the Federated Averaging (FedAvg) algorithm:
- Local training on each node
- Optional differential privacy noise injection
- Global parameter aggregation via MPI

## License

Apache License 2.0

## Citation

If you use this library in your research, please cite:

```
@software{nisqoptml,
  author = {Venkata Vikhyat Choppa},
  title = {NISQOptML: NISQ-Optimized Machine Learning},
  version = {0.1.0},
  year = {2025},
  url = {https://github.com/yourusername/nisqoptml}
}
```

## Author

**Venkata Vikhyat Choppa**  
Email: venkata_choppa@outlook.com
Github: https://github.com/vikhyatchoppa18

## Acknowledgments

Built with [PennyLane](https://pennylane.ai/), [Qiskit](https://qiskit.org/), and [PyTorch](https://pytorch.org/).
