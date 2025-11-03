"""
NISQOptML: NISQ-Optimized Machine Learning Library

A noise-resilient Quantum Machine Learning library with support for:
- Quantum Neural Networks (QNN)
- Error mitigation techniques
- Federated learning
- Differential privacy
- Distributed execution

Author: Venkata Vikhyat Choppa
Email: venkata_choppa@outlook.com
"""

from .qnn import QNN

# Benchmark module available for performance testing
try:
    from .benchmark import run_benchmark_suite, print_benchmark_table, encode_mnist_digit
    __all__ = ['QNN', 'run_benchmark_suite', 'print_benchmark_table', 'encode_mnist_digit']
except ImportError:
    __all__ = ['QNN']

__version__ = "0.1.0"
