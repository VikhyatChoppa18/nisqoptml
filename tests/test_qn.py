import nisqoptml as nq
import numpy as np
import pytest

def test_basic_qnn():
    """Test basic QNN creation and training"""
    model = nq.QNN(layers=1, qubits=2)
    X = np.random.rand(5, 2)
    y = np.random.rand(5, 2)
    model.fit(X, y, epochs=1)
    assert model.params is not None
    print("✓ Basic QNN test passed")

def test_qnn_with_mitigation():
    """Test QNN with mitigation enabled"""
    model = nq.QNN(layers=1, qubits=2, mitigation='auto')
    X = np.random.rand(3, 2)
    y = np.random.rand(3, 2)
    model.fit(X, y, epochs=1)
    assert model.mitigation == 'auto'
    print("✓ QNN with mitigation test passed")

def test_qnn_with_zne_mitigation():
    """Test QNN with ZNE mitigation"""
    model = nq.QNN(layers=1, qubits=2, mitigation='zne')
    X = np.random.rand(3, 2)
    y = np.random.rand(3, 2)
    model.fit(X, y, epochs=1)
    assert model.mitigation == 'zne'
    print("✓ QNN with ZNE mitigation test passed")

def test_qnn_explain():
    """Test QNN explain/sensitivity analysis"""
    model = nq.QNN(layers=1, qubits=2)
    X = np.random.rand(3, 2)
    y = np.random.rand(3, 2)
    model.fit(X, y, epochs=1)
    result = model.explain(noise_impact=False)
    assert result is not None
    print("✓ QNN explain test passed")

def test_qnn_explain_with_noise():
    """Test QNN explain with noise impact"""
    model = nq.QNN(layers=1, qubits=2)
    X = np.random.rand(3, 2)
    y = np.random.rand(3, 2)
    model.fit(X, y, epochs=1)
    result = model.explain(noise_impact=True)
    assert result is not None
    print("✓ QNN explain with noise test passed")

def test_qnn_federated_init():
    """Test QNN with federated flag"""
    model = nq.QNN(layers=1, qubits=2, federated=True)
    assert model.federated == True
    assert model.federated == True
    print("✓ QNN federated init test passed")

def test_qnn_federated_requires_flag():
    """Test that federated_fit requires federated=True"""
    model = nq.QNN(layers=1, qubits=2, federated=False)
    X = np.random.rand(3, 2)
    y = np.random.rand(3, 2)
    with pytest.raises(ValueError, match="Set federated=True"):
        model.federated_fit(X, y, rounds=1, local_epochs=1)
    print("✓ QNN federated requires flag test passed")

def test_qnn_dp_sigma():
    """Test QNN with DP noise"""
    model = nq.QNN(layers=1, qubits=2, federated=True, dp_sigma=0.1)
    assert model.dp_sigma == 0.1
    print("✓ QNN DP sigma test passed")

def test_qnn_distributed_flag():
    """Test QNN with distributed flag"""
    model = nq.QNN(layers=1, qubits=2, distributed=True)
    assert model.distributed == True
    print("✓ QNN distributed flag test passed")

def test_qnn_custom_config():
    """Test QNN with custom configuration"""
    model = nq.QNN(layers=2, qubits=4, mitigation='auto', distributed=False, federated=False, dp_sigma=0.05)
    assert model.layers == 2
    assert model.qubits == 4
    assert model.mitigation == 'auto'
    assert model.distributed == False
    assert model.federated == False
    assert model.dp_sigma == 0.05
    print("✓ QNN custom config test passed")

if __name__ == "__main__":
    # Run all tests
    test_basic_qnn()
    test_qnn_with_mitigation()
    test_qnn_with_zne_mitigation()
    test_qnn_explain()
    test_qnn_explain_with_noise()
    test_qnn_federated_init()
    test_qnn_federated_requires_flag()
    test_qnn_dp_sigma()
    test_qnn_distributed_flag()
    test_qnn_custom_config()
    print("\n✓ All tests passed!")
