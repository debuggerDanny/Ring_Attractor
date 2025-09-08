# Ring Attractor Testing Guide

This guide explains how to set up and run tests for the Ring Attractor project.

## 🚀 Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Using the test runner script
python run_tests.py

# Or using pytest directly
python -m pytest tests/

# On Windows
test.bat

# Using Makefile (Linux/macOS)
make test
```

## 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_attractors.py      # Ring Attractor core tests
│   ├── test_control_layers.py  # Control layer tests
│   └── test_model_manager.py   # Model management tests
├── integration/             # Integration tests
│   ├── test_policy_integration.py  # Policy wrapper tests
│   └── test_end_to_end.py         # Complete workflow tests
└── fixtures/               # Test data and utilities
```

## 🧪 Types of Tests

### Unit Tests
Test individual components in isolation:

```bash
# Run all unit tests
python run_tests.py --unit

# Test specific component
python run_tests.py --test-file tests/unit/test_attractors.py
```

### Integration Tests
Test interactions between components:

```bash
# Run all integration tests
python run_tests.py --integration

# Test policy integration
python run_tests.py --test-file tests/integration/test_policy_integration.py
```

### Performance Tests
Benchmark performance and identify bottlenecks:

```bash
# Run benchmark tests
python run_tests.py --benchmark
```

## 🔧 Test Configuration

### Pytest Markers

Tests are organized with markers for easy selection:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.requires_pyflyt` - Tests requiring PyFlyt
- `@pytest.mark.benchmark` - Performance benchmarks

### Running Specific Test Types

```bash
# Skip slow tests
python run_tests.py --fast

# Run only GPU tests
python -m pytest -m gpu

# Run tests that don't require PyFlyt
python -m pytest -m "not requires_pyflyt"
```

## 📊 Coverage Reporting

### Generate Coverage Report

```bash
# HTML coverage report
python run_tests.py --coverage --html-report

# Terminal coverage report
python run_tests.py --coverage
```

### View Coverage Report

```bash
# Open HTML report (generated in test_results/htmlcov/)
open test_results/htmlcov/index.html  # macOS
start test_results/htmlcov/index.html  # Windows
```

## ⚡ Parallel Testing

Speed up test execution:

```bash
# Run tests in parallel (4 workers)
python run_tests.py --parallel 4

# Auto-detect number of cores
python -m pytest -n auto
```

## 🎯 Testing Specific Components

### Test Ring Attractors

```bash
# Unit tests for Ring Attractor core
python run_tests.py --test-file tests/unit/test_attractors.py

# Test specific Ring Attractor function
python -m pytest tests/unit/test_attractors.py::TestRingAttractor::test_forward_shape
```

### Test Control Layers

```bash
# All control layer tests
python run_tests.py --test-file tests/unit/test_control_layers.py

# Test multi-axis layers
python -m pytest tests/unit/test_control_layers.py::TestMultiAxisRingAttractorLayer
```

### Test Model Management

```bash
# Model manager tests
python run_tests.py --test-file tests/unit/test_model_manager.py

# Test save/load functionality
python -m pytest tests/unit/test_model_manager.py::TestRingAttractorModelManager::test_save_model_full
```

### Test Integration

```bash
# Policy integration tests
python run_tests.py --test-file tests/integration/test_policy_integration.py

# End-to-end workflow tests
python run_tests.py --test-file tests/integration/test_end_to_end.py
```

## 🔍 Debugging Tests

### Verbose Output

```bash
# Maximum verbosity
python run_tests.py --verbose

# Debug mode with no capture
python -m pytest -vvs --tb=long
```

### Test Specific Functions

```bash
# Test single function
python run_tests.py --test-file tests/unit/test_attractors.py --test-function test_forward_shape

# Test single class
python run_tests.py --test-file tests/unit/test_attractors.py --test-class TestRingAttractor
```

## 🛠️ Writing New Tests

### Test File Template

```python
import pytest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from your_module import YourClass

class TestYourClass:
    """Test YourClass functionality"""
    
    def test_basic_functionality(self):
        """Test basic functionality works"""
        obj = YourClass()
        result = obj.method()
        
        assert result is not None
        assert isinstance(result, expected_type)
    
    def test_error_handling(self):
        """Test error handling"""
        obj = YourClass()
        
        with pytest.raises(ValueError):
            obj.method_with_invalid_input()
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6)
    ])
    def test_parametrized(self, input_val, expected):
        """Test with multiple parameter sets"""
        obj = YourClass()
        result = obj.double(input_val)
        assert result == expected
```

### Using Fixtures

```python
def test_with_fixtures(self, ring_config_medium, sample_batch_medium):
    """Test using shared fixtures"""
    layer = SomeLayer(config=ring_config_medium)
    output = layer(sample_batch_medium)
    
    assert output.shape[0] == sample_batch_medium.shape[0]
```

### Testing with Mock Objects

```python
from unittest.mock import Mock, patch

def test_with_mock(self, mock_model):
    """Test using mock objects"""
    result = some_function(mock_model)
    
    assert result is not None
    mock_model.predict.assert_called_once()
```

## 🎛️ Advanced Testing Options

### Environment Variables

```bash
# Set device for testing
CUDA_VISIBLE_DEVICES=0 python run_tests.py

# Set random seed for reproducibility
PYTHONHASHSEED=0 python run_tests.py
```

### Custom Test Configuration

Create `pytest.ini` modifications:

```ini
[tool:pytest]
# Add custom test paths
testpaths = tests custom_tests

# Add custom markers
markers =
    custom: Custom test marker
    experimental: Experimental features
```

### CI/CD Integration

```bash
# Generate JUnit XML for CI systems
python run_tests.py --junit-xml --coverage

# Run tests suitable for CI
python run_tests.py --fast --parallel 2 --coverage --junit-xml
```

## 📈 Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
def test_performance_benchmark(self, benchmark):
    """Benchmark test using pytest-benchmark"""
    layer = RingAttractorLayer()
    input_data = torch.randn(32, 64)
    
    result = benchmark(layer, input_data)
    assert result.shape == (32, 16)
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler pytest-memprof

# Run with memory profiling
python -m pytest --memprof tests/
```

## 🐛 Common Issues and Solutions

### Import Errors

```python
# Ensure project root is in path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
```

### CUDA Issues

```bash
# Skip CUDA tests if not available
python -m pytest -m "not gpu"

# Force CPU testing
CUDA_VISIBLE_DEVICES= python run_tests.py
```

### Slow Tests

```bash
# Skip slow tests during development
python run_tests.py --fast

# Run only fast unit tests
python -m pytest -m "unit and not slow"
```

### Mock Dependencies

```python
# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_deps():
    with patch('external_module.function') as mock:
        mock.return_value = "mocked_result"
        yield mock
```

## 📋 Test Checklist

When adding new functionality, ensure:

- [ ] Unit tests for core functionality
- [ ] Integration tests for component interactions
- [ ] Error handling tests
- [ ] Edge case tests
- [ ] Performance tests for critical paths
- [ ] Mock tests for external dependencies
- [ ] Documentation updates
- [ ] Test coverage > 80%

## 🎯 Testing Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should describe what is being tested
3. **Fixtures**: Use shared fixtures for common setup
4. **Mocking**: Mock external dependencies and I/O
5. **Coverage**: Aim for high test coverage
6. **Speed**: Keep tests fast, mark slow ones appropriately
7. **Determinism**: Tests should be reproducible
8. **Edge Cases**: Test boundary conditions and error cases

## 📚 Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov for coverage](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark for performance](https://pytest-benchmark.readthedocs.io/)
- [pytest-xdist for parallel testing](https://pytest-xdist.readthedocs.io/)