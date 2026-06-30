# Test Suite

Comprehensive test suite for the Binance Trading Bot enterprise features.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Unit tests for individual components
│   ├── test_risk_manager.py       # Risk management system tests
│   ├── test_order_executor.py     # Smart order execution tests
│   ├── test_fault_tolerance.py    # Fault tolerance tests
│   ├── test_audit_logger.py       # Audit logging tests
│   └── test_monitoring.py         # Metrics collection tests
└── integration/             # Integration tests
    └── test_api.py          # FastAPI endpoint tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit/
```

**Integration tests only:**
```bash
pytest tests/integration/
```

**Specific test file:**
```bash
pytest tests/unit/test_risk_manager.py
```

**Specific test class:**
```bash
pytest tests/unit/test_risk_manager.py::TestPositionSizeController
```

**Specific test function:**
```bash
pytest tests/unit/test_risk_manager.py::TestPositionSizeController::test_validate_normal_position
```

### Run with Markers

**Only unit tests:**
```bash
pytest -m unit
```

**Only integration tests:**
```bash
pytest -m integration
```

**Exclude slow tests:**
```bash
pytest -m "not slow"
```

### Coverage Reports

**Generate HTML coverage report:**
```bash
pytest --cov=src --cov-report=html
```

The report will be available at `htmlcov/index.html`

**Terminal coverage report:**
```bash
pytest --cov=src --cov-report=term-missing
```

### Verbose Output

**Show detailed test output:**
```bash
pytest -v
```

**Show print statements:**
```bash
pytest -s
```

**Show local variables on failure:**
```bash
pytest -l
```

## Test Coverage

### Unit Tests

1. **Risk Management System** (`test_risk_manager.py`)
   - Position size controller
   - Drawdown protection
   - Correlation analyzer
   - Black swan protection
   - Circuit breaker
   - Multi-layer validation

2. **Order Executor** (`test_order_executor.py`)
   - Order book analysis
   - Slippage prediction
   - Market impact estimation
   - Smart order splitting
   - TWAP/VWAP strategies
   - Adaptive execution

3. **Fault Tolerance** (`test_fault_tolerance.py`)
   - State snapshots
   - Automatic failover
   - Connection management
   - Emergency shutdown
   - Trading resumption
   - State restoration

4. **Audit Logger** (`test_audit_logger.py`)
   - Hash chain integrity
   - Event logging
   - Tamper detection
   - Chain verification
   - Concurrent logging

5. **Monitoring** (`test_monitoring.py`)
   - Prometheus metrics
   - Trade recording
   - System metrics
   - Portfolio metrics
   - Risk metrics

### Integration Tests

1. **API Endpoints** (`test_api.py`)
   - Health check
   - Prometheus metrics export
   - Order creation
   - Order history
   - Audit trail
   - System status
   - Emergency controls
   - Portfolio summary

## Test Fixtures

Shared fixtures in `conftest.py`:

- `sample_order`: Normal-sized order for testing
- `large_order`: Large order for testing splitting
- `risky_order`: Order that should trigger risk rejection
- `mock_order_book`: Simulated order book data
- `risk_manager`: Initialized risk management system
- `order_executor`: Initialized order executor
- `fault_tolerant_engine`: Mocked fault-tolerant engine
- `audit_logger`: Audit logger with mocked database
- `metrics_collector`: Prometheus metrics collector

## Writing New Tests

### Unit Test Template

```python
import pytest
from src.services.your_module import YourClass

class TestYourClass:
    """Test YourClass functionality"""

    @pytest.mark.asyncio
    async def test_your_feature(self):
        """Test specific feature"""
        # Arrange
        instance = YourClass()

        # Act
        result = await instance.your_method()

        # Assert
        assert result is not None
```

### Integration Test Template

```python
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

def test_your_endpoint():
    """Test your API endpoint"""
    client = TestClient(app)

    response = client.get("/your/endpoint")

    assert response.status_code == 200
    assert "expected_key" in response.json()
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml
```

## Mocking

Tests use mocking to avoid:
- Real API calls to Binance
- Actual database writes
- Redis connections
- External service dependencies

This ensures:
- Fast test execution
- No external dependencies
- Deterministic results
- Safe to run anywhere

## Troubleshooting

### Import Errors

Ensure the project root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Async Test Warnings

Install pytest-asyncio:
```bash
pip install pytest-asyncio
```

### Database Connection Errors

Tests mock database connections. If you see connection errors, check that mocks are properly configured in the test.

### Missing Dependencies

Install test dependencies:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Keep tests focused
3. **Descriptive names**: Use clear test function names
4. **Mock external dependencies**: Don't rely on external services
5. **Test edge cases**: Include boundary conditions
6. **Use fixtures**: Reuse common setup code
7. **Keep tests fast**: Unit tests should run in milliseconds
8. **Maintain independence**: Tests should not depend on each other

## Maintenance

- Run tests before committing code
- Update tests when changing functionality
- Maintain >80% code coverage
- Add tests for bug fixes
- Review test failures carefully
