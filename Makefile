.PHONY: help test test-unit test-integration test-coverage test-fast install-dev clean lint format check

# Default target
help:
	@echo "Ring Attractor Testing Makefile"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  help              Show this help message"
	@echo "  install-dev       Install development dependencies"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-coverage     Run tests with coverage reporting"
	@echo "  test-fast         Run fast tests (skip slow ones)"
	@echo "  test-benchmark    Run benchmark tests"
	@echo "  test-parallel     Run tests in parallel"
	@echo "  lint              Run code linting"
	@echo "  format            Format code"
	@echo "  check             Run all quality checks"
	@echo "  clean             Clean test artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test                    # Run all tests"
	@echo "  make test-unit               # Unit tests only"
	@echo "  make test-coverage           # Tests with coverage"
	@echo "  make test TEST_FILE=tests/unit/test_attractors.py  # Specific file"

# Variables
PYTHON := python
PYTEST := $(PYTHON) -m pytest
TEST_RUNNER := $(PYTHON) run_tests.py
TEST_DIR := tests
COVERAGE_DIR := test_results/htmlcov
JUNIT_XML := test_results/junit.xml

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-test.txt
	@echo "[OK] Development dependencies installed"

# Run all tests
test:
	@echo "Running all tests..."
	$(TEST_RUNNER) --all

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	$(TEST_RUNNER) --unit

# Run integration tests only  
test-integration:
	@echo "Running integration tests..."
	$(TEST_RUNNER) --integration

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(TEST_RUNNER) --all --coverage --html-report
	@echo "Coverage report available at: $(COVERAGE_DIR)/index.html"

# Run fast tests (skip slow ones)
test-fast:
	@echo "Running fast tests..."
	$(TEST_RUNNER) --all --fast

# Run benchmark tests
test-benchmark:
	@echo "Running benchmark tests..."
	$(TEST_RUNNER) --benchmark

# Run tests in parallel
test-parallel:
	@echo "Running tests in parallel..."
	$(TEST_RUNNER) --all --parallel 4

# Run specific test file
test-file:
	@if [ -z "$(TEST_FILE)" ]; then \
		echo "Error: TEST_FILE not specified. Use: make test-file TEST_FILE=path/to/test.py"; \
		exit 1; \
	fi
	@echo "Running test file: $(TEST_FILE)"
	$(TEST_RUNNER) --test-file $(TEST_FILE)

# Run specific test function
test-function:
	@if [ -z "$(TEST_FILE)" ] || [ -z "$(TEST_FUNCTION)" ]; then \
		echo "Error: TEST_FILE and TEST_FUNCTION required. Use: make test-function TEST_FILE=path/to/test.py TEST_FUNCTION=test_name"; \
		exit 1; \
	fi
	@echo "Running test function: $(TEST_FILE)::$(TEST_FUNCTION)"
	$(TEST_RUNNER) --test-file $(TEST_FILE) --test-function $(TEST_FUNCTION)

# Run tests with verbose output and JUnit XML
test-ci:
	@echo "Running tests for CI/CD..."
	$(TEST_RUNNER) --all --coverage --junit-xml --verbose --parallel 2
	@echo "CI test results saved to: $(JUNIT_XML)"

# Linting (if you have linting tools)
lint:
	@echo "Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		flake8 --max-line-length=100 --ignore=E203,W503 .; \
	else \
		echo "flake8 not found, skipping..."; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		echo "Running pylint..."; \
		pylint --disable=C0114,C0115,C0116 *.py tests/ || true; \
	else \
		echo "pylint not found, skipping..."; \
	fi

# Code formatting (if you have formatting tools)
format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		echo "Running black..."; \
		black --line-length=100 .; \
	else \
		echo "black not found, skipping..."; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		echo "Running isort..."; \
		isort .; \
	else \
		echo "isort not found, skipping..."; \
	fi

# Quality checks
check: lint test-fast
	@echo "All quality checks completed"

# Clean test artifacts
clean:
	@echo "Cleaning test artifacts..."
	rm -rf test_results/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[OK] Test artifacts cleaned"

# Quick test for development
dev-test: test-fast

# Full test suite for release
release-test: test-coverage lint

# Watch mode (if you have pytest-watch installed)
test-watch:
	@if command -v ptw >/dev/null 2>&1; then \
		echo "Running tests in watch mode..."; \
		ptw --runner "$(TEST_RUNNER) --fast"; \
	else \
		echo "pytest-watch not found. Install with: pip install pytest-watch"; \
	fi

# Debug mode - run with maximum verbosity and no capture
test-debug:
	@echo "Running tests in debug mode..."
	$(PYTEST) -vvs --tb=long $(TEST_DIR)/

# Test specific component
test-attractors:
	$(TEST_RUNNER) --test-file tests/unit/test_attractors.py

test-control-layers:
	$(TEST_RUNNER) --test-file tests/unit/test_control_layers.py

test-model-manager:
	$(TEST_RUNNER) --test-file tests/unit/test_model_manager.py

test-policy-integration:
	$(TEST_RUNNER) --test-file tests/integration/test_policy_integration.py

test-end-to-end:
	$(TEST_RUNNER) --test-file tests/integration/test_end_to_end.py