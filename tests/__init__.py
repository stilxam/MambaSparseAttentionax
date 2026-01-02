"""
Test suite for MambaSparseAttentionax package.

This package contains comprehensive tests for:
- Mamba2 architecture (mambax.py)
- Multi-Head Latent Attention (mhlax.py)
- Manifold-Constrained Hyper-Connection (mHC.py)

Test Structure:
    test_mambax.py: Tests for Mamba2 model, inference cache, and training
    test_mhlax.py: Tests for attention mechanisms and rotary embeddings
    test_mhc.py: Tests for manifold-constrained hyper-connections
    conftest.py: Shared fixtures and test utilities

Usage:
    Run all tests:
        pytest tests/

    Run specific test file:
        pytest tests/test_mambax.py

    Run with coverage:
        pytest tests/ --cov=src --cov-report=html

    Run with verbose output:
        pytest tests/ -v
"""

__all__ = []
