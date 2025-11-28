# Legacy Tests Archive

This directory contains legacy test files from earlier development phases of the ReasonForge MCP Server project.

## Status: ARCHIVED (January 2025)

These tests have been **replaced** by the comprehensive test suite now located in each package's `tests/` directory:

- `packages/reasonforge-expressions/tests/` - 32 tests (15 tools)
- `packages/reasonforge-algebra/tests/` - 37 tests (18 tools)
- `packages/reasonforge-analysis/tests/` - 31 tests (17 tools)
- `packages/reasonforge-geometry/tests/` - 28 tests (15 tools)
- `packages/reasonforge-statistics/tests/` - 35 tests (16 tools)
- `packages/reasonforge-physics/tests/` - 16 tests (16 tools)
- `packages/reasonforge-logic/tests/` - 13 tests (13 tools)

**Total**: 192 tests covering all 110 tools (100% coverage)

## Why Archived?

1. **Test Organization**: Tests are now organized by package for better maintainability
2. **Complete Coverage**: New comprehensive test suite covers all tools systematically
3. **Consistent Structure**: All tests follow the same pattern and use BaseReasonForgeServer
4. **Phase 3 Complete**: Test reorganization was part of Phase 3 improvements

## What's in This Directory?

These legacy files were created during initial development and exploration:

- Feature-specific tests (e.g., `test_calculus_operations.py`, `test_matrix_operations.py`)
- Experimental tests (e.g., `test_hybrid_tools.py`, `test_quantum_tools.py`)
- Early integration tests (e.g., `test_integration_complete.py`, `test_smoke.py`)
- Old server tests (e.g., `test_server.py`, `test_packages.py`)

## Should These Be Deleted?

**Not immediately**. They are kept as reference in case:
- They contain unique test cases not yet covered
- They test edge cases worth preserving
- Future development needs examples from early phases

## Migration Date

- Archived: January 22, 2025
- Replaced by: Package-specific comprehensive tests
- Reason: Phase 3 test reorganization

---

For current testing, always use: `pytest packages/`
