"""Pytest configuration for reasonforge-statistics tests."""

import sys
import os
from pathlib import Path

# Get project root (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent.parent

# Add package source directories to path
sys.path.insert(0, str(project_root / "packages" / "reasonforge" / "src"))
sys.path.insert(0, str(project_root / "packages" / "reasonforge-statistics" / "src"))
