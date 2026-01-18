"""
Run all foundation tests.

Execute from project root: python -m pytest src/test/
Or run this file directly: python src/test/run_tests.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the test module
from test_foundation import main as run_foundation_tests

if __name__ == "__main__":
    run_foundation_tests()
