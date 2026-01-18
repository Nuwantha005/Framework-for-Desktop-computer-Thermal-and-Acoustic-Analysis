#!/usr/bin/env python
"""
Convenience launcher for demos.

Run specific demo:
  python run_demo.py visualization
  python run_demo.py foundation

Or run from demos folder:
  python demos/demo_visualization.py
"""

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent / "demos"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    if len(sys.argv) < 2:
        print("Available demos:")
        print("  python run_demo.py visualization")
        print("  python run_demo.py foundation")
        print("\nOr run directly:")
        print("  python demos/demo_visualization.py")
        print("  python src/test/test_foundation.py")
        sys.exit(1)
    
    demo = sys.argv[1].lower()
    
    if demo == "visualization":
        demo_file = DEMO_DIR / "demo_visualization.py"
    elif demo == "foundation":
        demo_file = Path("src/test/test_foundation.py")
    else:
        print(f"Unknown demo: {demo}")
        sys.exit(1)
    
    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        sys.exit(1)
    
    # Execute the demo
    with open(demo_file) as f:
        code = f.read()
    
    exec(code, {"__name__": "__main__", "__file__": str(demo_file)})

if __name__ == "__main__":
    main()
