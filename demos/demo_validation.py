#!/usr/bin/env python3
"""
Solver Validation Demo

Load any case and run solver validation to check:
- Boundary condition satisfaction (Vn ≈ 0)  
- Solver-specific checks (e.g., source strength conservation)
- Generate diagnostic plots

Usage:
    python demos/demo_validation.py cases/cylinder_flow
    python demos/demo_validation.py cases/rounded_square --show-plots
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader

def main():
    parser = argparse.ArgumentParser(description="Run solver validation on any case")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    
    args = parser.parse_args()
    
    # Resolve case directory
    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
    
    print(f"Loading case: {case_dir}")
    
    try:
        # Load case
        case = CaseLoader.load_case(case_dir,mesh_level_index=0)
        print(f"✓ Case loaded: {case.name}")
        
        # Create and solve
        solver = case.create_solver()
        print(f"✓ Created solver: {solver.__class__.__name__}")
        
        print("Solving...")
        solver.solve()
        print("✓ Solver completed")
        
        # Validate
        output_dir = args.output_dir or case.output_dir
        results = solver.validate(output_dir=output_dir, show_plots=args.show_plots)
        
        print(f"\nValidation complete!")
        print(f"Results saved to: {results['output_directory']}")
        
        # Summary
        bc = results['boundary_condition']
        print(f"\nSummary:")
        print(f"  Normal velocity RMS: {bc['Vn_rms']:.2e}")
        print(f"  Normal velocity Max: {bc['Vn_max_abs']:.2e}")
        
        if 'solver_specific' in results:
            ss = results['solver_specific']
            if 'sigma_sum' in ss:
                print(f"  Mass conservation error: {ss['sigma_sum']:.2e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())