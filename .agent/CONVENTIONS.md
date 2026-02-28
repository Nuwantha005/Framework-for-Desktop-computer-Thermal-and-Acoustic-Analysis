# Project Conventions

## Environment
- Always use mamba environment `fyp`: `mamba activate fyp`
- Never install to base environment
- Package install: `mamba install -c conda-forge <pkg>` or `pip install <pkg>` inside fyp
- Project installed as editable: `pip install -e .`

## Code Style
- **Containers**: Use `@dataclass` for data holders; avoid deep inheritance
- **Type hints**: Annotate arrays with `NDArray[np.float64]` / `NDArray[np.int32]`
- **Array shapes**: Always `(N, 3)` for coordinates (z=0 for 2D); float64 nodes, int32 panel indices
- **Naming**: PascalCase classes, snake_case functions/variables, UPPER_SNAKE_CASE constants
- **Imports**: stdlib → third-party → local; absolute imports from `core`, `solvers`, `visualization`, `postprocessing`
- **Docstrings**: Google style with Args/Returns/Raises
- **Vectorization**: Prefer NumPy vectorized ops over Python loops; use `np.testing` for array assertions
- **Error handling**: Raise `ValueError`/`RuntimeError` with specific context messages

## File Organization
- Source code: `src/{core,solvers,postprocessing,visualization}/`
- Solvers: `src/solvers/panel2d/` (future: `panel3d/`, `boundary_layer/`, `thermal/`, `actuator/`)
- Case definitions: `cases/<name>/case.yaml` + `cases/<name>/shapes/*.json`
- Case outputs: `cases/<name>/out/` (gitignored)
- Validation pipeline: `validation/` (root-level, separate from src)
- Validation results: `validation_results/<case>/` (gitignored)
- Demos: `demos/*.py` (use `sys.path.insert` for imports)
- Tests: `src/test/test_*.py`

## Testing
- Framework: pytest with class-based grouping (`class TestMesh:`)
- Array comparison: `np.testing.assert_array_almost_equal`
- Run: `pytest src/test/` from project root
- No conftest.py or fixtures yet

## Documentation
- Agent context: `.agent/` (compact, updated each session)
- Full docs: `docs/` (MkDocs Material site)
- Docstring format: Google style
- All public classes/functions must have docstrings

## MCP Servers & Agent Tools

### Python MCP Server (`python-mcp-fyp`)
Runs Python in the `fyp` mamba environment. Prefer it over ad-hoc terminal shells for:
- Quick numerical checks: verify influence coefficient formulas, test quadrature rules
- Prototype integrations: test Von Kármán ODE on analytical velocity profiles
- Validate array operations and coordinate transforms
- Import checks: verify a library exists and its API before writing code

### Pylance MCP (`mcp_pylance_*`)
Provides IDE-grade Python intelligence. Use for:
- **`pylanceImports`**: Resolve and add correct import statements when wiring new modules
- **`pylanceDocString`**: Auto-generate Google-style docstrings for new public APIs
- **`pylanceSyntaxErrors`** / **`pylanceFileSyntaxErrors`**: Catch syntax errors before running
- **`pylanceInvokeRefactoring`**: Extract functions, rename symbols, inline variables
- **`pylanceRunCodeSnippet`**: Execute and verify small Python snippets inline
- **`pylanceInstalledTopLevelModules`**: Check what packages are available in the environment

### Agent Context System (`.agent/`)
- **`PROJECT_CONTEXT.md`**: Full project state — read at session start for orientation
- **`CONVENTIONS.md`**: This file — coding style, MCP usage, file organization
- **`modules/*.md`**: Per-module API docs — read the relevant one before modifying a module
- **`prompts/*.md`**: Reusable task prompts — reference when starting a workflow:
  - `implement-solver.md` — 6-step solver implementation guide
  - `validate-vt.md` — tangential velocity debugging loop
  - `implement-bl-solver.md` — Von Kármán BL solver with pluggable velocity profiles
  - `implement-thermal-bl.md` — BDIM thermal BL solver
  - `port-to-3d.md` — 3D panel method + PyVista extension
- **`decisions/*.md`**: Architectural decision records — read before making design choices, write after
- **`TASK_LOG.md`**: Session history — append after each session

### Copilot Custom Instructions
- `.github/copilot-instructions.md` — loaded automatically by VS Code Copilot every conversation
- Keep in sync with AGENTS.md (for OpenCode/other agents) — update both when conventions change
- Contains: environment, architecture, module map, file pointers, active focus, rules

### Workflow Pattern
1. Read relevant `.agent/prompts/*.md` for the task
2. Read relevant `.agent/modules/*.md` for current module state
3. Implement following the prompt's steps
4. Validate using the prompt's success criteria
5. Record decisions in `.agent/decisions/`
6. Update `.agent/PROJECT_CONTEXT.md` status
7. Append to `.agent/TASK_LOG.md`

## Git
- `.gitignore` excludes: `/notes`, `__pycache__/`, `**/out`, `/validation_results`, `**/of_case/`
- AGENTS.md and opencode.json are gitignored (local agent config)
- `.github/copilot-instructions.md` is tracked (Copilot reads it from the repo)
