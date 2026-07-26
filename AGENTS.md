# Development Instructions

## Project overview

This repository develops homotopy-path optimization methods for multivariate
polynomial systems.

The multivariate implementation is the primary implementation.
The existing univariate implementation is a preliminary experiment and must
remain available for reproducibility.

## Design documents

Before implementation, read the documents relevant to the assigned task under:

- `docs/design/mathematics/`
- `docs/design/architecture/`
- `docs/design/implementation/`

Treat these documents as the source of truth for the current design.

If the implementation request conflicts with the design documents, do not
silently introduce a new specification. Report the conflict and choose the
most conservative implementation unless the task explicitly instructs
otherwise.

## Python environment

Use `python3`, not `python`.

Install the Python package and test dependencies from the repository root with:

```bash
python3 -m pip install -e ".[test]"
```

Run all Python tests from the repository root with:

```bash
python3 -m pytest -q
```

Do not report that tests passed unless the test command was actually executed.

Do not skip tests only because the `python` command is unavailable. Use
`python3`.

## Reference Docker environment

The reference execution environment is the project Docker container.

Inside the reference container, the repository root is:

```text
/app
```

The reference Python test command is:

```bash
cd /app
python3 -m pytest -q
```

When working in Codex Cloud and the user's local Docker container is not
accessible, run the equivalent commands in the Codex environment and clearly
state that the tests were not run in the reference Docker container.

## Implementation scope

- Implement only the phase or task explicitly requested.
- Do not begin a later implementation phase without being asked.
- Avoid unrelated large-scale refactoring.
- Do not delete or break the existing univariate implementation.
- Design reusable code around multivariate polynomial systems.
- Preserve backward compatibility when reasonably possible.
- Prefer simple, explicit data structures over unnecessary abstractions.

## Python coding rules

- Use type hints for public functions and methods.
- Add docstrings that document array shapes, dtypes, and coefficient ordering.
- Use NumPy `Generator` objects for random sampling.
- Do not depend on NumPy's global random state.
- Do not mutate input arrays unless mutation is explicitly documented.
- Validate array shapes and reject invalid inputs with descriptive exceptions.
- Keep the Python and Julia coefficient order deterministic and consistent.

## Mathematical invariants

For the current Pham-type polynomial-system implementation:

- The leading monomial of equation `i` is `x_i^{d_i}`.
- Every leading coefficient is fixed to `1 + 0j`.
- The start system is `G_i(x) = x_i^{d_i} - 1`.
- The initial implementation does not use the gamma trick.
- The support and coefficient ordering are fixed for a given system
  specification.
- Bézier actions must not change the start point or target point.
- Bézier actions must not change leading coefficients.
- A zero latent action must produce the linear coefficient path.

Do not change these invariants unless the task explicitly requests a design
change and the related design documents are updated.

## Testing rules

Add or update tests for every behavior changed by the task.

For Python changes, run:

```bash
python3 -m pytest -q
```

For Julia changes, use the Julia test command documented in the relevant
design or implementation document.

For Python–Julia integration changes, run both the unit tests and the
integration tests required by the task.

If a required runtime or dependency is unavailable:

1. Attempt the documented installation or setup procedure.
2. Do not replace the requested test with an unrelated test.
3. Clearly report which tests could not be executed and why.

## Completion report

At the end of each task, report:

- What was implemented or changed.
- Files added, modified, or removed.
- Commands that were actually executed.
- Python and Julia versions used, when relevant.
- Test counts and results.
- The environment in which tests were executed.
- Any unresolved issues or design decisions.