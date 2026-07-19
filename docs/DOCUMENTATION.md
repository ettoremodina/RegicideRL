# Documentation policy

Regicide uses Google-style docstrings and `pdoc` to keep the API reference
close to the code that defines it. The goal is to explain contracts, domain
rules, and design intent; docstrings should not narrate statements that are
already clear from the implementation.

## What must be documented

The following objects require docstrings:

- every public module;
- every class;
- every public function and method;
- private functions that implement domain rules, algorithms, persistence,
  multiprocessing, non-obvious transformations, or lifecycle management.

Simple properties, conventional `dunder` methods, and one-line private adapters
do not require a docstring unless their behavior is surprising.

A private function is non-trivial when a maintainer needs information beyond
its name and signature to change it safely. Complexity alone is not the deciding
factor: a short function that enforces a Regicide invariant still needs
documentation.

## Content and style

Start with a short imperative or descriptive summary ending in a period. Add
only the sections that provide useful information:

```python
def choose_action(observation, environment):
    """Choose a legal action from the current public game state.

    The environment is required because search agents clone its hidden state;
    the observation alone is intentionally insufficient.

    Args:
        observation: Observation returned by ``RegicideEnv``.
        environment: Live environment associated with the observation.

    Returns:
        Global action identifier, or ``None`` when no action is legal.

    Raises:
        ValueError: If no environment is supplied.
    """
```

Use these conventions:

- explain why a constraint or fallback exists;
- document mutations, persistence, randomness, and process-safety;
- state units, ranges, shapes, and sentinel values;
- distinguish public information from hidden game state;
- use `Raises` only for exceptions that are part of the contract;
- use `Note` for performance or reproducibility constraints;
- avoid type repetition when annotations already make the type obvious;
- do not put change history, TODOs, or implementation walkthroughs in a
  docstring.

## Source and generated documentation

Hand-authored files directly under `docs/` are curated sources. Generated API
pages belong exclusively in `docs/api/` and are ignored by Git.

Build the complete API reference from the repository root:

```bash
python -m scripts.generate_docs
```

The command documents `game`, `agents`, `solvers`, `ml_logger`, `scripts`,
`ui`, and the top-level `benchmark` module. It replaces only `docs/api/`, never
the curated Markdown files.

## Review checklist

Before completing a documentation change:

1. Run `python -m pytest tests/test_documentation.py`.
2. Run `python -m scripts.generate_docs`.
3. Open `docs/api/index.html` and at least one page from each package.
4. Verify that examples and commands use current global action identifiers and
   canonical `artifacts/` paths.
5. Check that a docstring describes the observable contract rather than an
   outdated implementation detail.
