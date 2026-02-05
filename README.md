# Continual Learning (v2)

This repo is reorganized into two versions:

- v1/ contains the original implementation (archived as-is).
- v2/ contains the updated, theory-aligned implementation with explicit asynchronicity support.

## Quick Start (v2)

Run inside Docker (per project policy):

```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.10 \
  bash -lc "pip install -r v2/requirements.txt && python v2/example.py"
```

Validation (idiot check):

```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.10 \
  bash -lc "pip install -r v2/requirements.txt && python v2/validate_predictor.py"
```

Outputs are written to v2/outputs/.

## Structure

- v1/ original codebase
- v2/ updated codebase
- LICENSE

See v2/README.md for the new design and architecture.

## Paper

The updated paper lives in paper.md at the repo root and describes v2.

## V1 Archive and Changes

v1 is preserved as an archive under v1/.

Key changes in v2:

- Explicit delayed-observation wrapper to model asynchronicity.
- Blended observation path (delayed + predicted) for controller inputs.
- Shared divergence loss used to update both controller and predictor.
- Outputs saved into v2/outputs/.
- Validation and example scripts updated to match the async loop.
