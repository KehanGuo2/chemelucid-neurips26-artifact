# ChemElucid-Gym reviewer harness.
#
# Goals:
#   - reproducible verification environment for E&D reviewers
#   - no API keys required for any in-container command
#   - small enough to build in a few minutes on a laptop
#
# What the container can do (no API):
#   - python -m interactive.cli list-molecules
#   - python -m interactive.experiments.smoke_test
#   - python -m interactive.grade --episode-log <cached> --manifest core --out <out>
#   - python -m interactive.task_bundle ingest / validate / validate-rubric / validate-dag
#   - pytest interactive/tests/
#
# What the container intentionally does NOT do:
#   - live LLM/API episodes (would require provider keys; out of scope)
#   - serve a web UI or leaderboard
#   - bundle py2opsin / Java (IUPAC->SMILES is not on the verification path)

FROM python:3.11-slim-bookworm

# RDKit wheels bundle their own boost/cairo libs but still need a few
# system shared libraries at runtime; build-essential is installed only
# transiently for any source-builds that fall back from wheel.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libxrender1 \
       libxext6 \
       libsm6 \
       libgomp1 \
       libfreetype6 \
       libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy package metadata first so dependency layers cache across source edits.
COPY pyproject.toml requirements.txt README.md LICENSE ./

# Install only the dependencies needed for the no-API verification path:
# rdkit (chemistry), pytest (tests). py2opsin (IUPAC->SMILES + JRE) is
# explicitly omitted to keep the image small; smoke-test, grader, and
# task-bundle CLIs do not exercise it.
RUN pip install --upgrade pip \
    && pip install \
        "openai>=1.0.0" \
        "anthropic>=0.20.0" \
        "numpy>=1.21.0" \
        "matplotlib>=3.5.0" \
        "scipy>=1.8.0" \
        "rdkit>=2024.3.1" \
        "pytest>=7.0.0" \
        "PyYAML>=6.0"

# Copy the package and the immutable artefacts the verification path needs.
COPY interactive/ ./interactive/
COPY chem_defense/ ./chem_defense/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY data_split/ ./data_split/
COPY manifests/ ./manifests/
COPY analysis/ ./analysis/
COPY examples/ ./examples/
COPY docs/ ./docs/

# Editable install so any subsequent bind-mount on /workspace remains live.
RUN pip install --no-deps -e .

# A non-root user keeps file ownership tidy when /workspace is bind-mounted.
RUN useradd --create-home --uid 1000 reviewer \
    && chown -R reviewer:reviewer /workspace
USER reviewer

# By default, print the reviewer fingerprint: prove no API key is needed for
# the canonical "does the package import" check.
CMD ["python", "-m", "interactive.cli", "list-molecules"]
