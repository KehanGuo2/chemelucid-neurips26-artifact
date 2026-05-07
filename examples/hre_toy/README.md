# Sanitized Toy HRE Example

This directory contains a deliberately tiny, non-benchmark HRE example for
review-time inspection. It demonstrates:

- the public task schema (`toy_public_task_CNMR.json`),
- the sanitized grader/HRE node schema (`toy_grader_CNMR.json`),
- trajectory-to-node alignment (`toy_trajectory.json`),
- invocation of L1, L2, and L3 diagnostic code (`run_toy_scoring.py`).

The files are toy examples only. They are not part of the official benchmark
scope and do not expose complete chemistry-specific HRE templates, private
grader labels, withheld probes, or expert-curated reasoning graphs.
