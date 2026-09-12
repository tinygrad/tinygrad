# Associative scan progress for #3039

This fork contains an initial tensor-only inclusive associative scan implementation using recursive doubling.

Current scope:
- arbitrary associative binary Tensor combine function
- configurable scan axis
- reverse scans
- non-power-of-two lengths
- correctness coverage for add, multiply, maximum, axis handling, and reverse scans

The implementation intentionally avoids a sequential Python loop over elements; graph construction uses O(log n) combine stages.

Follow-up work after maintainer feedback may integrate the operation directly into the Tensor API and extend coverage to structured states needed by Mamba/state-space recurrences.
