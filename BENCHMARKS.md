# RocketSim-CUDA Benchmark Report

Generated at: 2026-10-03T04:44:10Z

## Asynchronous SPS Throughput & Latency

| Environments |  Step Latency (ms) |   Throughput (SPS) | Pool VRAM (MB) |
|--------------|--------------------|--------------------|----------------|
|        4,096 |             0.0224 |        182,931,882 |           3.11 |
|       16,384 |             0.0457 |        358,815,795 |          12.45 |
|       32,768 |             0.0679 |        482,761,983 |          24.91 |
|       65,536 |             0.1472 |        445,098,962 |          49.81 |

### Methodology & Invariants
- **Timing Mechanism:** Asynchronous GPU hardware timing via `cudaEventRecord` / `cudaEventElapsedTime`.
- **Zero-Copy Pipeline:** Pure on-device tensor execution with zero host-device PCIe round trips.
- **Memory Management:** Monolithic pre-allocated VRAM memory pool (0-byte dynamic allocation delta).
- **Precision:** IEEE-754 strict compliance (`--fmad=false --prec-div=true --prec-sqrt=true -ftz=false`).
