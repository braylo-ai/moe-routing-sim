**SIMULATED.** Synthetic router traces (Zipf global popularity × per-domain Dirichlet preference) and an
analytic cost model with ASSUMED hardware: {'d_model': 2048, 'd_ff_expert': 1024, 'bytes_per_act': 2, 'tflops_per_device': 400.0, 'link_GBps_per_device': 50.0, 'tokens_per_step_per_device': 512, 'label': 'ASSUMED hardware'}.

#### domain concentration 0.1 (strong request→expert locality)

| devices | policy | remote expert hits | remote devices / token·layer | cross-device KB / token | traffic vs naive | load imbalance (max/mean) | sim throughput vs naive |
|---|---|---|---|---|---|---|---|
| 4 | naive: contiguous experts, round-robin requests | 75.0% | 2.10 | 134.3 | 1.00× | 1.06 | 1.00× |
| 4 | co-activation placement, round-robin requests | 74.9% | 2.05 | 131.2 | 0.98× | 1.27 | 0.99× |
| 4 | affinity placement, round-robin requests | 75.7% | 1.98 | 126.7 | 0.94× | 1.12 | 1.03× |
| 4 | locality-aware: affinity placement + locality request routing | 55.4% | 1.71 | 109.4 | 0.81× | 1.12 | 1.08× |
| 4 | locality-aware + 2 hot-expert replicas/layer | 42.6% | 1.38 | 88.3 | 0.66× | 1.13 | 1.23× |
| 8 | naive: contiguous experts, round-robin requests | 87.5% | 2.97 | 190.2 | 1.00× | 1.21 | 1.00× |
| 8 | co-activation placement, round-robin requests | 87.7% | 2.98 | 190.8 | 1.00× | 1.29 | 1.14× |
| 8 | affinity placement, round-robin requests | 87.8% | 2.78 | 177.9 | 0.94× | 1.16 | 1.17× |
| 8 | locality-aware: affinity placement + locality request routing | 65.5% | 2.34 | 149.7 | 0.79× | 1.16 | 1.23× |
| 8 | locality-aware + 2 hot-expert replicas/layer | 50.3% | 1.80 | 115.2 | 0.61× | 1.23 | 1.52× |
| 16 | naive: contiguous experts, round-robin requests | 93.7% | 3.50 | 224.0 | 1.00× | 1.23 | 1.00× |
| 16 | co-activation placement, round-robin requests | 93.8% | 3.52 | 225.4 | 1.01× | 2.21 | 1.07× |
| 16 | affinity placement, round-robin requests | 93.6% | 3.30 | 211.1 | 0.94× | 1.46 | 0.87× |
| 16 | locality-aware: affinity placement + locality request routing | 81.9% | 2.96 | 189.3 | 0.85× | 1.46 | 0.88× |
| 16 | locality-aware + 2 hot-expert replicas/layer | 65.4% | 2.40 | 153.3 | 0.68× | 1.34 | 2.12× |

#### domain concentration 0.3 (medium request→expert locality)

| devices | policy | remote expert hits | remote devices / token·layer | cross-device KB / token | traffic vs naive | load imbalance (max/mean) | sim throughput vs naive |
|---|---|---|---|---|---|---|---|
| 4 | naive: contiguous experts, round-robin requests | 75.0% | 2.09 | 134.0 | 1.00× | 1.09 | 1.00× |
| 4 | co-activation placement, round-robin requests | 75.1% | 2.06 | 131.6 | 0.98× | 1.36 | 0.96× |
| 4 | affinity placement, round-robin requests | 75.3% | 2.04 | 130.4 | 0.97× | 1.05 | 1.04× |
| 4 | locality-aware: affinity placement + locality request routing | 61.5% | 1.84 | 117.7 | 0.88× | 1.05 | 1.04× |
| 4 | locality-aware + 2 hot-expert replicas/layer | 48.7% | 1.54 | 98.4 | 0.73× | 1.03 | 1.19× |
| 8 | naive: contiguous experts, round-robin requests | 87.5% | 2.96 | 189.3 | 1.00× | 1.29 | 1.00× |
| 8 | co-activation placement, round-robin requests | 87.5% | 2.97 | 190.2 | 1.01× | 1.41 | 1.12× |
| 8 | affinity placement, round-robin requests | 86.8% | 2.87 | 183.9 | 0.97× | 1.18 | 1.15× |
| 8 | locality-aware: affinity placement + locality request routing | 72.6% | 2.56 | 163.6 | 0.86× | 1.18 | 1.18× |
| 8 | locality-aware + 2 hot-expert replicas/layer | 57.6% | 2.06 | 131.6 | 0.70× | 1.06 | 1.49× |
| 16 | naive: contiguous experts, round-robin requests | 93.8% | 3.48 | 222.5 | 1.00× | 1.69 | 1.00× |
| 16 | co-activation placement, round-robin requests | 93.7% | 3.51 | 224.6 | 1.01× | 2.24 | 1.10× |
| 16 | affinity placement, round-robin requests | 93.6% | 3.38 | 216.2 | 0.97× | 1.29 | 0.91× |
| 16 | locality-aware: affinity placement + locality request routing | 86.0% | 3.14 | 201.3 | 0.90× | 1.29 | 0.92× |
| 16 | locality-aware + 2 hot-expert replicas/layer | 69.7% | 2.60 | 166.5 | 0.75× | 1.24 | 2.14× |

#### domain concentration 1.0 (weak request→expert locality)

| devices | policy | remote expert hits | remote devices / token·layer | cross-device KB / token | traffic vs naive | load imbalance (max/mean) | sim throughput vs naive |
|---|---|---|---|---|---|---|---|
| 4 | naive: contiguous experts, round-robin requests | 75.0% | 2.08 | 133.4 | 1.00× | 1.05 | 1.00× |
| 4 | co-activation placement, round-robin requests | 74.9% | 2.04 | 130.5 | 0.98× | 1.43 | 0.93× |
| 4 | affinity placement, round-robin requests | 75.5% | 2.07 | 132.8 | 1.00× | 1.04 | 1.00× |
| 4 | locality-aware: affinity placement + locality request routing | 67.4% | 1.95 | 124.8 | 0.94× | 1.04 | 0.99× |
| 4 | locality-aware + 2 hot-expert replicas/layer | 54.5% | 1.68 | 107.7 | 0.81× | 1.07 | 1.17× |
| 8 | naive: contiguous experts, round-robin requests | 87.5% | 2.95 | 189.1 | 1.00× | 1.08 | 1.00× |
| 8 | co-activation placement, round-robin requests | 87.5% | 2.96 | 189.4 | 1.00× | 1.49 | 1.07× |
| 8 | affinity placement, round-robin requests | 87.3% | 2.95 | 188.5 | 1.00× | 1.16 | 1.11× |
| 8 | locality-aware: affinity placement + locality request routing | 79.0% | 2.74 | 175.4 | 0.93× | 1.16 | 1.12× |
| 8 | locality-aware + 2 hot-expert replicas/layer | 63.8% | 2.26 | 144.5 | 0.76× | 1.08 | 1.46× |
| 16 | naive: contiguous experts, round-robin requests | 93.7% | 3.49 | 223.2 | 1.00× | 1.32 | 1.00× |
| 16 | co-activation placement, round-robin requests | 93.7% | 3.51 | 224.9 | 1.01× | 2.19 | 1.08× |
| 16 | affinity placement, round-robin requests | 93.9% | 3.44 | 220.4 | 0.99× | 1.40 | 0.88× |
| 16 | locality-aware: affinity placement + locality request routing | 89.7% | 3.31 | 211.6 | 0.95× | 1.40 | 0.89× |
| 16 | locality-aware + 2 hot-expert replicas/layer | 73.3% | 2.76 | 176.8 | 0.79× | 1.21 | 1.99× |
