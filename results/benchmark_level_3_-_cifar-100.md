# Benchmark Results - Level 3 - CIFAR-100

| Model                     |   Params |   Test Acc (%) |   Compression Ratio | Compression   | vs Deep Reference (pp)   | vs Shallow Control (pp)   |
|:--------------------------|---------:|---------------:|--------------------:|:--------------|:-------------------------|:--------------------------|
| Deep-Unshared             |   256276 |          33.32 |             1       | 1.00x         | +0.00 pp                 | +8.74 pp                  |
| Shallow                   |    47956 |          24.58 |             5.34398 | 5.34x         | -8.74 pp                 | +0.00 pp                  |
| Shared-Single             |    47956 |          11.63 |             5.34398 | 5.34x         | -21.69 pp                | -12.95 pp                 |
| Shared-2-Cyclic           |    89620 |          11.23 |             2.85958 | 2.86x         | -22.09 pp                | -13.35 pp                 |
| Shared-3-Cyclic           |   131284 |          15.01 |             1.95207 | 1.95x         | -18.31 pp                | -9.57 pp                  |
| Internal-Router-Soft      |   131431 |          14.84 |             1.94989 | 1.95x         | -18.48 pp                | -9.74 pp                  |
| Internal-Router-Hard      |   131431 |          15.42 |             1.94989 | 1.95x         | -17.90 pp                | -9.16 pp                  |
| External-Controller-Soft  |   144806 |          15.56 |             1.76979 | 1.77x         | -17.76 pp                | -9.02 pp                  |
| External-Controller-Hard  |   144806 |          13.44 |             1.76979 | 1.77x         | -19.88 pp                | -11.14 pp                 |
| Adaptive-Depth-Controller |    60698 |          10.22 |             4.22215 | 4.22x         | -23.10 pp                | -14.36 pp                 |
