# Benchmark Results - Level 1 - 5-Arm Spiral

| Model                     |   Params |   Test Acc (%) |   Compression Ratio | Compression   | vs Deep Reference (pp)   | vs Shallow Control (pp)   |
|:--------------------------|---------:|---------------:|--------------------:|:--------------|:-------------------------|:--------------------------|
| Deep-Unshared             |    25477 |           95.3 |             1       | 1.00x         | +0.00 pp                 | +9.60 pp                  |
| Shallow                   |     4677 |           85.7 |             5.4473  | 5.45x         | -9.60 pp                 | +0.00 pp                  |
| Shared-Single             |     4677 |           99.6 |             5.4473  | 5.45x         | +4.30 pp                 | +13.90 pp                 |
| Shared-2-Cyclic           |     8837 |           99.9 |             2.88299 | 2.88x         | +4.60 pp                 | +14.20 pp                 |
| Shared-3-Cyclic           |    12997 |          100   |             1.96022 | 1.96x         | +4.70 pp                 | +14.30 pp                 |
| Internal-Router-Soft      |    13192 |           99.7 |             1.93125 | 1.93x         | +4.40 pp                 | +14.00 pp                 |
| Internal-Router-Hard      |    13192 |           99.5 |             1.93125 | 1.93x         | +4.20 pp                 | +13.80 pp                 |
| External-Controller-Soft  |    13351 |           97.2 |             1.90825 | 1.91x         | +1.90 pp                 | +11.50 pp                 |
| External-Controller-Hard  |    13351 |           99.7 |             1.90825 | 1.91x         | +4.40 pp                 | +14.00 pp                 |
| Adaptive-Depth-Controller |     4827 |           99.6 |             5.27802 | 5.28x         | +4.30 pp                 | +13.90 pp                 |
