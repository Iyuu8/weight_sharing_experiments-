# Benchmark Results - Level 2 - 4x4 Checkerboard

| Model                     |   Params |   Test Acc (%) |   Compression Ratio | Compression   | vs Deep Reference (pp)   | vs Shallow Control (pp)   |
|:--------------------------|---------:|---------------:|--------------------:|:--------------|:-------------------------|:--------------------------|
| Deep-Unshared             |    25282 |           95.4 |             1       | 1.00x         | +0.00 pp                 | +3.70 pp                  |
| Shallow                   |     4482 |           91.7 |             5.64079 | 5.64x         | -3.70 pp                 | +0.00 pp                  |
| Shared-Single             |     4482 |           93.1 |             5.64079 | 5.64x         | -2.30 pp                 | +1.40 pp                  |
| Shared-2-Cyclic           |     8642 |           94.8 |             2.92548 | 2.93x         | -0.60 pp                 | +3.10 pp                  |
| Shared-3-Cyclic           |    12802 |           94.3 |             1.97485 | 1.97x         | -1.10 pp                 | +2.60 pp                  |
| Internal-Router-Soft      |    12997 |           94.7 |             1.94522 | 1.95x         | -0.70 pp                 | +3.00 pp                  |
| Internal-Router-Hard      |    12997 |           94.4 |             1.94522 | 1.95x         | -1.00 pp                 | +2.70 pp                  |
| External-Controller-Soft  |    13156 |           94.1 |             1.92171 | 1.92x         | -1.30 pp                 | +2.40 pp                  |
| External-Controller-Hard  |    13156 |           92.8 |             1.92171 | 1.92x         | -2.60 pp                 | +1.10 pp                  |
| Adaptive-Depth-Controller |     4632 |           93.3 |             5.45812 | 5.46x         | -2.10 pp                 | +1.60 pp                  |
