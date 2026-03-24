## Final Results

| Model | Accuracy | Precision | Recall | F1-score | Best Val Acc | Epochs |
|------|---------|-----------|--------|---------|--------------|--------|
| Baseline MLP | 0.6680 | 0.7086 | 0.6680 | 0.6339 | 0.6080 | 2 |
| BatchNorm MLP | 0.7700 | 0.7685 | 0.7700 | 0.7610 | 0.7400 | 2 |
| Dropout MLP | 0.6340 | 0.6662 | 0.6340 | 0.5989 | 0.6080 | 2 |
| BatchNorm + Dropout MLP | 0.7300 | 0.7283 | 0.7300 | 0.7102 | 0.6780 | 2 |
| Baseline (High LR) | 0.7420 | 0.7489 | 0.7420 | 0.7244 | 0.6940 | 2 |
| BatchNorm (High LR) | 0.7200 | 0.7501 | 0.7200 | 0.7182 | 0.7000 | 2 |
