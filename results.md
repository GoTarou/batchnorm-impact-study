\nRunning experiment: baseline_mlp
Epoch 1/2 | Train Loss: 1.8651 | Train Acc: 0.3880 | Val Loss: 1.2548 | Val Acc: 0.5600
Epoch 2/2 | Train Loss: 0.9754 | Train Acc: 0.6430 | Val Loss: 0.9153 | Val Acc: 0.6080
\nRunning experiment: batchnorm_mlp
Epoch 1/2 | Train Loss: 1.4804 | Train Acc: 0.6180 | Val Loss: 1.4628 | Val Acc: 0.6700
Epoch 2/2 | Train Loss: 0.9315 | Train Acc: 0.8010 | Val Loss: 1.0222 | Val Acc: 0.7400
\nRunning experiment: dropout_mlp
Epoch 1/2 | Train Loss: 2.0648 | Train Acc: 0.3030 | Val Loss: 1.5178 | Val Acc: 0.5100
Epoch 2/2 | Train Loss: 1.3836 | Train Acc: 0.5180 | Val Loss: 1.0795 | Val Acc: 0.6080
\nRunning experiment: batchnorm_dropout_mlp
Epoch 1/2 | Train Loss: 1.8826 | Train Acc: 0.4270 | Val Loss: 1.6057 | Val Acc: 0.5880
Epoch 2/2 | Train Loss: 1.3394 | Train Acc: 0.6750 | Val Loss: 1.1458 | Val Acc: 0.6780
\nRunning experiment: baseline_high_lr
Epoch 1/2 | Train Loss: 1.7167 | Train Acc: 0.3880 | Val Loss: 1.2136 | Val Acc: 0.5620
Epoch 2/2 | Train Loss: 0.9250 | Train Acc: 0.6410 | Val Loss: 0.8348 | Val Acc: 0.6940
\nRunning experiment: batchnorm_high_lr
Epoch 1/2 | Train Loss: 1.2363 | Train Acc: 0.6000 | Val Loss: 1.1168 | Val Acc: 0.6140
Epoch 2/2 | Train Loss: 0.6567 | Train Acc: 0.7560 | Val Loss: 0.8225 | Val Acc: 0.7000
\nFinal Results

--------------------------------------------------------------------------------

baseline_mlp: Acc=0.6680, Precision=0.7086, Recall=0.6680, F1=0.6339, Best Val Acc=0.6080, Epochs=2
batchnorm_mlp: Acc=0.7700, Precision=0.7685, Recall=0.7700, F1=0.7610, Best Val Acc=0.7400, Epochs=2
dropout_mlp: Acc=0.6340, Precision=0.6662, Recall=0.6340, F1=0.5989, Best Val Acc=0.6080, Epochs=2
batchnorm_dropout_mlp: Acc=0.7300, Precision=0.7283, Recall=0.7300, F1=0.7102, Best Val Acc=0.6780, Epochs=2
baseline_high_lr: Acc=0.7420, Precision=0.7489, Recall=0.7420, F1=0.7244, Best Val Acc=0.6940, Epochs=2
batchnorm_high_lr: Acc=0.7200, Precision=0.7501, Recall=0.7200, F1=0.7182, Best Val Acc=0.7000, Epochs=2
