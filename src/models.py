import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, use_batchnorm: bool = False, use_dropout: bool = False, dropout_p: float = 0.3):
        super().__init__()

        layers = []

        def add_block(in_features, out_features):
            layers.append(nn.Linear(in_features, out_features))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))

        add_block(28 * 28, 512)
        add_block(512,256)
        add_block(256,128)
        add_block(128,64) 
        add_block(64,32)      
        layers.append(nn.Linear(32, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
