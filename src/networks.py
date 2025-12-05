import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, layers, input_channels =1, bias=True, device=None):
        super().__init__()
        self.name = "ConvNet"
        self.bias = bias
        self.layers_config = layers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_channels = input_channels
        conv_blocks = []
        for num_filters, fsize in layers[1:]:
            block = nn.Sequential(
                nn.Conv2d(in_channels, num_filters, fsize, stride=1, padding=1, bias=bias),
                nn.BatchNorm2d(num_filters, affine=True, track_running_stats=False),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            conv_blocks.append(block)
            in_channels = num_filters

        self.layers = nn.Sequential(*conv_blocks)
        self.description = f'ConvNet: {self.layers}'
        self.to(self.device)

    def forward(self, X):
        if X.ndim == 3:
            X = X.unsqueeze(0)
        elif X.ndim != 4:
            raise ValueError(f"Expected input of shape (N,H,W,C), got {X.shape}")
        # Convert (N,H,W,C) → (N,C,H,W)
        #if X.shape[1] != 1:  # if channel not first
            #X = X.permute(0, 3, 1, 2).contiguous().to(self.device)
        X = X.to(self.device)
        out = self.layers(X)
        return out.flatten(1)

