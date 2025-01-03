import torch
import torch.nn as nn

class ChannelAttention(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//16, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(channels//16, channels, 1, 1, 0, bias=True)
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out=self.fc2(self.relu(self.fc1(self.avg(x))))
        max_out=self.fc2(self.relu(self.fc1(self.max(x))))

        out=avg_out+max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class myCBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention(kernel_size)
        self.relu= nn.ReLU()

    def forward(self, x):
        out=self.ca(x)*x
        out=self.sa(out)*out
        out=self.relu(out)
        return out
    
