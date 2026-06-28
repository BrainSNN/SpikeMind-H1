import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional, layer


class SpikeSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, tau: float = 2.0):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = layer.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.sn = neuron.LIFNode(tau=tau, detach_reset=True)
        self.fc2 = layer.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.sn(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class SpikeCBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 1, tau: float = 2.0):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.ca_fc1 = layer.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.ca_sn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.ca_fc2 = layer.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.ca_gate = nn.Sigmoid()

        padding = spatial_kernel // 2
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)
        self.sa_gate = nn.Sigmoid()

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)

        max_attn = self.ca_fc2(self.ca_sn(self.ca_fc1(max_pool)))
        avg_attn = self.ca_fc2(self.ca_sn(self.ca_fc1(avg_pool)))
        x = x * self.ca_gate(max_attn + avg_attn)

        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa = self.sa_gate(self.sa_conv(torch.cat([sa_max, sa_avg], dim=1)))
        return x * sa


class SpikeGroupConvBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 2, downsample: bool = True, tau: float = 2.0):
        super().__init__()
        stride = (1, 2) if downsample else 1
        self.gconv1 = layer.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False)
        self.sn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.gconv2 = layer.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn = layer.BatchNorm2d(channels)

    def forward(self, x):
        x = self.gconv1(x)
        x = self.sn(x)
        x = self.gconv2(x)
        x = self.bn(x)
        return x


class SpikeResidualBlock(nn.Module):
    """
    Approximation of the paper's spike-based residual block:
    channel split + point-wise conv + depth-wise conv + spike activation + point-wise conv + residual fusion.
    """

    def __init__(self, channels: int, downsample: bool = True, tau: float = 2.0):
        super().__init__()
        assert channels % 2 == 0, "channels must be even for channel split"
        half = channels // 2
        stride = (1, 2) if downsample else 1

        self.branch = nn.Sequential(
            layer.Conv2d(half, half, kernel_size=1, bias=False),
            layer.BatchNorm2d(half),
            layer.Conv2d(
                half,
                half,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=half,
                bias=False,
            ),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Conv2d(half, half, kernel_size=1, bias=False),
            layer.BatchNorm2d(half),
        )
        self.downsample = downsample

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        y = self.branch(x2)
        if self.downsample:
            x1 = F.avg_pool2d(x1, kernel_size=(1, 2), stride=(1, 2), ceil_mode=False)
            x2 = F.avg_pool2d(x2, kernel_size=(1, 2), stride=(1, 2), ceil_mode=False)
        return y + x1 + x2


class LSNNet(nn.Module):
    """
    A practical PyTorch + SpikingJelly implementation inspired by the paper.

    Expected input shape:
        [B, C, T]  -> e.g. [batch, 3 leads, 1500 samples]
    Internal shape:
        [B, 1, C, T]

    Notes:
    - The paper describes the macro-architecture clearly, but some low-level implementation
      details are not fully specified in the PDF. This implementation follows the published
      structure while keeping key hyperparameters configurable.
    - In SpikingJelly training, the same sample is unfolded over `sim_steps` to let LIF neurons
      integrate dynamics across simulation steps.
    """

    def __init__(
        self,
        in_leads: int = 3,
        num_classes: int = 2,
        filters: int = 16,
        sim_steps: int = 8,
        dropout: float = 0.1,
        tau: float = 2.0,
        cbam_spatial_kernel: int = 1,
    ):
        super().__init__()
        self.in_leads = in_leads
        self.num_classes = num_classes
        self.filters = filters
        self.sim_steps = sim_steps

        self.stem = nn.Sequential(
            layer.Conv2d(1, filters, kernel_size=3, padding=1, bias=False),
            layer.MaxPool2d(kernel_size=3, stride=1, padding=1),
            layer.Dropout(dropout),
        )

        self.block1 = SpikeResidualBlock(filters, downsample=True, tau=tau)
        self.se = SpikeSEBlock(filters//2, reduction=4, tau=tau)
        self.block2 = SpikeGroupConvBlock(filters//2, groups=2, downsample=True, tau=tau)
        self.block3 = SpikeResidualBlock(filters//2, downsample=True, tau=tau)
        self.block4 = SpikeGroupConvBlock(filters//4, groups=2, downsample=True, tau=tau)
        self.cbam = SpikeCBAM(filters//4, reduction=4, spatial_kernel=cbam_spatial_kernel, tau=tau)

        self.head_conv = layer.Conv2d(filters//4, filters//4, kernel_size=1, bias=False)
        self.head_bn = layer.BatchNorm2d(filters//4)
        self.head_sn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.head_pool = layer.AdaptiveAvgPool2d((1, 1))
        self.head_dropout = layer.Dropout(dropout)
        self.fc = layer.Linear(filters//4, num_classes, bias=True)

    def _forward_single_step(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.se(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.cbam(x)
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_sn(x)
        x = self.head_pool(x)
        x = self.head_dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_leads:
            raise ValueError(f"Expected {self.in_leads} EEG leads, got {x.shape[1]}")

        x = x.unsqueeze(1)  # [B, 1, C, T]
        logits_seq = []
        for _ in range(self.sim_steps):
            logits_seq.append(self._forward_single_step(x))
        logits_seq = torch.stack(logits_seq, dim=0)  # [S, B, num_classes]
        return logits_seq.mean(0), logits_seq

    # def reset(self):
    #     functional.reset_net(self)

class CNN(nn.Module):
    def __init__(self, in_chans,n_class):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(in_chans,64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,256,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256,n_class)
    
    def forward(self,x):
        x = self.CNN(x)
        x = x.view(-1,256)
        return self.classifier(x),1


if __name__ == "__main__":
    model = LSNNet(in_leads=3, num_classes=2, filters=16, sim_steps=8)
    x = torch.randn(4, 3, 1500)
    y, y_seq = model(x)
    print("logits:", y.shape)
    print("logits_seq:", y_seq.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
