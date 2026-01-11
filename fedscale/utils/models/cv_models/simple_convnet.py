"""
    FedBuff CNN for CIFAR-10, implemented in PyTorch.
"""

__all__ = ['FedBuffCIFAR10', 'fedbuff_cifar10']

import os
import torch.nn as nn
import torch.nn.init as init


class FedBuffBlock(nn.Module):
    """
    FedBuff specific convolution block consisting of Conv -> GroupNorm -> ReLU -> MaxPool -> Dropout.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(FedBuffBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=2)
        self.gn = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.gn(x)
        x = self.activ(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class FedBuffCIFAR10(nn.Module):
    """
    FedBuff-CNN model for CIFAR-10.

    Parameters:
    ----------
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """

    def __init__(self,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(FedBuffCIFAR10, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        # Block 1
        self.block1 = FedBuffBlock(
            in_channels=in_channels,
            out_channels=32)
        
        # Block 2
        self.block2 = FedBuffBlock(
            in_channels=32,
            out_channels=32)

        # Block 3
        self.block3 = FedBuffBlock(
            in_channels=32,
            out_channels=32)

        # Block 4
        self.block4 = FedBuffBlock(
            in_channels=32,
            out_channels=32)

        # Final Classifier
        # Input calculation: 32x32 -> (pad2) 34 -> (pool) 17 -> (pad2) 19 -> (pool) 9 
        # -> (pad2) 11 -> (pool) 5 -> (pad2) 7 -> (pool) 3.
        # Feature map size: 32 channels * 3 * 3
        self.fc = nn.Linear(
            in_features=32 * 3 * 3,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        # Ensure input is 4D (N, C, H, W)
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_fedbuff_cifar10(model_name=None,
                        pretrained=False,
                        root=os.path.join("~", ".torch", "models"),
                        **kwargs):
    """
    Create FedBuff-CNN model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = FedBuffCIFAR10(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError(
                "Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def fedbuff_cifar10(**kwargs):
    """
    FedBuff-CNN model for CIFAR-10 from 'FedBuff: High-Performance Federated Learning on Cloud'.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_fedbuff_cifar10(model_name="fedbuff_cifar10", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        fedbuff_cifar10,
    ]

    for model in models:
        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        
        # Verify output shape
        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()