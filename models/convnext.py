# convnext.py - ConvNeXt model definitions
# Part of the torchsuite library

# ------------------------------------------------------------------------------
# Original implementation by Meta Platforms, Inc. (Facebook AI Research)
# Source: https://github.com/facebookresearch/ConvNeXt
# Licensed under the MIT License (see LICENSE.meta or LICENSE file in the source repo)
#
# We adapted and integrated parts of the original code in this module.
# All credit for the original architecture and pre-trained weights goes to the original authors.
# ------------------------------------------------------------------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class LayerNorm(nn.Module):

    """
    LayerNorm module that supports two data formats: 'channels_last' (default) and 'channels_first'.

    This module normalizes the input tensor over the specified dimensions using learnable 
    scale (weight) and shift (bias) parameters. It supports both common layout conventions:

    - 'channels_last': input shape is (N, H, W, C)
    - 'channels_first': input shape is (N, C, H, W)

    Args:
        normalized_shape (int): Number of features (channels) to normalize.
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6
        data_format (str): One of ['channels_last', 'channels_first']. Default: 'channels_last'
    """

    def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            data_format="channels_last"
            ):
        super().__init__()
        
        # Learnable scaling and shifting parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        self.eps = eps
        self.data_format = data_format
        
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("Only 'channels_last' and 'channels_first' are supported.")

        # For channels_last, PyTorch expects the shape as a tuple
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):

        """
        Apply layer normalization to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (N, H, W, C) for 'channels_last' 
                        or (N, C, H, W) for 'channels_first'

        Returns:
            Tensor: Normalized output tensor with the same shape as input
        """

        if self.data_format == "channels_last":
            # Use PyTorch's built-in layer_norm for channels_last
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Manually compute mean and variance over the channel dimension
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):

    """
    ConvNeXt Block.

    This block is inspired by the ConvNeXt architecture and supports two equivalent 
    formulations of the block structure. It uses the (2) variant internally, which 
    has been found to be slightly faster in PyTorch:

    (1) DepthwiseConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DepthwiseConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input and output channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Initial value for layer scaling (gamma). Default: 1e-6
    """

    def __init__(
            self,
            dim,
            drop_path=0.,
            layer_scale_init_value=1e-6
            ):
        super().__init__()

        # Depthwise convolution: operates independently on each channel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Layer normalization in channels-last format (N, H, W, C)
        self.norm = LayerNorm(dim, eps=1e-6)

        # First pointwise (1x1) convolution, implemented as a linear layer
        self.pwconv1 = nn.Linear(dim, 4 * dim)

        # Non-linear activation
        self.act = nn.GELU()

        # Second pointwise (1x1) convolution, reducing back to original dimension
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Learnable layer scaling parameter (gamma)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        # DropPath for stochastic depth regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the ConvNeXt block.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Output tensor of shape (N, C, H, W)
        """
        input = x

        # Apply depthwise convolution
        x = self.dwconv(x)

        # Change format to channels-last for layer normalization and linear layers
        x = x.permute(0, 2, 3, 1)

        # Normalize, apply linear transformations and activation
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Apply layer scaling if enabled
        if self.gamma is not None:
            x = self.gamma * x

        # Revert to channels-first format
        x = x.permute(0, 3, 1, 2)

        # Apply residual connection with optional stochastic depth
        x = input + self.drop_path(x)

        return x


class ConvNeXt(nn.Module):

    """
    ConvNeXt Model

    A modernized convolutional neural network architecture inspired by transformer designs,
    as proposed in "A ConvNet for the 2020s" (https://arxiv.org/pdf/2201.03545.pdf).

    This model follows a hierarchical structure with multiple downsampling stages and residual
    blocks (ConvNeXt Blocks), incorporating design principles like depthwise convolutions, 
    LayerNorm, and GELU activations.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        output_dim (int): Number of output classes for classification or ouput scores for regression. Default: 1000
        depths (list of int): Number of ConvNeXt blocks at each stage. Default: [3, 3, 9, 3]
        dims (list of int): Number of channels (feature dimensions) at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Maximum stochastic depth rate for residual blocks. Default: 0.0
        layer_scale_init_value (float): Initial value for learnable layer scaling. Default: 1e-6
        dropout (float): Dropout rate for regularization. Default: 0.3
        head_init_scale (float): Scaling factor for the classifier head weights and biases. Default: 1.0
        mlp_hidden_dim (int): If specified, adds a hidden layer with GELU and dropout. Default: None
    """

    def __init__(self, in_chans=3, output_dim=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, 
                 head_init_scale=1., dropout=0.3,
                 mlp_hidden_dim=None):
        super().__init__()

        # Stem and downsampling layers to reduce spatial resolution and increase channel depth
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),  # Initial patch embedding
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # Add 3 intermediate downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages consist of stacked ConvNeXt Blocks with increasing channel dimensions
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Progressive drop path
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final normalization and classification/regression head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        if dropout is not None:
            if not (0.0 <= dropout <= 1.0):
                raise ValueError("dropout must be None or a float between 0 and 1")
        else:
            dropout = 0.0

        if mlp_hidden_dim:
            self.head = nn.Sequential(
                nn.Linear(dims[-1], mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout), 
                nn.Linear(mlp_hidden_dim, output_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(dims[-1], output_dim)
            )

        # Initialize model weights
        self.apply(self._init_weights)
        #self.head.weight.data.mul_(head_init_scale)
        #self.head.bias.data.mul_(head_init_scale)
        # Scale only the final classification layer
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        elif isinstance(self.head, nn.Sequential):
            final_linear = None
            for layer in reversed(self.head):
                if isinstance(layer, nn.Linear):
                    final_linear = layer
                    break
            if final_linear is not None:
                final_linear.weight.data.mul_(head_init_scale)
                final_linear.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):

        """
        Initialize Conv2D and Linear layers using truncated normal and zero bias.
        """

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):

        """
        Extract features from input using stem, downsampling, and ConvNeXt stages.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Global-pooled feature vector of shape (N, C)
        """

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])  # Global average pooling over spatial dimensions
        x = self.norm(x)
        return x

    def forward(self, x):

        """
        Forward pass of the full ConvNeXt model including classification head.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Output logits of shape (N, num_classes)
        """

        x = self.forward_features(x)
        x = self.head(x)
        return x


# Dictionary of pretrained model URLs
# These are used to load weights from ImageNet-1K or ImageNet-22K training
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# -------- ConvNeXt Tiny --------
# @register_model  # Uncomment this if using a model registry like in timm
def convnext_tiny(pretrained=False, freeze_backbone=False, **kwargs):
    
    """
    ConvNeXt-Tiny model builder.

    Args:
        pretrained (bool): If True, load pretrained weights.
        freeze_backbone (bool): If True, freeze all backbone layers (stages). Default: False
        kwargs: Additional arguments passed to ConvNeXt.
    """

    # Initialize model with tiny configuration
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    # Load pretrained weights if requested
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    
    # Optionally freeze backbone (all ConvNeXt stages)
    if freeze_backbone:      
        for stage in model.stages:
            for param in stage.parameters():
                param.requires_grad = False
    
    return model

# -------- ConvNeXt Small --------
@register_model
def convnext_small(pretrained=False, freeze_backbone=False, **kwargs):

    """
    ConvNeXt-Small model builder. Larger depth than Tiny, same dimensions.

    Args:
        pretrained (bool): Load pretrained weights.        
        freeze_backbone (bool): If True, freeze all backbone layers (stages). Default: False
        kwargs: Additional arguments passed to ConvNeXt.
    """

    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    # Optionally freeze backbone (all ConvNeXt stages)
    if freeze_backbone:      
        for stage in model.stages:
            for param in stage.parameters():
                param.requires_grad = False

    return model

# -------- ConvNeXt Base --------
@register_model
def convnext_base(pretrained=False, in_22k=False, freeze_backbone=False, **kwargs):

    """
    ConvNeXt-Base model builder.
    
    Args:
        pretrained (bool): Load pretrained weights.
        in_22k (bool): Load weights trained on ImageNet-22K instead of 1K.
        freeze_backbone (bool): If True, freeze all backbone layers (stages). Default: False
        kwargs: Additional arguments passed to ConvNeXt.
    """

    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

    if pretrained:
        # Choose URL based on pretraining dataset
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    # Optionally freeze backbone (all ConvNeXt stages)
    if freeze_backbone:      
        for stage in model.stages:
            for param in stage.parameters():
                param.requires_grad = False

    return model

# -------- ConvNeXt Large --------
@register_model
def convnext_large(pretrained=False, in_22k=False, freeze_backbone=False, **kwargs):

    """
    ConvNeXt-Large model builder.

    Args:
        pretrained (bool): Load pretrained weights.
        in_22k (bool): Load weights trained on ImageNet-22K instead of 1K.
        freeze_backbone (bool): If True, freeze all backbone layers (stages). Default: False
        kwargs: Additional arguments passed to ConvNeXt.
    """

    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        #model.load_state_dict(checkpoint["model"])
        state_dict = checkpoint['model']
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)
        model.load_state_dict(state_dict, strict=False)

    # Optionally freeze backbone (all ConvNeXt stages)
    if freeze_backbone:      
        for stage in model.stages:
            for param in stage.parameters():
                param.requires_grad = False

    return model

# -------- ConvNeXt XLarge --------
@register_model
def convnext_xlarge(pretrained=False, in_22k=False, freeze_backbone=False, **kwargs):
    
    """
    ConvNeXt-XLarge model builder.
    
    Only pretrained weights from ImageNet-22K are available for this model.

    Args:
        pretrained (bool): Load pretrained weights.
        in_22k (bool): Must be True. Asserts if False.
        freeze_backbone (bool): If True, freeze all backbone layers (stages). Default: False
        kwargs: Additional arguments passed to ConvNeXt.
    """

    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)

    if pretrained:
        # Only available for 22K
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    # Optionally freeze backbone (all ConvNeXt stages)
    if freeze_backbone:      
        for stage in model.stages:
            for param in stage.parameters():
                param.requires_grad = False

    return model