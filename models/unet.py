import torch
from torch import nn

class DoubleConv2D(nn.Module):
    
    """
    A class to define a block of two consecutive convolutional layers followed by
    batch normalization (optional) and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        
        # Define the double convolution block with optional batch normalization
        if batch_norm:
            self.double_conv2d = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv2d = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):

        # Forward pass through the double convolution block
        return self.double_conv2d(x)
    

class DownConvert(nn.Module):

    """
    A class for the downsampling part of the U-Net, consisting of:
    1. Double convolution for feature extraction
    2. Max pooling for downsampling the spatial dimensions
    """

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        
        # Initialize the double convolution block followed by max pooling
        self.conv = DoubleConv2D(in_channels, out_channels, batch_norm)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        # Pass through the convolutional block and downsampling layer
        conv = self.conv(x)
        down = self.down(conv)
        return conv, down
    
class UpConvert(nn.Module):
    
    """
    A class for the upsampling part of the U-Net, consisting of:
    1. Transposed convolution for upsampling the spatial dimensions
    2. A double convolution block for feature refinement
    """

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        
        # Initialize the transposed convolution (upsampling) and a double convolution block
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Upsampling
        self.conv = DoubleConv2D(in_channels, out_channels, batch_norm)

    def forward(self, x1, x2):

        # Perform upsampling and concatenate with the corresponding downsampled features
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))

class UNet(nn.Module):
    
    """
    The main U-Net architecture, consisting of an encoder-decoder structure with skip connections.
    It includes:
    1. Downsampling (encoder) part to capture context.
    2. Upsampling (decoder) part to recover spatial resolution.
    3. Skip connections to preserve high-resolution features from the encoder.
    """

    def __init__(self, in_channels, num_classes, num_layers=5, batch_norm=True):
        super().__init__()

        # Initialization
        self.num_layers = num_layers
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Initial downsampling layer (64 channels)
        self.down_convs.append(DownConvert(
            in_channels=in_channels,
            out_channels=64,
            batch_norm=batch_norm))

        # Create the downsampling layers
        for i in range(1, num_layers - 1):
            self.down_convs.append(DownConvert(
                in_channels=2**(i-1) * 64,
                out_channels=2**i * 64,
                batch_norm=batch_norm))

        # Last downsampling layer (without max pooling, only convolution)
        self.last_layer = DoubleConv2D(
            in_channels=2**(num_layers-1) * 64,
            out_channels=2**num_layers * 64,
            batch_norm=batch_norm)

        # Create the upsampling layers
        for i in range(num_layers-1, 0, -1):
            self.up_convs.append(UpConvert(
                in_channels=2**i * 64,
                out_channels=2**(i-1) * 64,
                batch_norm=batch_norm))

        # Output layer to match the number of classes
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        """
        Forward pass through the U-Net:
        1. Pass through downsampling layers (encoder).
        2. Pass through upsampling layers (decoder) with skip connections.
        3. Output the final result using a 1x1 convolution to match `num_classes`.
        """
        
        # To store intermediate feature maps for skip connections
        convs = []

        # Forward pass through the downsampling layers
        for down in self.down_convs:
            c, x = down(x)
            convs.append(c)
        
        # Forward pass through the upsampling layers with skip connections
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x, convs[-(i + 1)])

        # Final output layer
        return self.output(x)


    