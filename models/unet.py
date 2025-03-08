import os
import shutil
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
        x1 = self.up(x1)
        return self.conv(torch.cat([x1, x2], dim=1))

class UNetEnhanced(nn.Module):
    
    """
    The U-Net architecture, consisting of an encoder-decoder structure with skip connections.
    Unlike the vanilla implementation, the model allows to configure the number of layers and
    enable batch normalization.

    The model includes:
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
            in_channels=2**(num_layers-2) * 64,
            out_channels=2**(num_layers-1) * 64,
            batch_norm=batch_norm)

        # Create the upsampling layers
        for i in range(num_layers-1, 0, -1): #[4, 3, 2, 1]
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
        
        # Last layer
        x = self.last_layer(x)

        # Forward pass through the upsampling layers with skip connections
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x, convs[-(i + 1)])

        # Final output layer
        return self.output(x)


class UNetVanilla(nn.Module):

    """
    The vanilla U-Net model for image segmentation.

    This model consists of:
    - A contracting path (encoder) with downsampling and feature extraction.
    - A bottleneck layer for deep feature representation.
    - An expansive path (decoder) for upsampling and feature reconstruction.
    - A final output layer to generate segmentation maps.

    Parameters:
    -----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    num_classes : int
        Number of output classes for segmentation.

    Example Usage:
    --------------
    model = UNetStandard(in_channels=3, num_classes=1)
    """   

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownConvert(in_channels, 64)
        self.down_convolution_2 = DownConvert(64, 128)
        self.down_convolution_3 = DownConvert(128, 256)
        self.down_convolution_4 = DownConvert(256, 512)

        self.bottle_neck = DoubleConv2D(512, 1024)

        self.up_convolution_1 = UpConvert(1024, 512)
        self.up_convolution_2 = UpConvert(512, 256)
        self.up_convolution_3 = UpConvert(256, 128)
        self.up_convolution_4 = UpConvert(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        """
        Forward pass of the U-Net model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes, height, width).
        """

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out

def remove_read_only_attribute(path):
    
    """
    Removes the read-only attribute from a file or directory.

    This function ensures that all files and subdirectories within a given 
    path have their permissions set to be writable.

    Args:
        path (str): The path to the file or directory.
    """

    if os.path.exists(path):
        os.chmod(path, 0o777)  # Make sure the file is writable
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files + dirs:
                    filepath = os.path.join(root, name)
                    os.chmod(filepath, 0o777)
        else:
            os.chmod(path, 0o777)

def clone_and_move_repo():
    
    """
    Clones the pretrained UNet repository if not already cloned and moves the required files.

    If the `unet_mberkay0.py` model file does not exist in the `models/` directory, 
    this function will:
    - Clone the GitHub repository containing the pretrained backbones.
    - Move the required model files to `models/`.
    - Remove the cloned repository after copying the necessary files.
    """

    repo_url = "https://github.com/mberkay0/pretrained-backbones-unet.git"
    repo_name = "tmp"
    models_dir = "models"
    backbones_dir = os.path.join("backbones_unet")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # If already cloned, avoid re-downloading
    if not os.path.exists(backbones_dir):

        # Clone repository
        print(f"Cloning {repo_url}...")
        os.system(f"git clone {repo_url} {repo_name}")
        print("Repository cloned successfully.")        

        # Move `backbones_unet` folder to `models/`
        shutil.move(os.path.join(repo_name, "backbones_unet"), ".")
        shutil.move(os.path.join(backbones_dir, "model", "unet.py"), os.path.join(models_dir, "unet_mberkay0.py"))

        # Remove read-only attributes and delete the temp repo
        if os.path.exists(repo_name):
            remove_read_only_attribute(repo_name)
            shutil.rmtree(repo_name)

    try:
        from backbones_unet.__init__ import __available_models__
    except ImportError as e:
        print(f"Error importing __available_models__: {e}")
        __available_models__ = []
    
    # Remove backbones_unet
    if os.path.exists(backbones_dir):
        remove_read_only_attribute(backbones_dir)
        shutil.rmtree(backbones_dir)

    return __available_models__

def create_unet(
    model_type='standard',
    in_channels=3,
    num_classes=1,
    print_available_models=False,
    **kwargs):
    
    """
    Creates a UNet model based on the specified type.

    This function supports three types of UNet models:
    - 'enhanced': Instantiates `UNetEnhanced`.
        Kwargs: num_layers (default: 5), batch_norm (default=True)
    - 'vanilla': Instantiates `UNetVanilla`.
    - 'pretrained': Loads a pretrained UNet model from an external repository.
        https://github.com/mberkay0/pretrained-backbones-unet.git
        Kwargs (some): backbone (default: 'resnet50'), pretrained (default: True), encoder_freeze (default: False)
    If `model_type` is 'pretrained', it ensures the required model file is 
    available by cloning the repository if necessary.

    Args:
        model_type (str): The type of UNet model to create ('enhanced', 'vanilla', or 'pretrained').
        in_channels (int): The number of input channels for the model.
        num_classes (int): The number of output classes for segmentation.
        print_available_models (bool, optional): Whether to print available models when using 'pretrained'. Default is False.
        **kwargs: Additional parameters for model configuration.

    Returns:
        model: An instance of the selected UNet model.

    Raises:
        ValueError: If an invalid `model_type` is provided.
    """

    if model_type == "enhanced":
        model = UNetFlexible(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "vanilla":
        model = UNetStandard(
            in_channels=in_channels,
            num_classes=num_classes
        )
    
    elif model_type == "pretrained":
        # Clone and move repo files if not already done
        available_models = clone_and_move_repo()

        from models import unet_mberkay0
        #from models.__init__ import __available_models__
        
        # Print available models
        if print_available_models:
            print(f"Available backbones: {available_models}")
        
        # Create the U-Net model with pretrained backbones
        model = unet_mberkay0.Unet(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from 'flexible', 'standard', or 'pretrained'.")
    
    print(f"Model ``{model_type}`` created sucessfully!")

    return model