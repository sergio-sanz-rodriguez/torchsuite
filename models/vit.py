import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.nn.init import trunc_normal_
from .pretrained_models import build_pretrained_model
#, xavier_normal_, zeros_, orthogonal_, kaiming_normal_

class HyperspectralToRGB(nn.Module):

    """
    Simple hyperspectral-to-RGB projection using a single 3D convolution.

    Expects a hyperspectral cube (with spectral dimension represented as channels in a 3D tensor)
    and produces a 3-channel (RGB-like) output by learning a 3D convolutional projection.

    Args:
        None

    Returns:
        torch.Tensor: Output tensor after 3D convolution. Shape depends on the input shape and
            convolution settings, but typically follows the same spatial dimensions as the input
            with 3 output channels.
    """

    def __init__(self):

        """
        Initializes the HyperspectralToRGB module.        
        """

        super(HyperspectralToRGB, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=125, out_channels=3, kernel_size=(3, 3, 3), padding=1)
    
    def forward(self, x):

        """
        Forward pass for hyperspectral-to-RGB projection.

        Args:
            x (torch.Tensor): Hyperspectral input tensor expected by Conv3d.

        Returns:
            torch.Tensor: Projected tensor with 3 output channels.
        """

        return self.conv1(x) 

class SpatialConcatenation(nn.Module):

    """
    SpatialConcatenation performs a spatial reshaping of input feature maps
    to generate a 3-channel image-like tensor of a desired spatial dimension.

    The process involves:
      1. Reducing the number of input channels using a 1x1 convolution.
      2. Rearranging channels into spatial blocks to create a larger image.
      3. Optionally resizing the output spatially to match a target resolution.

    Args:
        in_channels (int): Number of input channels.
        spatial_dim (int): Target spatial size (height and width) of the output.
        row_blocks (int): Number of vertical (row) blocks used for spatial concatenation.
        column_blocks (int): Number of horizontal (column) blocks used for spatial concatenation.
        device (torch.device): Device to run the layer on.
    """
    
    def __init__(
        self,
        in_channels: int=125,
        spatial_dim: int=384,
        row_blocks: int=6,
        column_blocks: int=6,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        super(SpatialConcatenation, self).__init__()

        """
        Initializes the SpatialConcatenation module.

        Args:
            in_channels (int): Number of input channels. Defaults to 125.
            spatial_dim (int): Target output spatial resolution (height and width). Defaults to 384.
            row_blocks (int): Number of vertical blocks used for spatial concatenation. Defaults to 6.
            column_blocks (int): Number of horizontal blocks used for spatial concatenation. Defaults to 6.
            device (torch.device): Device to place learnable layers on. Defaults to CUDA if available else CPU.

        Returns:
            None
        """

        # Convolutional layer to reduce the number of channels to 108
        self.spatial_dim = spatial_dim
        self.row_blocks = row_blocks
        self.column_blocks = column_blocks
        out_channels = 3 * row_blocks * column_blocks

        # Expand the image by spatially concatenating channels
        self.convert_channels = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1).to(device)

        # Compute the number of layers needed to downsample to the target size
        self.num_halvings = int(torch.log2(torch.tensor(768 // spatial_dim, dtype=torch.float)).item())
    
        # Sequential block for downsampling the spatial size using Conv2d with stride=2
        layers = []
        for _ in range(self.num_halvings):
            layers.append(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(num_features=3))
            layers.append(nn.ReLU(inplace=True))

        self.downsampling = nn.Sequential(*layers)

        # Optional final adaptive pooling to adjust exactly to target size
        #final_output_size = 768 // (2 ** self.num_halvings)
        #if final_output_size != spatial_dim:
        self.adaptive_pool = nn.AdaptiveAvgPool2d((spatial_dim, spatial_dim))
        #else:
        #    self.adaptive_pool = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        """
        Forward pass that converts an input tensor into a 3-channel, image-like representation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 3, spatial_dim, spatial_dim].
        """
                
        # Step 1: Modify the channel dimension
        x = self.convert_channels(x)
        #x = x[:, :108, :, :]

        # Step 2: Reshape the tensor to (BATCH, 3, img_dim * row_blocks, img_dim * column_blocks)
        #x = x.view(x.size(0), 3, x.size(-2) * self.row_blocks, x.size(-1) * self.column_blocks)  
        x = x.reshape(x.size(0), 3, x.size(-2) * self.row_blocks, x.size(-1) * self.column_blocks)  
        
        # Step 3: Resample the tensor to target WxH
        x = F.interpolate(x, size=(self.spatial_dim, self.spatial_dim), mode='bilinear', align_corners=False)
        # Step 3: Apply CNN-based downsampling
        #x = self.downsampling(x)

        # Step 4: Final resizing (if necessary) using AdaptiveAvgPool2d
        #x = self.adaptive_pool(x)
        #x = self.pool(x)

        # === Visualization block (for the first sample in the batch only) ===
        #output_img = x[0].detach().cpu().clamp(0, 1)  # shape: (3, H, W)
        #img = TF.to_pil_image(output_img)
        #plt.imshow(img)
        #plt.title("Output Image from SpatialConcatenation")
        #plt.axis("off")
        #plt.show()
        
        return x
    
class HyperspectralConv3D(nn.Module):

    """
    Hyperspectral-to-RGB (or 3-channel) embedding using a learnable 3D convolution.

    Treats the spectral dimension as the depth dimension for Conv3d by reshaping an input
    hyperspectral image (B, C, H, W) into (B, 1, C, H, W), then applies a 3D convolution
    that typically spans (part of) the spectral axis to produce a 3-channel output.

    Args:
        in_channels (int): Number of input channels for Conv3d. Defaults to 1 (after unsqueeze).
        out_channels (int): Number of output channels (e.g., 3 for RGB-like). Defaults to 3.
        kernel_size (tuple): 3D convolution kernel size (D, H, W). Defaults to (125, 1, 1).
        stride (int): Convolution stride. Defaults to 1.
        padding (int): Convolution padding. Defaults to 0.
        bias (bool): Whether to use bias in convolution. Defaults to False.
        device (torch.device): Device to place learnable layers on. Defaults to CUDA if available else CPU.

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, out_channels, H, W].
    """

    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=3,
        kernel_size=(125, 1, 1),
        stride: int=1,
        padding: int=0, #(7 // 2, 0, 0),
        bias: bool=False,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
        ):

        """
        Initializes the HyperspectralConv3D module.

        Args:
            in_channels (int): Number of input channels for Conv3d. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 3.
            kernel_size (tuple): 3D kernel size (D, H, W). Defaults to (125, 1, 1).
            stride (int): Convolution stride. Defaults to 1.
            padding (int): Convolution padding. Defaults to 0.
            bias (bool): Whether to use bias. Defaults to False.
            device (torch.device): Device to place learnable layers on. Defaults to CUDA if available else CPU.

        Returns:
            None
        """

        super(HyperspectralConv3D, self).__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,  # input is (B, 1, C, H, W)
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, #(kernel_size[0] // 2, 0, 0),  # preserve spectral size
            bias=False
        ).to(device)
        
    def forward(self, x):

        """
        Forward pass for 3D convolutional hyperspectral embedding.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, C, H, W], where C is the spectral bands.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, H, W].
        """

        # Input x: (B, C, H, W)
        x = x.unsqueeze(1)  # -> (B, 1, C, H, W)
        #print(x.shape)
        x = self.conv3d(x)  # -> (B, 3, C, H, W)
        #print(x.shape)
        x = x.squeeze(2)    # -> (B, 3, H, W)
        #print(x.shape)
        return x


class SpectralViT(nn.Module):

    """
    Wrapper model that adapts hyperspectral inputs for a pretrained RGB ViT-style backbone.

    This module:
      1) Converts a hyperspectral tensor into a 3-channel image-like tensor using a chosen embedding algorithm,
      2) Feeds the resulting (B, 3, H, W) tensor into a pretrained vision model built by `build_pretrained_model`.

    Args:
        model (str): Name/identifier for the pretrained backbone to build. Defaults to 'vit_b_16_224'.
        embedding_alg (str): Hyperspectral-to-3ch embedding algorithm. Supported values:
            'spatial_concat' (SpatialConcatenation) or anything else (HyperspectralConv3D). Defaults to 'spatial_concat'.
        spatial_dim (int): Output spatial resolution used by SpatialConcatenation. Defaults to 384.
        block_size_spatial_emb (int): Number of row/column blocks for SpatialConcatenation. Defaults to 6.
        num_classes (int): Number of output classes (or regression outputs). Defaults to 1.
        dropout (float): Dropout used in the pretrained head (as supported by `build_pretrained_model`). Defaults to 0.0.
        kernel_size (tuple): Kernel size for HyperspectralConv3D. Defaults to (125, 1, 1).
        freeze_model (bool): If True, freezes the pretrained backbone parameters. Defaults to False.
        seed (int): Random seed forwarded to `build_pretrained_model`. Defaults to 42.
        device (torch.device): Device to run the model on. Defaults to CUDA if available else CPU.

    Returns:
        torch.Tensor: Model output of shape [batch_size, num_classes].
    """

    def __init__(
        self,
        model: str='vit_b_16_224',
        embedding_alg: str='spatial_concat',     
        spatial_dim: int=384,
        block_size_spatial_emb: int=6, 
        num_classes: int = 1,
        dropout: float=0.0,
        kernel_size: tuple=(125, 1, 1),
        freeze_model: bool=False,
        seed: int=42,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
        ):

        """
        Initializes the SpectralViT model.

        Args:
            model (str): Pretrained backbone identifier. Defaults to 'vit_b_16_224'.
            embedding_alg (str): Embedding strategy for hyperspectral inputs. Defaults to 'spatial_concat'.
            spatial_dim (int): Target spatial resolution for SpatialConcatenation. Defaults to 384.
            block_size_spatial_emb (int): Row/column blocks for SpatialConcatenation. Defaults to 6.
            num_classes (int): Number of output classes/targets. Defaults to 1.
            dropout (float): Dropout probability forwarded to the backbone builder. Defaults to 0.0.
            kernel_size (tuple): Kernel size for HyperspectralConv3D. Defaults to (125, 1, 1).
            freeze_model (bool): Whether to freeze the backbone parameters. Defaults to False.
            seed (int): Random seed forwarded to the backbone builder. Defaults to 42.
            device (torch.device): Device to run the model on. Defaults to CUDA if available else CPU.

        Returns:
            None
        """
        
        super().__init__()

        self.device = device
        
        if embedding_alg == 'spatial_concat':
            self.embedding = SpatialConcatenation(
                in_channels=kernel_size[0],
                spatial_dim=spatial_dim,
                row_blocks=block_size_spatial_emb,
                column_blocks=block_size_spatial_emb,
                device=device
                )
        else:
            self.embedding = HyperspectralConv3D(
                in_channels=1,
                out_channels=3,
                kernel_size=kernel_size,
                stride=1,
                padding = 0, #(kernel_size // 2, 0, 0),
                bias=False,
                device=device
                )

        self.model = build_pretrained_model(
            model=model,
            output_dim=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_model,
            seed=seed,
            device=device,
            )

    def forward(self, x):

        """
        Forward pass for SpectralViT.

        Args:
            x (torch.Tensor): Hyperspectral input tensor. Expected shape depends on `embedding_alg`:
                - For SpatialConcatenation: [batch_size, C, H, W]
                - For HyperspectralConv3D:  [batch_size, C, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes].
        """

        x = self.embedding(x)  # -> (B, 3, H, W)
        x = self.model(x)  # -> (B, num_classes)

        return x

# Implementation of a vision transformer following the paper "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"

class PatchEmbedding(nn.Module):

    """
    Turns a 2D input image into a 1D sequence of learnable patch embeddings, and
    adds special tokens + positional embeddings.

    This module:
      1) Projects an image into non-overlapping patches via a Conv2d (patchify + linear projection),
      2) Flattens patches into a token sequence,
      3) Prepends a learnable class token and appends a learnable distillation token,
      4) Adds learnable positional embeddings,
      5) Applies dropout to the final token sequence.

    Args:
        img_size (int): Input image resolution (assumes square images). Must be divisible by `patch_size`.
            Defaults to 224.
        in_channels (int): Number of input channels (e.g., 3 for RGB). Defaults to 3.
        patch_size (int): Patch size used to split the image into non-overlapping patches. Defaults to 16.
        emb_dim (int): Embedding dimension for each token. Defaults to 768.
        emb_dropout (float): Dropout probability applied after adding positional embeddings. Defaults to 0.1.

    Returns:
        torch.Tensor: Token embeddings of shape [batch_size, num_patches + 2, emb_dim],
            where the extra 2 tokens correspond to [class_token, distillation_token].
    """

    # Initialize the class with appropriate variables
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 emb_dim:int=768,
                 emb_dropout:float=0.1):
        super().__init__()

        # Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # Create a layer to turn an image into patches
        # from [batch, in_channels, height, width] to [batch, out_channels, H_patches, W_patches] with H_patches = H / patch_size and W_patches = W / patch_size
        self.conv_proj = nn.Conv2d(in_channels=in_channels,
                                   out_channels=emb_dim, # This defines the number of filters in the Conv2D
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        # from batch, out_channels, H_patches, W_patches] to [batch_size, emb_dim, H_patches * W_patches]
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector. dim=0 is the batch size, dim=1 is the embedding dimension.
                                  end_dim=3)
        
        # Token embedding class
        self.class_token = trunc_normal_(nn.Parameter(torch.zeros(1, 1, emb_dim), requires_grad=True), std=0.02)
                       
        # Position embedding class
        num_patches = (img_size * img_size) // patch_size**2
        self.pos_embedding = trunc_normal_(nn.Parameter(torch.zeros(1, num_patches+1, emb_dim), requires_grad=True), std=0.02)
        
        # Create embedding dropout value
        self.emb_dropout = nn.Dropout(p=emb_dropout)

    # Define the forward method
    def forward(self, x):
         
        # Linear projection of patches 
        x = self.conv_proj(x)

        # Flatten the linear transformed patches
        x = self.flatten(x)

        # Adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        x = x.permute(0, 2, 1)

        # Create class token and prepend
        class_token = self.class_token.expand(x.shape[0], -1, -1) # Expand to match with batch size
        x = torch.cat((class_token, x), dim=1)
        
        # Create position embedding
        x = x + self.pos_embedding

        # Run embedding dropout (Appendix B.1)              
        x = self.emb_dropout(x)

        return x


class MultiheadSelfAttentionBlock(nn.Module):

    """
    Multi-head self-attention (MSA) block with pre-normalization.

    Applies LayerNorm followed by `nn.MultiheadAttention` over the input token sequence.

    Args:
        emb_dim (int): Embedding dimension of the token sequence. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        dropout (float): Dropout probability inside the attention module. Defaults to 0.

    Returns:
        torch.Tensor: Attention output of shape [batch_size, seq_len, emb_dim].
    """

    # Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        # Create the Multi-Head Attention (MSA) layer
        self.self_attention = nn.MultiheadAttention(embed_dim=emb_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True) # does our batch dimension come first?

    # Create a forward() method to pass the data throguh the layers
    def forward(self, x):

        # Normalization layer
        x = self.layer_norm(x)

        # Multihead attention layer
        x, _ = self.self_attention(query=x, # query embeddings
                                   key=x, # key embeddings
                                   value=x, # value embeddings
                                   need_weights=False) # do we need the weights or just the layer outputs?
        return x
    
class MultiheadSelfAttentionBlockV2(nn.Module):

    """
    Custom multi-head self-attention (MSA) block implemented with scaled dot-product attention.

    This variant:
      1) Applies LayerNorm (pre-norm),
      2) Splits embeddings into multiple heads,
      3) Computes scaled dot-product attention per head (via `F.scaled_dot_product_attention`),
      4) Recombines heads and adds a residual connection.

    Args:
        emb_dim (int): Embedding dimension of the token sequence. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        dropout (float): Dropout probability applied to attention weights. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_len, emb_dim].
    """
    
    def __init__(self,
                 emb_dim: int = 768,  # Hidden size D (ViT-Base)
                 num_heads: int = 12,  # Heads (ViT-Base)
                 dropout: float = 0.0):  # Dropout for attention weights
        super().__init__()

        # Create the Norm layer (LayerNorm)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        # Store parameters
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.head_dim = emb_dim // num_heads  # Ensure emb_dim is divisible by num_heads

        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

    def split_into_heads(self, x):
        
        """
        Split input tensor into multiple heads.
        """

        batch_size, seq_len, emb_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def combine_heads(self, x):
        
        """
        Combine the heads back into a single tensor.
        """

        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        return x.contiguous().view(batch_size, seq_len, self.emb_dim)

    def forward(self, x):
        
        """
        Forward pass for the MSA block.
        """

        # Apply LayerNorm to the input
        normed_x = self.layer_norm(x)

        # Split the input tensor into multiple heads
        query = self.split_into_heads(normed_x)
        key = self.split_into_heads(normed_x)
        value = self.split_into_heads(normed_x)

        # Perform scaled dot-product attention for each head
        attn_output = F.scaled_dot_product_attention(query=query,
                                                     key=key,
                                                     value=value,
                                                     dropout_p=self.dropout,
                                                     is_causal=False)  # Set to True if causal attention is needed

        # Combine the heads back into a single tensor
        output = self.combine_heads(attn_output)

        # Add residual connection
        output = x + output

        return output


class MLPBlock(nn.Module):

    """
    Feed-forward MLP block with pre-normalization.

    Applies LayerNorm followed by a 2-layer MLP with GELU activation and dropout,
    mapping emb_dim -> mlp_size -> emb_dim.

    Args:
        emb_dim (int): Embedding dimension of the token sequence. Defaults to 768.
        mlp_size (int): Hidden dimension of the MLP. Defaults to 3072.
        dropout (float): Dropout probability applied after dense layers. Defaults to 0.1.

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_len, emb_dim].
    """

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 emb_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base (X4)
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()

        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        # Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=emb_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity"
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=emb_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer"
        )

    # Create a forward() method to pass the data throguh the layers
    def forward(self, x):

        # Putting methods together
        return self.mlp(self.layer_norm(x))
    

class TransformerEncoderBlock(nn.Module):

    """
    Transformer encoder block with pre-norm MSA and MLP sublayers.

    This block applies:
      1) MSA (with LayerNorm) + residual connection (optionally with DropPath),
      2) MLP (with LayerNorm) + residual connection (optionally with DropPath).

    Args:
        emb_dim (int): Embedding dimension. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        mlp_size (int): Hidden dimension of the MLP. Defaults to 3072.
        attn_dropout (float): Dropout probability inside the attention module. Defaults to 0.
        mlp_dropout (float): Dropout probability inside the MLP. Defaults to 0.1.
        drop_path_rate (float): DropPath probability for residual branches. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_len, emb_dim].
    """

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 attn_dropout:float=0, # Amount of dropout for attention layers
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base                 
                 ): 
        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(emb_dim=emb_dim,
                                                     num_heads=num_heads,
                                                     dropout=attn_dropout)

        # Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(emb_dim=emb_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    # 4. Create a forward() method
    def forward(self, x):

        # Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x
    

# Create a ViT class that inherits from nn.Module
class ViT(nn.Module):

    """
    Vision Transformer (ViT) model.

    Implements a ViT-style image classifier by:
      1) Converting an input image into a sequence of patch tokens (patch embedding),
      2) Adding a learnable class token and learnable positional embeddings,
      3) Processing the token sequence with a stack of Transformer encoder blocks,
      4) Applying a final LayerNorm and a classification head on the class token.

    Defaults match the ViT-Base/16 configuration (img_size=224, patch_size=16, 12 layers, 768 dim, 12 heads).

    Args:
        img_size (int): Input image resolution (assumes square images). Must be divisible by `patch_size`.
            Defaults to 224.
        in_channels (int): Number of input channels (e.g., 3 for RGB). Defaults to 3.
        patch_size (int): Patch size used to split the image into non-overlapping patches. Defaults to 16.
        num_transformer_layers (int): Number of Transformer encoder layers. Defaults to 12.
        emb_dim (int): Token embedding dimension. Defaults to 768.
        mlp_size (int): Hidden dimension of the MLP inside each encoder block. Defaults to 3072.
        num_heads (int): Number of attention heads per encoder block. Defaults to 12.
        emb_dropout (float): Dropout probability applied to embeddings (after adding positional embeddings).
            Defaults to 0.1.
        attn_dropout (float): Dropout probability inside attention layers. Defaults to 0.
        mlp_dropout (float): Dropout probability inside MLP layers. Defaults to 0.1.
        classif_head_hidden_units (int): Optional hidden dimension for an extra layer in the classifier head.
            If 0, uses a single Linear layer. Defaults to 0.
        num_classes (int): Number of output classes. Defaults to 1000.

    Returns:
        torch.Tensor: Class logits of shape [batch_size, num_classes].
    """

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 emb_dropout:float=0.1, # Dropout for patch and position embeddings
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers                                  
                 classif_head_hidden_units:int=0, # Extra hidden layer in classification header
                 num_classes:int=1000): # Default for ImageNet but can customize this
        
        """
        Initializes a Vision Transformer (ViT) model with specified hyperparameters (ViT-Base parameters by default). 

        The constructor sets up the ViT model by configuring the input image size, number of transformer layers,
        embedding dimension, number of attention heads, MLP size, and dropout rates, based on the ViT-Base configuration 
        as detailed in the original ViT paper. These parameters are also customizable to suit different downstream tasks.

        Args:
        - img_size (int, optional): The resolution of the input images. Default is 224.
        - in_channels (int, optional): The number of input image channels. Default is 3 (RGB).
        - patch_size (int, optional): The size of patches to divide the input image into. Default is 16.
        - num_transformer_layers (int, optional): The number of transformer layers. Default is 12 for ViT-Base.
        - emb_dim (int, optional): The dimensionality of the embedding space. Default is 768 for ViT-Base.
        - mlp_size (int, optional): The size of the MLP hidden layers. Default is 3072 for ViT-Base.
        - num_heads (int, optional): The number of attention heads in each transformer layer. Default is 12.
        - emb_dropout (float, optional): The dropout rate applied to patch and position embeddings. Default is 0.1.
        - attn_dropout (float, optional): The dropout rate applied to attention layers. Default is 0.
        - mlp_dropout (float, optional): The dropout rate applied to the MLP layers. Default is 0.1.        
        - classif_head_hidden_units (int, optional): The number of hidden units in the classification header. Default is 0 (no extra hidden layer).
        - num_classes (int, optional): The number of output classes. Default is 1000 for ImageNet, but can be customized.

        Note:
        This initialization is based on the ViT-Base/16 model as described in the Vision Transformer paper. Custom values can
        be provided for these parameters based on the specific task or dataset.
        """

        super().__init__() # don't forget the super().__init__()!

        # Create patch embedding layer
        self.embedder = PatchEmbedding(img_size=img_size,
                                        in_channels=in_channels,
                                        patch_size=patch_size,
                                        emb_dim=emb_dim,
                                        emb_dropout=emb_dropout)

        # Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(emb_dim=emb_dim,
                                                               num_heads=num_heads,
                                                               mlp_size=mlp_size,
                                                               attn_dropout=attn_dropout,
                                                               mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        
        # Alternative using pytorch build-in functions
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
        #                                                nhead=num_heads,
        #                                                dim_feedforward=mlp_size,
        #                                                dropout=mlp_dropout,
        #                                                activation="gelu",
        #                                                batch_first=True,
        #                                                norm_first=True)
        
        # Create the stacked transformer encoder
        #self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
        #                                     num_layers=num_transformer_layers)
        
        # Create classifier head
        if classif_head_hidden_units:
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=emb_dim),
                nn.Linear(in_features=emb_dim, out_features=classif_head_hidden_units),
                nn.GELU(),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(in_features=classif_head_hidden_units, out_features=num_classes)                
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=emb_dim),
                nn.Linear(in_features=emb_dim, out_features=num_classes)
            )
        
        # Initialize LayerNorm
        #for m in self.classifier:
        #    if isinstance(m, nn.LayerNorm):
        #        m.weight.data.fill_(1.0)
        #        m.bias.data.fill_(0.0)
        #    elif isinstance(m, nn.Linear):
        #        # Apply Xavier (Glorot) initialization
        #        #xavier_normal_(m.weight)
        #        #orthogonal_(m.weight)
        #        #kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #        if m.bias is not None:
        #            zeros_(m.bias)


    def copy_weights(self,
                      model_weights: torchvision.models.ViT_B_16_Weights):

        """
        Copies the pretrained weights from a ViT model (Vision Transformer) to the current model.
        This method assumes that the current model has a structure compatible with the ViT-base architecture.
        
        Args:
            model_weights (torchvision.models.ViT_B_16_Weights): The pretrained weights of the ViT model.
                This should be a state dictionary from a ViT-B_16 architecture, such as the one provided
                by torchvision's ViT_B_16_Weights.DEFAULT.

        Notes:
            - This method manually copies weights from the pretrained ViT model to the corresponding layers of the current model.
            - It supports the ViT-base architecture with 12 transformer encoder layers and expects a similar
            structure in the target model (e.g., embedder, encoder layers, classifier).
            - This method does not update the optimizer state or any other model parameters beyond the weights.

        Example:
            pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            model.copy_weights(pretrained_vit_weights)
        """

        # Get the current state_dict of ViT
        state_dict = self.state_dict()

        # Get the actual model weights from the input model
        pretrained_state_dict = model_weights.get_state_dict()

        # Update the parameters element-wise
        state_dict['embedder.class_token'].copy_(pretrained_state_dict['class_token'])
        state_dict['embedder.pos_embedding'].copy_(pretrained_state_dict['encoder.pos_embedding'])
        state_dict['embedder.conv_proj.weight'].copy_(pretrained_state_dict['conv_proj.weight'])
        state_dict['embedder.conv_proj.bias'].copy_(pretrained_state_dict['conv_proj.bias'])

        # Dynamically get the number of encoder layers from model_weights
        encoder_layer_keys = [key for key in pretrained_state_dict.keys() if 'encoder.layers' in key]
        num_encoder_layers = len(set([key.split('.')[2] for key in encoder_layer_keys]))

        # Update encoder layers
        for layer in range(num_encoder_layers):
            state_dict[f'encoder.{layer}.msa_block.layer_norm.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_1.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.layer_norm.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_1.bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_2.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_1.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_1.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_2.bias']
            )
        
        # Update classifier
        state_dict['classifier.0.weight'].copy_(pretrained_state_dict['encoder.ln.weight'])
        state_dict['classifier.0.bias'].copy_(pretrained_state_dict['encoder.ln.bias'])
        #state_dict['classifier.1.weight'].copy_(pretrained_state_dict['heads.head.weight'])
        #state_dict['classifier.1.bias'].copy_(pretrained_state_dict['heads.head.bias'])

        # Reload updated state_dict into the model
        self.load_state_dict(state_dict)

        print("[INFO] Model weights copied successfully.")
        print("[INFO] Model weights are trainable by default. Use function set_params_frozen to freeze them.")

    def set_params_frozen(self,                          
                          except_head:bool=True):
        
        """
        Freezes parameters of different components, allowing exceptions.

        Args:        
            except_head (bool): If True, excludes the classifier head from being frozen.
        """

        for param in self.embedder.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = except_head

    #def compile(self, backend='eager'):
    #    """
    #    Compiles the model with the selected backend using torch.compile.
    #    Args:
    #        backend (str): The backend to use. Options: 'eager', 'aot_eager', 'inductor'.
    #    """
    #    # Check if the provided backend is valid
    #    if backend not in ['eager', 'aot_eager', 'inductor', 'cudagraphs', 'onnxrt']:
    #        raise ValueError(f"Invalid backend selected: {backend}.")

        # Compile the model with the selected backend
    #    self = torch.compile(self, backend=backend)
    #    print(f"Model compiled with backend: {backend}")

    # Create a forward() method
    def forward(self, x):

        """
        Forward pass of the Vision Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes].
        """

        # Create patch embedding (equation 1)
        #x = self.embedder(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        #x = self.encoder(x)

        # Put 0 index logit through classifier (equation 4)
        #x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index
        
        x = self.classifier(self.encoder(self.embedder(x))[:,0])

        return x


# Create a ViT class that inherits from nn.Module
class ViTv2(nn.Module):

    """
    Vision Transformer (ViT) model.

    Implements a ViT-style image classifier by:
      1) Converting an input image into a sequence of patch tokens (patch embedding),
      2) Adding a learnable class token and learnable positional embeddings,
      3) Processing the token sequence with a stack of Transformer encoder blocks,
      4) Applying a final LayerNorm and a classification head on the class token.

    Defaults match the ViT-Base/16 configuration (img_size=224, patch_size=16, 12 layers, 768 dim, 12 heads).

    Args:
        img_size (int): Input image resolution (assumes square images). Must be divisible by `patch_size`.
            Defaults to 224.
        in_channels (int): Number of input channels (e.g., 3 for RGB). Defaults to 3.
        patch_size (int): Patch size used to split the image into non-overlapping patches. Defaults to 16.
        num_transformer_layers (int): Number of Transformer encoder layers. Defaults to 12.
        emb_dim (int): Token embedding dimension. Defaults to 768.
        mlp_size (int): Hidden dimension of the MLP inside each encoder block. Defaults to 3072.
        num_heads (int): Number of attention heads per encoder block. Defaults to 12.
        emb_dropout (float): Dropout probability applied to embeddings (after adding positional embeddings).
            Defaults to 0.1.
        attn_dropout (float): Dropout probability inside attention layers. Defaults to 0.
        mlp_dropout (float): Dropout probability inside MLP layers. Defaults to 0.1.
        classif_head_hidden_units (int): Optional hidden dimension for an extra layer in the classifier head.
            If 0, uses a single Linear layer. Defaults to 0.
        num_classes (int): Number of output classes. Defaults to 1000.

    Returns:
        torch.Tensor: Class logits of shape [batch_size, num_classes].
    """
    
    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 emb_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 emb_dropout:float=0.1, # Dropout for patch and position embeddings
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers                                  
                 classif_heads:nn.Module=None, # Classification head(s)
                 num_classes:int=1000): # Default for ImageNet but can customize this
        
        """
        Initializes a Vision Transformer (ViT) model with specified hyperparameters (ViT-Base parameters by default). 
        V2 is identical to V1 except that the classification head can be passed as an argument, allowing for customization 
        of the number of hidden layers and units per layer.

        The constructor sets up the ViT model by configuring the input image size, number of transformer layers,
        embedding dimension, number of attention heads, MLP size, and dropout rates, based on the ViT-Base configuration 
        as detailed in the original ViT paper. These parameters are also customizable to suit different downstream tasks.

        Args:
        - img_size (int, optional): The resolution of the input images. Default is 224.
        - in_channels (int, optional): The number of input image channels. Default is 3 (RGB).
        - patch_size (int, optional): The size of patches to divide the input image into. Default is 16.
        - num_transformer_layers (int, optional): The number of transformer layers. Default is 12 for ViT-Base.
        - emb_dim (int, optional): The dimensionality of the embedding space. Default is 768 for ViT-Base.
        - mlp_size (int, optional): The size of the MLP hidden layers. Default is 3072 for ViT-Base.
        - num_heads (int, optional): The number of attention heads in each transformer layer. Default is 12.
        - emb_dropout (float, optional): The dropout rate applied to patch and position embeddings. Default is 0.1.
        - attn_dropout (float, optional): The dropout rate applied to attention layers. Default is 0.
        - mlp_dropout (float, optional): The dropout rate applied to the MLP layers. Default is 0.1.
        - classif_heads (nn.Module, optional): An optional extra classification headers. Default is None, no hidden layer is used.        
        - num_classes (int, optional): The number of output classes. Default is 1000 for ImageNet, but can be customized.

        Note:
        This initialization is based on the ViT-Base/16 model as described in the Vision Transformer paper. Custom values can
        be provided for these parameters based on the specific task or dataset.

        Usage of classif_heads:
        - If provided, it will be used as the final classification layer(s) of the model.
        - If None, a default single-layer classification head will be used with the specified number of classes.
        - This allows for flexibility in the final layer(s) of the model, enabling customization based on the task requirements.
        
        def create_classification_heads(num_heads: int, emb_dim: int, num_classes: int) -> list:        
            heads = []
            for i in range(num_heads):
                head = nn.Sequential(
                    nn.LayerNorm(normalized_shape=emb_dim),
                    nn.Linear(in_features=emb_dim, out_features=emb_dim // 2),
                    nn.GELU(),
                    nn.Linear(in_features=emb_dim // 2, out_features=num_classes)
                )
            heads.append(head)
            return heads -> classif_heads
        """

        super().__init__() # don't forget the super().__init__()!

        # Create patch embedding layer
        self.embedder = PatchEmbedding(img_size=img_size,
                                        in_channels=in_channels,
                                        patch_size=patch_size,
                                        emb_dim=emb_dim,
                                        emb_dropout=emb_dropout)

        # Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(emb_dim=emb_dim,
                                                               num_heads=num_heads,
                                                               mlp_size=mlp_size,
                                                               attn_dropout=attn_dropout,
                                                               mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        
        # Alternative using pytorch build-in functions
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
        #                                                nhead=num_heads,
        #                                                dim_feedforward=mlp_size,
        #                                                dropout=mlp_dropout,
        #                                                activation="gelu",
        #                                                batch_first=True,
        #                                                norm_first=True)
        
        # Create the stacked transformer encoder
        #self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
        #                                     num_layers=num_transformer_layers)
        
        # 4. Create classifier head
        if classif_heads:
            self.classifier = nn.ModuleList(classif_heads)
        else:
            classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=emb_dim),
                nn.Linear(in_features=emb_dim, out_features=num_classes)
            )
            self.classifier = nn.ModuleList([classifier])


    def copy_weights(self,
                     model_weights: torchvision.models.ViT_B_16_Weights):

        """
        Copies the pretrained weights from a ViT model (Vision Transformer) to the current model.
        This method assumes that the current model has a structure compatible with the ViT-base architecture.
        
        Args:
            model_weights (torchvision.models.ViT_B_16_Weights): The pretrained weights of the ViT model.
                This should be a state dictionary from a ViT-B_16 architecture, such as the one provided
                by torchvision's ViT_B_16_Weights.DEFAULT.

        Notes:
            - This method manually copies weights from the pretrained ViT model to the corresponding layers of the current model.
            - It supports the ViT-base architecture with 12 transformer encoder layers and expects a similar
            structure in the target model (e.g., embedder, encoder layers, classifier).
            - This method does not update the optimizer state or any other model parameters beyond the weights.

        Example:
            pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            model.copy_weights(pretrained_vit_weights)
        """

        # Get the current state_dict of ViT
        state_dict = self.state_dict()

        # Get the actual model weights from the input model
        pretrained_state_dict = model_weights.get_state_dict()

        # Update the parameters element-wise
        state_dict['embedder.class_token'].copy_(pretrained_state_dict['class_token'])
        state_dict['embedder.pos_embedding'].copy_(pretrained_state_dict['encoder.pos_embedding'])
        state_dict['embedder.conv_proj.weight'].copy_(pretrained_state_dict['conv_proj.weight'])
        state_dict['embedder.conv_proj.bias'].copy_(pretrained_state_dict['conv_proj.bias'])

        # Dynamically get the number of encoder layers from model_weights
        encoder_layer_keys = [key for key in pretrained_state_dict.keys() if 'encoder.layers' in key]
        num_encoder_layers = len(set([key.split('.')[2] for key in encoder_layer_keys]))

        # Update encoder layers
        for layer in range(num_encoder_layers):
            state_dict[f'encoder.{layer}.msa_block.layer_norm.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_1.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.layer_norm.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_1.bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.in_proj_bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.in_proj_bias']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.weight']
            )
            state_dict[f'encoder.{layer}.msa_block.self_attention.out_proj.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.self_attention.out_proj.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.layer_norm.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.ln_2.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_1.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.0.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_1.bias']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.weight'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_2.weight']
            )
            state_dict[f'encoder.{layer}.mlp_block.mlp.3.bias'].copy_(
                pretrained_state_dict[f'encoder.layers.encoder_layer_{layer}.mlp.linear_2.bias']
            )
        
        # Update classifier
        state_dict['classifier.0.weight'].copy_(pretrained_state_dict['encoder.ln.weight'])
        state_dict['classifier.0.bias'].copy_(pretrained_state_dict['encoder.ln.bias'])
        #state_dict['classifier.1.weight'].copy_(pretrained_state_dict['heads.head.weight'])
        #state_dict['classifier.1.bias'].copy_(pretrained_state_dict['heads.head.bias'])

        # Reload updated state_dict into the model
        self.load_state_dict(state_dict)

    def set_params_frozen(self,                          
                          except_head:bool=True):

        """
        Freezes parameters of different components, allowing exceptions.

        Args:        
            except_head (bool): If True, excludes the classifier head from being frozen.
        """

        for param in self.embedder.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = except_head

    def compile(self):
        """Compile the model using torch.compile for optimization."""
        self.__compiled__ = torch.compile(self)
        print("Model compiled successfully with torch.compile.")

    # Create a forward() method
    def forward(self, x):

        """
        Forward pass of the Vision Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes].
        """

        # Create patch embedding (equation 1)
        x = self.embedder(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.encoder(x)

        # Pass the outpu embeddings through the classification heads (as a list)
        x_list = [head(x[:, 0]) for head in self.classifier]
        x = torch.mean(torch.stack(x_list), dim=0)

        return x
    

    # Algorithm to be tested. We have a luma reconstructed (previoulsy compressed) block and you want
    # to predit the chroma blocks using ViT.

from timm.models.vision_transformer import vit_small_patch16_224

# Without QP
class ViTChromaPredictor(nn.Module):
    def __init__(self, input_size=32, patch_size=4, embed_dim=384, output_channels=2, qp_embed_dim=128):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Load a ViT (you can also define your own or use LViT)
        self.vit = vit_small_patch16_224(pretrained=False, img_size=input_size, in_chans=1, num_classes=0)
        
        self.qp_embedding = nn.Sequential(
            nn.Linear(1, qp_embed_dim),
            nn.ReLU(),
            nn.Linear(qp_embed_dim, embed_dim)
        )

        # Conv head: reshape [B, N, D] → [B, D, H', W']
        self.conv_head_1 = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(128, output_channels, kernel_size=1)  # Output chroma channels (Cb, Cr)
        )

        self.conv_head_2 = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=patch_size),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        B = x.size(0)
        x = self.vit.patch_embed(x)  # [B, N, D]
        x = x + self.vit.pos_embed[:, 1:(self.num_patches + 1)]
        x = self.vit.blocks(x)
        x = self.vit.norm(x)  # [B, N, D]

        # Reshape to feature map
        H_p = W_p = self.input_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p)  # [B, D, H', W']

        # Conv head
        chroma_pred = self.conv_head_1(x)  # [B, 2, H, W]
        return chroma_pred


from torchvision.models.vision_transformer import vit_b_16  # or a smaller one

class ViTChromaPredictor(nn.Module):
    def __init__(self, patch_size=4, embed_dim=768, output_channels=2, meta_embed_dim=128):
        super().__init__()
        
        # ViT encoder (can use pretrained encoder if you like)
        self.patch_size = patch_size
        self.vit = vit_b_16(pretrained=False)
        self.vit.conv_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)  # for grayscale

        # Map QP to an embedding vector
        self.meta_embedding = nn.Sequential(
            nn.Linear(3, meta_embed_dim), #[qp, h_img, w_img]
            nn.ReLU(),
            nn.Linear(meta_embed_dim, embed_dim)
        )
        
        # CNN decoder head (predicts chroma block from ViT features)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=1)  # 2 channels for U and V
        )

    def forward(self, luma_block, h_chroma, w_chroma, qp, h_img, w_img):
        """
        luma_blcok: [B, 1, H, W] — grayscale input block
        h_chroma, w_chroma: int — target chroma height and width
        qp: [B] or [B,1] — scalar QP values (normalized)
        h_img, w_img: [B] or [B,1] — scalar image height/width (normalized)
        """

        # Remove last dimension if QP, h_img or w_img are [B,1] tensors
        qp = qp.squeeze(-1) if qp.ndim == 2 else qp
        h_img = h_img.squeeze(-1) if h_img.ndim == 2 else h_img
        w_img = w_img.squeeze(-1) if w_img.ndim == 2 else w_img
        
        # Ensure the input luma block size is divisible by patch size (required by ViT)
        B, _, h_luma, w_luma = luma_block.shape
        assert h_luma % self.patch_size == 0 and w_luma % self.patch_size == 0, "Luma block size must be divisible by patch size"

        # ViT encoding

        # Pass luma through the ViT convolutional projection to create patch embeddings
        # Resulting shape: [B, embed_dim, H_p, W_p], where H_p = h_luma/patch_size
        x = self.vit.conv_proj(luma_block)  # [B, embed_dim, H_p, W_p]
        h_p, w_p = x.shape[2], x.shape[3] #8x8 by default

        # Flatten spatial patches and transpose to get sequence of patch embeddings
        # Resulting shape: [B, N, D], where N = number of patches (H_p * W_p), D = embedding dim
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # 1) Prepare, 2) embed, and 3) add meta features as a token
        meta_input = torch.stack([qp, w_img, h_img], dim=1).float() #[B, 3]
        meta_embed = self.meta_embedding(meta_input).unsqueeze(1)  # [B, 1, D]
        x = torch.cat([meta_embed, x], dim=1)  # [B, N+1, D]

        # Run transformer encoder on sequence of tokens including meta token. Skip classification head.
        x = self.vit.encoder(x)  # [B, N+1, D]

        # Discard the meta token to keep only patch tokens
        x = x[:, 1:, :]  # [B, N, D]

        # Reshape tokens back to spatial layout
        x = x.transpose(1, 2).reshape(B, self.vit.hidden_dim, h_p, w_p)  # [B, D, h_p, w_p]

        # If the spatial size after encoding differs from target chroma size, then interpolate the signal
        if (h_p != h_chroma) or (w_p != w_chroma):
            x = F.interpolate(x, size=(h_chroma, w_chroma), mode='bilinear', align_corners=False)

        # CNN decoder: decode the features to predict chroma channels (U and V)
        chroma = self.decoder(x)  # [B, 2, H, W]

        return chroma
