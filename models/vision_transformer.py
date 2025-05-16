import torch
import torchvision
from torch import nn
from torch.nn.init import trunc_normal_
#, xavier_normal_, zeros_, orthogonal_, kaiming_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

class HyperspectralToRGB(nn.Module):
    def __init__(self):
        super(HyperspectralToRGB, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=125, out_channels=3, kernel_size=(3, 3, 3), padding=1)
    
    def forward(self, x):
        return self.conv1(x) 

class SpatialConcatenation(nn.Module):
    def __init__(
        self,
        in_channels: int=125,
        spatial_dim: int=384,
        row_blocks: int=6,
        column_blocks: int=6,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        super(SpatialConcatenation, self).__init__()

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
        # Input x: (B, C, H, W)
        x = x.unsqueeze(1)  # -> (B, 1, C, H, W)
        #print(x.shape)
        x = self.conv3d(x)  # -> (B, 3, C, H, W)
        #print(x.shape)
        x = x.squeeze(2)    # -> (B, 3, H, W)
        #print(x.shape)
        return x


class SpectralViT(nn.Module):
    def __init__(
        self,
        model: str='vitbase16',
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

        self.model = self.create_model(
            model=model,
            num_classes=num_classes,
            dropout=dropout,
            freeze=freeze_model,
            seed=seed,
            device=device,
            )

    @staticmethod
    def create_model(
        model: str = "vitbase16",
        num_classes: int = 1,
        dropout: float = 0.0,
        freeze: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed: int = 42
    ):

        model_map = {
            # ViT-Base / -Large / -Huge
            "vitbase16":    (torchvision.models.vit_b_16,    torchvision.models.ViT_B_16_Weights.DEFAULT,                  768),
            "vitbase16_2":  (torchvision.models.vit_b_16,    torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,  768),
            "vitbase32":    (torchvision.models.vit_b_32,    torchvision.models.ViT_B_32_Weights.DEFAULT,                  768),
            "vitlarge16":   (torchvision.models.vit_l_16,    torchvision.models.ViT_L_16_Weights.DEFAULT,                 1024),
            "vitlarge16_2": (torchvision.models.vit_l_16,    torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1, 1024),
            "vitlarge32":   (torchvision.models.vit_l_32,    torchvision.models.ViT_L_32_Weights.DEFAULT,                 1024),
            "vithuge14":    (torchvision.models.vit_h_14,    torchvision.models.ViT_H_14_Weights.DEFAULT,                 1280),

            # Swin-T / -S / -B
            "swin_t":       (torchvision.models.swin_t,      torchvision.models.Swin_T_Weights.DEFAULT,                   768),
            "swin_s":       (torchvision.models.swin_s,      torchvision.models.Swin_S_Weights.DEFAULT,                   768),
            "swin_b":       (torchvision.models.swin_b,      torchvision.models.Swin_B_Weights.DEFAULT,                  1024),

            # Swin-V2 T / S / B
            "swin_v2_t":    (torchvision.models.swin_v2_t,   torchvision.models.Swin_V2_T_Weights.DEFAULT,                768),
            "swin_v2_s":    (torchvision.models.swin_v2_s,   torchvision.models.Swin_V2_S_Weights.DEFAULT,                768),
            "swin_v2_b":    (torchvision.models.swin_v2_b,   torchvision.models.Swin_V2_B_Weights.DEFAULT,               1024),

            # EfficientNet
            "efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT, 1280),
            "efficientnet_b1": (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT, 1280),
            "efficientnet_b2": (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT, 1408),
            "efficientnet_b3": (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.DEFAULT, 1536),
            "efficientnet_b4": (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.DEFAULT, 1792),

            # ResNet v1
            "resnet18":     (torchvision.models.resnet18,    torchvision.models.ResNet18_Weights.IMAGENET1K_V1,                 512),
            "resnet34":     (torchvision.models.resnet34,    torchvision.models.ResNet34_Weights.IMAGENET1K_V1,                 512),
            "resnet50":     (torchvision.models.resnet50,    torchvision.models.ResNet50_Weights.IMAGENET1K_V1,                2048),
            "resnet101":    (torchvision.models.resnet101,   torchvision.models.ResNet101_Weights.IMAGENET1K_V1,               2048),
            "resnet152":    (torchvision.models.resnet152,   torchvision.models.ResNet152_Weights.IMAGENET1K_V1,               2048),

            # ResNet v2
            "resnet50_v2":  (torchvision.models.resnet50,    torchvision.models.ResNet50_Weights.IMAGENET1K_V2,                     2048),
            "resnet101_v2": (torchvision.models.resnet101,   torchvision.models.ResNet101_Weights.IMAGENET1K_V2,                    2048),
            "resnet152_v2": (torchvision.models.resnet152,   torchvision.models.ResNet152_Weights.IMAGENET1K_V2,                    2048),

            # MobileNet
            "mobilenet_v2": (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.DEFAULT,           1280),
            "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, torchvision.models.MobileNet_V3_Small_Weights.DEFAULT, 1024),
            "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.DEFAULT, 1280),

            # ConvNeXt
            "convnext_t":   (torchvision.models.convnext_tiny,  torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,         768),
            "convnext_s":   (torchvision.models.convnext_small, torchvision.models.ConvNeXt_Small_Weights.DEFAULT,       768),
            "convnext_b":   (torchvision.models.convnext_base,  torchvision.models.ConvNeXt_Base_Weights.DEFAULT,       1024),
            "convnext_l":   (torchvision.models.convnext_large, torchvision.models.ConvNeXt_Large_Weights.DEFAULT,      1536),
        }

        
        if model not in model_map:
            raise ValueError(f"Unsupported ViT model name: {model}")

        vit_fn, weights_enum, hidden_dim = model_map[model]
        model = vit_fn(weights=weights_enum, dropout=dropout).to(device)

        # Optional freezing
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        # Set the seed for general torch operations
        torch.manual_seed(seed)

        # Set the seed for CUDA torch operations (ones that happen on the GPU)
        torch.cuda.manual_seed(seed)

        # Replace head
        # Replace head / classifier / fc
        if hasattr(model, "heads"):  # ViT models
            model.heads = nn.Linear(hidden_dim, num_classes).to(device)
        elif hasattr(model, "head"):  # ViT models
            model.head = nn.Linear(hidden_dim, num_classes).to(device)
        elif hasattr(model, "classifier"):  # EfficientNet, MobileNet, ConvNeXt
            if isinstance(model.classifier, nn.Sequential):  # MobileNet, EfficientNet
                model.classifier[-1] = nn.Linear(hidden_dim, num_classes).to(device)
            else:
                model.classifier = nn.Linear(hidden_dim, num_classes).to(device)
        elif hasattr(model, "fc"):  # ResNet
            model.fc = nn.Linear(hidden_dim, num_classes).to(device)
        else:
            raise ValueError(f"Cannot replace classifier head for model type: {type(model)}")


        #model.heads = nn.Sequential(
        #        nn.LayerNorm(normalized_shape=emb_dim),
        #        nn.Linear(in_features=emb_dim, out_features=num_classes)
        #    )

        #model.heads = nn.Sequential(
        #    nn.LayerNorm(hidden_dim),
        #    nn.Linear(hidden_dim, 100),
        #    nn.GELU(),
        #    nn.Linear(100, 1)
        #)

        return model

    def forward(self, x):

        x = self.embedding(x)  # -> (B, 3, H, W)
        x = self.model(x)  # -> (B, num_classes)

        return x



# Implementation of a vision transformer following the paper "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"

class PatchEmbedding(nn.Module):

    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        emb_dim (int): Size of embedding to turn image into. Defaults to 768.
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
        self.conv_proj = nn.Conv2d(in_channels=in_channels,
                                   out_channels=emb_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
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
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        
        # Create position embedding
        x = x + self.pos_embedding

        # Run embedding dropout (Appendix B.1)              
        x = self.emb_dropout(x)

        return x


class MultiheadSelfAttentionBlock(nn.Module):

    """
    Creates a multi-head self-attention block ("MSA block" for short).
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttentionBlockV2(nn.Module):

    """
    Creates a custom multi-head self-attention block using scaled dot-product attention.
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
        """Split input tensor into multiple heads."""
        batch_size, seq_len, emb_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def combine_heads(self, x):
        """Combine the heads back into a single tensor."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        return x.contiguous().view(batch_size, seq_len, self.emb_dim)

    def forward(self, x):
        """Forward pass for the MSA block."""
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
    Creates a layer normalized multilayer perceptron block ("MLP block" for short).
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
    Creates a Transformer Encoder block.
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
    Creates a Vision Transformer architecture with ViT-Base hyperparameters by default.
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
        - classif_head (nn.Module, optional): An optional extra classification header. Default is None, no hidden layer is used.        
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