import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.nn.init import trunc_normal_

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
        
        # distillation token embedding class
        self.distillation_token = trunc_normal_(nn.Parameter(torch.zeros(1, 1, emb_dim), requires_grad=True), std=0.02)
                       
        # Position embedding class
        num_patches = (img_size * img_size) // patch_size**2
        self.pos_embedding = trunc_normal_(nn.Parameter(torch.zeros(1, num_patches+2, emb_dim), requires_grad=True), std=0.02)
        
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

        # Create class token
        class_token = self.class_token.expand(x.shape[0], -1, -1) # Expand to match with batch size
        
        # Create distillation
        distillation_token = self.distillation_token.expand(x.shape[0], -1, -1)

        # Prepend classification token and append distillation token
        x = torch.cat((class_token, x, distillation_token), dim=1)
        
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
    
class DropPath(nn.Module):
    
    """
    Stochastic depth (DropPath) regularization.

    Randomly drops entire residual paths on a per-sample basis during training,
    and rescales surviving paths by 1/(1 - drop_prob).

    Args:
        drop_prob (float): Probability of dropping the residual path. Defaults to 0.0.

    Returns:
        torch.Tensor: Tensor with the same shape as input `x`.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob

        # Work with (B, ..., C) tensors
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask

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
                 drop_path_rate: float=0.0, # Amount of stochastic depth (DropPath): probability of dropping residual paths
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
        
        # Create drop paths
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    # Create a forward() method
    def forward(self, x):

        # Create residual connection for MSA block (add the input to the output)
        x =  self.drop_path1(self.msa_block(x)) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.drop_path2(self.mlp_block(x)) + x

        return x
    

# Create a ViT-base DeiT class that inherits from nn.Module
class DeiT(nn.Module):

    """
    Data-efficient Image Transformer (DeiT) model with classification and distillation heads.

    This implementation follows a ViT-style backbone with:
      - Patch embedding + positional embeddings,
      - A stack of Transformer encoder blocks,
      - A final LayerNorm,
      - Two heads: a classifier head (class token) and a distillation head (distillation token).

    Args:
        img_size (int): Input image resolution (assumes square images). Defaults to 224.
        in_channels (int): Number of input channels. Defaults to 3.
        patch_size (int): Patch size used for tokenization. Defaults to 16.
        num_transformer_layers (int): Number of Transformer encoder layers. Defaults to 12.
        emb_dim (int): Embedding dimension. Defaults to 768.
        mlp_size (int): Hidden dimension of the MLP in each encoder block. Defaults to 3072.
        num_heads (int): Number of attention heads per encoder block. Defaults to 12.
        emb_dropout (float): Dropout probability applied to embeddings. Defaults to 0.1.
        attn_dropout (float): Dropout probability inside attention layers. Defaults to 0.0.
        mlp_dropout (float): Dropout probability inside MLP layers. Defaults to 0.1.
        drop_path_rate (float): Maximum DropPath probability across layers (linearly scheduled).
            Defaults to 0.0.
        classif_head_hidden_units (int): Optional hidden units for an extra layer in the classifier head.
            If 0, uses a single Linear layer. Defaults to 0.
        distill_head_hidden_units (int): Optional hidden units for an extra layer in the distillation head.
            If 0, uses a single Linear layer. Defaults to 0.
        num_classes (int): Number of output classes. Defaults to 1000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (xc, xd) where:
            - xc is the classifier logits from the class token of shape [batch_size, num_classes],
            - xd is the distillation logits from the distillation token of shape [batch_size, num_classes].
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
                 attn_dropout:float=0.0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 drop_path_rate: float=0.0, # Stochastic depth (DropPath): probability of dropping residual paths
                 classif_head_hidden_units:int=0, # Extra hidden layer in classification header
                 distill_head_hidden_units:int=0, # Extra hidden layer in distillation header
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
        - attn_dropout (float, optional): The dropout rate applied to attention layers. Default is 0.0.
        - mlp_dropout (float, optional): The dropout rate applied to the MLP layers. Default is 0.1.        
        - drop_path_rate (float, optional): Stochastic depth (DropPath): probability of dropping residual paths. Default is 0.0.
        - classif_head_hidden_units (int, optional): The number of hidden units in the classification header. Default is 0 (no extra hidden layer).
        - distill_head_hidden_units (int, optional): The number of hidden units in the distillation header. Default is 0 (no extra hidden layer).
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
        dpr = torch.linspace(0, drop_path_rate, steps=num_transformer_layers).tolist()
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(emb_dim=emb_dim,
                                                               num_heads=num_heads,
                                                               mlp_size=mlp_size,
                                                               attn_dropout=attn_dropout,
                                                               mlp_dropout=mlp_dropout,
                                                               drop_path_rate=dpr[i]) for i in range(num_transformer_layers)])
        
        # Alternative using pytorch build-in functions
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
        #                                               nhead=num_heads,
        #                                                dim_feedforward=mlp_size,
        #                                                dropout=mlp_dropout,
        #                                                activation="gelu",
        #                                                batch_first=True,
        #                                                norm_first=True)
        
        # Create the stacked transformer encoder
        #self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
        #                                     num_layers=num_transformer_layers)
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        # Create classifier head
        if classif_head_hidden_units:
            self.classifier = nn.Sequential(        
                nn.Linear(in_features=emb_dim, out_features=classif_head_hidden_units),
                nn.GELU(),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(in_features=classif_head_hidden_units, out_features=num_classes)                
            )
        else:
            self.classifier = nn.Linear(in_features=emb_dim, out_features=num_classes)

        
        # Create distillation head
        if distill_head_hidden_units:
            self.distiller = nn.Sequential(                
                nn.Linear(in_features=emb_dim, out_features=distill_head_hidden_units),
                nn.GELU(),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(in_features=distill_head_hidden_units, out_features=num_classes)                
            )
        else:
            self.distiller = nn.Linear(in_features=emb_dim, out_features=num_classes)

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
        state_dict['embedder.pos_embedding'][:, :-1].copy_(pretrained_state_dict['encoder.pos_embedding'])
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
        if isinstance(self.classifier, nn.Sequential):
            state_dict['classifier.0.weight'].copy_(pretrained_state_dict['encoder.ln.weight'])
            state_dict['classifier.0.bias'].copy_(pretrained_state_dict['encoder.ln.bias'])
        else:
            state_dict['classifier.weight'].copy_(pretrained_state_dict['encoder.ln.weight'])
            state_dict['classifier.bias'].copy_(pretrained_state_dict['encoder.ln.bias'])

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

    # Create a forward() method
    def forward(self, x):
        
        """
        Forward pass of the Vision Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes].
        """

        # Extract batch size
        
        # Create patch embedding (equation 1)
        x = self.embedder(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.encoder(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Put 0 index logit through classifier and -1 index logit through distiller        
        xc = self.classifier(x[:,0])
        xd = self.distiller(x[:,-1])
        
        # print("DeiT.forward returning:", type((xc, xd)), xc.shape, xd.shape)

        return xc, xd

