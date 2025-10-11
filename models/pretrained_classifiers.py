import torch
import torchvision
from torch import nn

def replace_classifier(model, hidden_dim, num_classes, device):

    """
    Replace the classification head of the model with a new linear layer
    matching the number of output classes.

    Args:
        model (torch.nn.Module): The pretrained model.
        hidden_dim (int): Number of input features to the classification head.
        num_classes (int): Number of output classes.
        device (torch.device): Device to move the new head to.
    """

    if hasattr(model, "heads"):  # Vision Transformer (ViT) models
        model.heads = nn.Linear(hidden_dim, num_classes).to(device)
    elif hasattr(model, "head"):  # Some ViT variants
        model.head = nn.Linear(hidden_dim, num_classes).to(device)
    elif hasattr(model, "classifier"):  # EfficientNet, MobileNet, ConvNeXt
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(hidden_dim, num_classes).to(device)
        else:
            model.classifier = nn.Linear(hidden_dim, num_classes).to(device)
    elif hasattr(model, "fc"):  # ResNet models
        model.fc = nn.Linear(hidden_dim, num_classes).to(device)
    else:
        raise ValueError(f"Cannot replace classifier head for model type: {type(model)}")


def build_pretrained_classifier(
    model: str = "vit_b_16_224",
    num_classes: int = 1,
    dropout: float = None,
    freeze: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42
) -> nn.Module:
    
    """
    Creates and configures a pretrained image classification model from torchvision,
    with support for various architectures (ViT, Swin, ResNet, EfficientNet, MobileNet, ConvNeXt, etc.).

    This function:
      - Loads the requested model with pretrained weights.
      - Replaces its classification head for a new task (num_classes output).
      - Optionally freezes the pretrained layers.
      - Sets random seeds for reproducibility.
      - Moves the model to the specified device.
    
    Supported vision models:

    Args:
        model (str): Model identifier string (e.g., "vitbase16", "swin_t", "resnet50"). 
            
            Supported vision models include:

            - "vit_b_16_224":  ViT-Base with 16x16 patches, input size 224x224
            - "vit_b_16_384":  ViT-Base with 16x16 patches, input size 384x384
            - "vit_b_32_224":  ViT-Base with 32x32 patches, input size 224x224
            - "vit_l_16_224":  ViT-Large with 16x16 patches, input size 224x224
            - "vit_l_16_384":  ViT-Large with 16x16 patches, input size 384x384
            - "vit_l_32_224":  ViT-Large with 32x32 patches, input size 224x224
            - "vit_h_14_224":  ViT-Huge with 14x14 patches, input size 224x224
            - "vit_h_14_518":  ViT-Huge with 14x14 patches, input size 518x518
            - "swin_t_224":    Swin Transformer Tiny, input size 224x224
            - "swin_s_224":    Swin Transformer Small, input size 224x224
            - "swin_b_224":    Swin Transformer Base, input size 224x224
            - "swin_v2_t_256": Swin V2 Transformer Tiny, input size 256x256
            - "swin_v2_s_256": Swin V2 Transformer Small, input size 256x256
            - "swin_v2_b_256": Swin V2 Transformer Base, input size 256x256
            - "efficientnet_b0" to "efficientnet_b4": EfficientNet variants B0-B4
            - "resnet18", "resnet34", "resnet50", "resnet101", "resnet152": ResNet family
            - "resnet50_v2", "resnet101_v2", "resnet152_v2": ResNet v2 family
            - "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large": MobileNet variants
            - "convnext_t", "convnext_s", "convnext_b", "convnext_l": ConvNeXt variants

        num_classes (int): Number of output classes for the classification head.
        dropout (float): Dropout probability for the new head (if applicable).
            - If none (default) the default dropout probability is used.
        freeze (bool): Whether to freeze all pretrained weights.
        device (torch.device): Device to load the model onto.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.nn.Module: Configured and ready-to-train PyTorch model.
    """
        
        
    # Map of model names to their constructor, pretrained weights, and hidden dimension of the last feature layer.
    model_map = {

        # Vision Transformers (ViT)
        "vit_b_16_224": (torchvision.models.vit_b_16,    torchvision.models.ViT_B_16_Weights.DEFAULT,                    768),
        "vit_b_16_384": (torchvision.models.vit_b_16,    torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,     768),
        "vit_b_32_224": (torchvision.models.vit_b_32,    torchvision.models.ViT_B_32_Weights.DEFAULT,                    768),
        "vit_l_16_224": (torchvision.models.vit_l_16,    torchvision.models.ViT_L_16_Weights.DEFAULT,                   1024),
        "vit_l_16_384": (torchvision.models.vit_l_16,    torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1,    1024),
        "vit_l_32_224": (torchvision.models.vit_l_32,    torchvision.models.ViT_L_32_Weights.DEFAULT,                   1024),
        "vit_h_14_224": (torchvision.models.vit_h_14,    torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1, 1280),
        "vit_h_14_518": (torchvision.models.vit_h_14,    torchvision.models.ViT_H_14_Weights.DEFAULT,                   1280),

        # Swin Transformers
        "swin_t":       (torchvision.models.swin_t,      torchvision.models.Swin_T_Weights.DEFAULT,                    768),
        "swin_s":       (torchvision.models.swin_s,      torchvision.models.Swin_S_Weights.DEFAULT,                    768),
        "swin_b":       (torchvision.models.swin_b,      torchvision.models.Swin_B_Weights.DEFAULT,                   1024),

        # Swin V2
        "swin_v2_t":    (torchvision.models.swin_v2_t,   torchvision.models.Swin_V2_T_Weights.DEFAULT,                 768),
        "swin_v2_s":    (torchvision.models.swin_v2_s,   torchvision.models.Swin_V2_S_Weights.DEFAULT,                 768),
        "swin_v2_b":    (torchvision.models.swin_v2_b,   torchvision.models.Swin_V2_B_Weights.DEFAULT,                1024),

        # EfficientNet family
        "efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT,   1280),
        "efficientnet_b1": (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT,   1280),
        "efficientnet_b2": (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT,   1408),
        "efficientnet_b3": (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.DEFAULT,   1536),
        "efficientnet_b4": (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.DEFAULT,   1792),

        # ResNet v1
        "resnet18":     (torchvision.models.resnet18,    torchvision.models.ResNet18_Weights.IMAGENET1K_V1,            512),
        "resnet34":     (torchvision.models.resnet34,    torchvision.models.ResNet34_Weights.IMAGENET1K_V1,            512),
        "resnet50":     (torchvision.models.resnet50,    torchvision.models.ResNet50_Weights.IMAGENET1K_V1,           2048),
        "resnet101":    (torchvision.models.resnet101,   torchvision.models.ResNet101_Weights.IMAGENET1K_V1,          2048),
        "resnet152":    (torchvision.models.resnet152,   torchvision.models.ResNet152_Weights.IMAGENET1K_V1,          2048),

        # ResNet v2 (newer weights)
        "resnet50_v2":  (torchvision.models.resnet50,    torchvision.models.ResNet50_Weights.IMAGENET1K_V2,           2048),
        "resnet101_v2": (torchvision.models.resnet101,   torchvision.models.ResNet101_Weights.IMAGENET1K_V2,          2048),
        "resnet152_v2": (torchvision.models.resnet152,   torchvision.models.ResNet152_Weights.IMAGENET1K_V2,          2048),

        # MobileNet
        "mobilenet_v2":        (torchvision.models.mobilenet_v2,        torchvision.models.MobileNet_V2_Weights.DEFAULT,        1280),
        "mobilenet_v3_small":  (torchvision.models.mobilenet_v3_small,  torchvision.models.MobileNet_V3_Small_Weights.DEFAULT,  1024),
        "mobilenet_v3_large":  (torchvision.models.mobilenet_v3_large,  torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,  1280),

        #  ConvNeXt
        "convnext_t":   (torchvision.models.convnext_tiny,  torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,          768),
        "convnext_s":   (torchvision.models.convnext_small, torchvision.models.ConvNeXt_Small_Weights.DEFAULT,         768),
        "convnext_b":   (torchvision.models.convnext_base,  torchvision.models.ConvNeXt_Base_Weights.DEFAULT,         1024),
        "convnext_l":   (torchvision.models.convnext_large, torchvision.models.ConvNeXt_Large_Weights.DEFAULT,        1536),
    }
    
    # Verify if the requested model is supported
    if model not in model_map:
        raise ValueError(f"Unsupported model name: {model}")

    # Retrieve the constructor function, pretrained weights, and hidden dim for the model
    fn, weights_enum, hidden_dim = model_map[model]

    # Prepare kwargs for model constructor — only add dropout if specified (including 0.0)        
    kwargs = {}
    if dropout is not None:
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be None or a float between 0 and 1")
        kwargs['dropout'] = dropout

    # Instantiate the model with pretrained weights and optional dropout
    model = fn(weights=weights_enum, **kwargs).to(device)

    # Freeze or unfreeze model parameters according to 'freeze' flag
    for param in model.parameters():
        param.requires_grad = not freeze

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Replace the classification head with a new one matching the desired number of classes
    replace_classifier(model, hidden_dim, num_classes, device)

    return model


