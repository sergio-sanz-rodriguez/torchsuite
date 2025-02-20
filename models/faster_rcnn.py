import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNN(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50",
        weights: str = "DEFAULT",
        hidden_layer: int = 256,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu" 
    ):
        super().__init__()

        """
        Initialization of the FasterRCNN class:
        - num_classes: Number of output classes for detection, excluding background (int). Default is 1.
        - weights: The pretrained weights to load for the backbone (str). Default is "DEFAULT".
        - backbone: Backbone architecture to use, such as 'resnet50', 'mobilenet_v3_large', etc. Default is 'resnet50'.
        - hidden_layer: Number of hidden units for the mask prediction head. Default is 256.
        - device: Target device: GPU or CPU
        """
        
        # Check if the specified backbone is available
        backbone_list = ['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        assert backbone in backbone_list, f"[ERROR] Backbone '{backbone}' not recognized."

        assert isinstance(num_classes, int), "[ERROR] num_classes must be an integer."
        assert isinstance(hidden_layer, int) and hidden_layer > 0, "[ERROR] hidden_layer must be a positive integer."

        # Create the Faster R-CNN model based on the selected backbone
        if backbone == 'resnet50':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        elif backbone == 'resnet50_v2':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        elif backbone == 'mobilenet_v3_large':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        else:
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

        # Replace the classification head (bounding box predictor)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor (for Mask R-CNN models)
        if "maskrcnn" in self.model.__class__.__name__.lower():
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        self.model.to(device)

    def forward(self, images, targets=None):
        """
        Forward pass through the model:
        - images: Input images (tensor or list of tensors).
        - targets: Ground truth targets for training (optional, only needed for training).
        """
        return self.model(images, targets)