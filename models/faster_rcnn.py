import os
import sys
import torch
import torchvision
from typing import Union
from pathlib import Path
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
sys.path.append(os.path.abspath("../engines"))
from engines.common import Logger

class StandardFasterRCNN(torch.nn.Module):

    """
        Creates a Faster Region-based CNN (RCNN) architecture using pytorch's predefined backbones for R-CNN: 'resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320'.
        More information is found in this link: https://pytorch.org/vision/master/models/faster_rcnn.html
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50", #['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        weights: Union[str, Path] = "DEFAULT",
        hidden_layer: int = 256,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

        """
        Ceates a Faster Region-based CNN (RCNN) architecture using predefined backbones: 'resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320'.
        - num_classes: Number of output classes for detection, excluding background (int). Default is 1.
        - weights: The pretrained weights to load for the backbone (str). Default is "DEFAULT".
        - backbone: Backbone architecture to use. Default is 'resnet50'. List of supported networks order by accuracy-speed tradefoof:
                    1. 'resnet50_v2: very high accuracy, moderate-high speed
                    2. 'resnet50': high accuracy, moderate speed
                    3. 'mobilenet_v3_large': moderate accuracy, very high speed
                    4. 'mobilenet_v3_large_320': moderate-high accuracy, very high speed
                    ['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        - hidden_layer: Number of hidden units for the mask prediction head. Default is 256.
        - device: Target device: GPU or CPU
        """

        super().__init__()

        logger = Logger()
        
        # Check if the specified backbone is available
        backbone_list = ['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        if backbone not in backbone_list:
            logger.error(f"Backbone '{backbone}' not recognized. Using default 'resnet50'.")
        
        if not isinstance(num_classes, int) or num_classes <= 0:
            logger.error(f"'num_classes' must be a positive integer. Using default value of 1.")

        if not isinstance(hidden_layer, int) or hidden_layer <= 0:
            logger.error(f"'hidden_layer' must be a positive integer. Using default value of 256.")

        # Load default pretrained weights if "DEFAULT" or None
        if backbone == 'resnet50':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        elif backbone == 'resnet50_v2':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        elif backbone == 'mobilenet_v3_large':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        else:
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        
        # Replace the classification head (bounding box predictor)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor (for Mask R-CNN models)
        if "maskrcnn" in self.model.__class__.__name__.lower():
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Load custom weights if provided
        if isinstance(weights, (str, Path)) and weights != "DEFAULT":
            weights_path = Path(weights)
            if weights_path.exists() and weights_path.suffix in {".pth", ".pt", ".pkl", ".h5", ".torch"}:
                # Load the custom weights
                checkpoint = torch.load(weights_path, map_location=device)
                # Remove the 'model.' prefix from the keys in the checkpoint
                checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
                # Update the model with the checkpoint's state_dict
                self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"[ERROR] Custom weights path '{weights}' is not valid or does not point to a valid checkpoint file.")

        # Move the model to the specified device
        self.model.to(device)

    def forward(self, images, targets=None):

        """
        Forward pass through the model:
        - images: Input images (tensor or list of tensors).
        - targets: Ground truth targets for training (optional, only needed for training).
        """
        
        return self.model(images, targets)


class CustomFasterRCNN(torch.nn.Module):

    """
        Creates a Faster Region-based CNN (RCNN) architecture using customized configurations, such as the backbone, the anchor, and the ROI pooler configuration
        More information is found in this link: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: torch.nn.Module = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT"),
        anchor_generator: torch.nn.Module = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            ),
        roi_pooler: torch.nn.Module = MultiScaleRoIAlign(
                featmap_names=["0"],
                output_size=7,
                sampling_ratio=2
            ),
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    ):

        """
        Creates a custom Faster Region-based CNN (RCNN) architecture with customizable configurations.
        This includes the choice of backbone, anchor generator, and RoI pooler.
        - num_classes (int): Number of object classes (including background).
        - backbone (torch.nn.Module): Backbone model (default is ResNet50 FPN), weights shall be loaded outside the class.
        - anchor_generator (torch.nn.Module): Custom Anchor Generator for RPN (default setup with sizes and aspect ratios).
        - roi_pooler (torch.nn.Module): Custom RoI pooler configuration (default is MultiScaleRoIAlign with 7x7 output).
        - device (torch.device): Device to run the model on (default is "cuda" if available, otherwise "cpu").
        """

        super().__init__()

        logger = Logger()

        # Check out number of classes
        if not isinstance(num_classes, int):
            logger.error(f"'num_classes' must be an integer.")
        #assert isinstance(num_classes, int), "[ERROR] num_classes must be an integer."
        
        # Extract the feature layer of the backbone
        backbone = backbone.features

        # Determine the number of output channels for the backbone
        # A dummy input tensor is used to infer the output size
        dummy_input = torch.randn(1, 3, 224, 224)
        output = backbone(dummy_input)
        backbone.out_channels = output.size(1)  

        # Create the Faster R-CNN model with the custom backbone, anchor generator, and ROI pooler
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        
        # Move the model to device
        self.model.to(device)

    def forward(self, images, targets=None):

        """
        Forward pass through the model:
        - images: Input images (tensor or list of tensors).
        - targets: Ground truth targets for training (optional, only needed for training).
        """
        
        return self.model(images, targets)


