"""
Contains functionality for creating PyTorch DataLoaders for object dectection.
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torchvision.io import read_image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F

# Implement a class that processes the database
class ProcessDataset(torch.utils.data.Dataset):

    """
    Custom Dataset class for loading and processing image and mask pairs for object detection.
    It supports applying transformations to the data during training.
    """

    def __init__(
            self,
            root,
            image_path,
            mask_path,
            transforms,
            num_classes=1):

        """
        Initializes the dataset with image and mask paths and transformations.

        Parameters:
        root (str): Root directory where images and masks are stored.
        image_path (str): Relative path to the folder containing image files.
        mask_path (str): Relative path to the folder containing mask files.
        transforms (callable, optional): Optional transformations to apply to the images and targets.
        num_classes (int): Number of classes excluding the background. 
            If there are more objects ids than num_classes, then the ids are upper clipped to num_classes.
            If num_classes is 1 (default), then it is a binary classification task, such as ROI detection.
        """

        self.root = root
        self.image_path = image_path
        self.mask_path = mask_path
        self.transforms = transforms
        self.num_classes = num_classes

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_path))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_path))))

    def __getitem__(self, idx):

        """
        Retrieves a single sample (image and its corresponding mask) from the dataset.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: (image, target) where:
            - image (Tensor): Processed image.
            - target (dict): A dictionary containing:
                - 'boxes': Bounding box coordinates for each object.
                - 'masks': Binary masks for each object.
                - 'labels': Labels for each object (1 class in this case).
                - 'image_id': Index of the image.
                - 'area': Area of the bounding box for each object.
                - 'iscrowd': Boolean indicating whether an object is crowded.
        """

        # Load images and masks
        img_path = os.path.join(self.root, self.image_path, self.imgs[idx])
        mask_path = os.path.join(self.root, self.mask_path, self.masks[idx])

        # Read image
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        # Read mask
        mask = read_image(mask_path)

        # Instances are encoded as different colors
        obj_ids = torch.unique(mask)
        
        # First id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # Remap objects ids to be sequencital from 1 to num_objs
        id_map = {old_id.item(): new_id + 1 for new_id, old_id in enumerate(obj_ids)}
        remapped_ids = torch.tensor([id_map[i.item()] for i in obj_ids], dtype=torch.int64)

        # Split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # There is only one class
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        #labels = obj_ids
        labels = torch.clamp(remapped_ids, max=self.num_classes)    

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):

        """
        Returns the total number of samples in the dataset.

        Returns:
        int: Number of images (and corresponding masks) in the dataset.
        """

        return len(self.imgs)