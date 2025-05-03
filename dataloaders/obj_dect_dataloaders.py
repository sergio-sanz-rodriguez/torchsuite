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
        Initializes the dataset with image and mask paths and transformations. Bounding boxes are created
        from the mask images.

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


# Implement a class that processes the database
class ProcessDatasetYOLO(torch.utils.data.Dataset):

    """
    Custom PyTorch Dataset class for loading images and their corresponding 
    YOLO-formatted label files, converting the labels to XYXY format, and returning 
    the data in a format compatible with torchvision detection models.
    
    Args:
        root (str): Root directory containing the image and label folders.
        image_path (str): Subdirectory within root where images are stored.
        label_path (str): Subdirectory within root where label files are stored.
        transforms (callable, optional): A function/transform to apply to both image and target.
        num_classes (int): Number of classes (default is 1 for binary detection).
    """

    def __init__(self, root, image_path, label_path, transforms=None, num_classes=1):
        self.root = root
        self.image_path = image_path
        self.label_path = label_path
        self.transforms = transforms
        self.num_classes = num_classes

        # Load all image and label files, ensuring alignment
        self.imgs = sorted(os.listdir(os.path.join(root, image_path)))
        self.labels = sorted(os.listdir(os.path.join(root, label_path)))

    def yolo_to_xyxy(self, label_path, img_width, img_height):

        """
        Converts YOLO-format bounding boxes to XYXY format (COCO-style).

        Args:
            label_path (str): Path to the label file.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            Tuple[Tensor, Tensor]: Bounding boxes in both YOLO and XYXY formats.
        """
        
        bboxes_yolo = []
        bboxes_xyxy = []
        with open(label_path, "r") as file:
            for line in file:
                data = line.strip().split()
                class_id = int(data[0])
                x_centre, y_centre, width, height = map(float, data[1:])

                # Convert YOLO to COCO format
                x_min = (x_centre - width / 2) * img_width
                y_min = (y_centre - height / 2) * img_height
                x_max = (x_centre + width / 2) * img_width
                y_max = (y_centre + height / 2) * img_height
                
                bboxes_yolo.append((class_id, x_centre, y_centre, width, height))
                bboxes_xyxy.append((class_id, x_min, y_min, x_max, y_max))

        return torch.tensor(bboxes_yolo, dtype=torch.float32), torch.tensor(bboxes_xyxy, dtype=torch.float32)

    def __getitem__(self, idx):

        # Load images and labels
        img_path = os.path.join(self.root, self.image_path, self.imgs[idx])
        label_path = os.path.join(self.root, self.label_path, self.labels[idx])

        # Read image
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        # Image dimensions
        img_height, img_width = img.shape[1], img.shape[2]

        # Get bounding boxes
        bboxes_yolo, bboxes_xyxy = self.yolo_to_xyxy(label_path, img_width, img_height)

        # There is only one class
        num_objs = bboxes_xyxy.size(0)

        # Calculate labels
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Ensure bounding boxes exist
        if num_objs > 0:
            bboxes_xyxy = bboxes_xyxy[:, 1:]  # Remove class_id
            xmin, ymin, xmax, ymax = bboxes_xyxy.unbind(dim=1)
            bboxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

            bboxes_yolo = bboxes_yolo[:, 1:]
            xc, yc, w, h = bboxes_yolo.unbind(dim=1)
            bboxes_yolo = torch.stack([xc, yc, w, h], dim=1)
        else:
            # Handle empty cases
            bboxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            bboxes_yolo = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        # Create the target
        target = {
            "boxes":      tv_tensors.BoundingBoxes(bboxes_xyxy, format="XYXY", canvas_size=F.get_size(img)),
            "labels":     labels,
        }

        # Preprocess image and target
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
