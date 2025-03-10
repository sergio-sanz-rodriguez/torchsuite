"""
Contains functionality for creating PyTorch DataLoaders for object dectection.
"""

import os
import sys
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as Ftv
from PIL import Image
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F

sys.path.append(os.path.abspath("../engines"))
from engines.common import Logger

class SegmentationTransforms(Logger):

    """
    A class to apply transformations on both images and masks for segmentation tasks.

    Args:
        train (bool): Whether the transformation is for training or evaluation. Default is True.
        img_size (tuple): The mask size to resize the image to, should be a tuple (H, W). Default is (512, 512).
        mean_std_norm (bool): Whether to normalize the image using the mean and standard deviation of ImageNet. Default is True.
        augment_magnitude (int): Controls the magnitude of augmentations, from 1 (weak) to 5 (strong). Default is 3.

    Raises:
        ValueError: If `img_size` is not a tuple of two integers.
        TypeError: If the input parameters are of an incorrect type.
    """

    def __init__(
        self,
        train=True,
        img_size=(512, 512),
        mean_std_norm=True, 
        augment_magnitude=3
        ):
        
        if not isinstance(img_size, tuple) or len(img_size) != 2 or not all(isinstance(i, int) for i in img_size):
            self.error("img_size must be a tuple of two integers representing (H, W).")
        
        if not isinstance(train, bool):
            self.error("train must be a boolean value.")
        
        if not isinstance(mean_std_norm, bool):
            self.error("mean_std_norm must be a boolean value.")
        
        if not isinstance(augment_magnitude, int) or not (1 <= augment_magnitude <= 5):
            self.error("augment_magnitude must be an integer between 1 and 5.")

        self.train = train
        self.img_size = img_size
        self.mean_std_norm = mean_std_norm
        self.augment_magnitude = augment_magnitude

    def __call__(self, img, mask):

        """
        Apply transformations to the given image and its corresponding mask.

        Args:
            img (PIL.Image or Tensor): The input image.
            mask (Tensor): The corresponding segmentation mask.

        Returns:
            Tuple[Tensor, Tensor]: The transformed image and updated mask.
        """

        # Ensure img is a PIL image
        if not isinstance(img, Image.Image):
            img = Ftv.to_pil_image(img)

        # Resize both image and mask
        img = Ftv.resize(img, self.img_size, interpolation=Ftv.InterpolationMode.BILINEAR)
        mask = Ftv.resize(mask, self.img_size, interpolation=Ftv.InterpolationMode.NEAREST)

        if self.train:
            
            # Define augmentation intensity based on augment_magnitude
            angle_range =  5 * self.augment_magnitude           # Rotation range: (-5, 5) to (-25, 25)
            blur_sigma =   (0.1, 0.3 * self.augment_magnitude)  # Blur intensity: (0.1, 0.3) to (0.1, 1.5)
            zoom_factor =  1.0 + (0.1 * self.augment_magnitude) # Zoom-out factor: 1.1 to 1.5
            color_jitter = 0.1 * self.augment_magnitude         # Color jitter intensity: 0.05 to 0.25
            dropout_size = 0.04* self.augment_magnitude         # Dropout size: 0.04 to 0.2

            # Horizontal flip
            if torch.rand(1) < 0.5:
                img = Ftv.hflip(img)
                mask = Ftv.hflip(mask)
            
            # Vertical flip
            if torch.rand(1) < 0.3:
                img = Ftv.vflip(img)
                mask = Ftv.vflip(mask)
            
            # Rotation
            if torch.rand(1) < 0.3:
                angle = torch.randint(-angle_range, angle_range, (1,)).item()
                img = Ftv.rotate(img, angle)
                mask = Ftv.rotate(mask, angle)
            
            # Gaussian blur for image only
            if torch.rand(1) < 0.2:
                img = Ftv.gaussian_blur(img, kernel_size=3, sigma=blur_sigma)
            
            # Random zoom out
            if torch.rand(1) < 0.3:
                img, mask = self.random_zoom_out(img, mask, scale_factor=zoom_factor)
        
            # Color augmentations for image only
            img = T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter)(img)

            # Random grayscale transformation
            if torch.rand(1) < 0.2:  # You can adjust the probability of applying grayscale
                img = Ftv.rgb_to_grayscale(img, num_output_channels=3)

            # Apply coarse dropout
            if torch.rand(1) < 0.2:
                img, mask = self.coarse_dropout(img, mask, dropout_size=dropout_size)

        # Convert image to tensor if necessary
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)

        # Pixel normalization using Imagenet values
        if self.mean_std_norm:
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, mask


    @staticmethod
    # Custom zoom out function
    def random_zoom_out(img, mask, scale_factor):
        
        # Get the original size
        width, height = img.size

        # Apply scaling to image
        new_width, new_height = int(width / scale_factor), int(height / scale_factor)
        
        # Resize both image and mask
        img_resized = Ftv.resize(img, (new_height, new_width), interpolation=Ftv.InterpolationMode.BICUBIC)
        mask_resized = Ftv.resize(mask, (new_height, new_width), interpolation=Ftv.InterpolationMode.NEAREST)

        # Calculate the random padding
        padding_left = random.randint(0, width - new_width)
        padding_top = random.randint(0, height - new_height)
        padding_right = width - new_width - padding_left
        padding_bottom = height - new_height - padding_top

        # Pad the resized image and mask back to the original size
        img_resized = Ftv.pad(img_resized, (padding_left, padding_top, padding_right, padding_bottom), fill=0)
        mask_resized = Ftv.pad(mask_resized, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

        return img_resized, mask_resized
    
    @staticmethod
    def coarse_dropout(img, mask, dropout_size=0.2):
        """
        Apply coarse dropout on image and mask.
        Args:
            img (PIL.Image or Tensor): The input image.
            mask (Tensor): The corresponding segmentation mask.
            dropout_prob (float): The probability to apply dropout.
            dropout_size (float): Size of the dropout region relative to the image size.
        """
        
        height, width = img.size[1], img.size[0]  # height, width of the image
        dropout_height = int(height * dropout_size)
        dropout_width = int(width * dropout_size)
        
        # Randomly select a location for the dropout
        top_left_x = torch.randint(0, width - dropout_width, (1,)).item()
        top_left_y = torch.randint(0, height - dropout_height, (1,)).item()

        # Create dropout mask: a square of zeros (dropout area) on a tensor of ones
        dropout_mask = torch.ones((height, width))
        dropout_mask[top_left_y:top_left_y+dropout_height, top_left_x:top_left_x+dropout_width] = 0  # Dropout area

        # Convert img to a tensor if it is a PIL image
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)

        # Apply the mask to image and mask (only applies dropout on the image)
        img = img * dropout_mask
        mask = mask * dropout_mask
            
        return img, mask


# Implement a class that processes the database
class ProcessDatasetSegmentation(torch.utils.data.Dataset):

    """
    Custom Dataset class for loading and processing image and mask pairs for object detection.
    It supports applying transformations to the data during training.
    """

    def __init__(
            self,
            root,
            image_path,
            mask_path,
            transforms=None,
            class_dictionary=None):

        """
        Initializes the dataset with image and mask paths and transformations.

        Parameters:
        root (str): Root directory where images and masks are stored.
        image_path (str): Relative path to the folder containing image files.
        mask_path (str): Relative path to the folder containing mask files.
        transforms (callable, optional): Optional transformations to apply to the images and masks.
        class_dictionary (dict, optional): Mapping of category IDs to class names.
        """

        self.root = root
        self.image_path = image_path
        self.mask_path = mask_path
        self.transforms = transforms
        
        # Class dictionary mapping category IDs to class names
        if class_dictionary is None:
            raise ValueError("class_dictionary must be provided to define category mappings.")

        self.class_dictionary = class_dictionary
        self.num_classes = len(class_dictionary)

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_path))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_path))))

    def __getitem__(self, idx):

        """
        Retrieves a single sample (image and its corresponding mask) from the dataset.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: (image, mask) where:
            - image (Tensor): Processed image.
            - one-hot mask (Tensor): Processed mask.
        """

        # Load images and masks
        img_path = os.path.join(self.root, self.image_path, self.imgs[idx])
        mask_path = os.path.join(self.root, self.mask_path, self.masks[idx])

        # Read image
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        # Read mask
        mask = read_image(mask_path)

        # Get unique object ids ignoring background (0)
        obj_ids = torch.unique(mask)
        #obj_ids = obj_ids[obj_ids != 0]

        # Map ids in class_dictionary to channel indexes for the one-hot mask (one class per channel)
        class_mapping = {orig_id: channel for channel, (orig_id, _) in enumerate(self.class_dictionary.items())}

        # Initialize the one-hot encoded mask
        H, W = mask.shape[-2:]
        one_hot_mask = torch.zeros((self.num_classes, H, W), dtype=torch.uint8)

        # Fill the one-hot mask with class indexes
        for obj_id in obj_ids:

            # Get the channel index
            ch_idx = class_mapping.get(obj_id.item(), None)
            if ch_idx is not None:

                # Map object ID to a class index
                one_hot_mask[ch_idx] = (mask == obj_id).to(dtype=torch.uint8)  # Zero-based index
        
        # Apply transformations
        if self.transforms is not None:
            img, one_hot_mask = self.transforms(img, one_hot_mask)

        return img, one_hot_mask

    def __len__(self):

        """
        Returns the total number of samples in the dataset.

        Returns:
        int: Number of images (and corresponding masks) in the dataset.
        """

        return len(self.imgs)
