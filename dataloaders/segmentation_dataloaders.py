"""
Contains functionality for creating PyTorch DataLoaders for object dectection.
"""

import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as Ftv
from PIL import Image
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F

# Custom zoom out function
def random_zoom_out(img, mask, scale_factor):
    # Get the original size
    width, height = img.size

    # Apply scaling to image (scale down for zoom-out)
    new_width, new_height = int(width / scale_factor), int(height / scale_factor)
    
    # Resize both image and mask to the zoomed-out size
    img_resized = Ftv.resize(img, (new_height, new_width), interpolation=Ftv.InterpolationMode.BICUBIC)
    mask_resized = Ftv.resize(mask, (new_height, new_width), interpolation=Ftv.InterpolationMode.NEAREST)

    # Pad the resized image and mask back to the original size
    img_resized = Ftv.pad(img_resized, (0, 0, width - new_width, height - new_height), fill=0)
    mask_resized = Ftv.pad(mask_resized, (0, 0, width - new_width, height - new_height), fill=0)

    return img_resized, mask_resized


# Function to apply transformations on both images and masks
def segmentation_transforms(
        train=True,
        img_size=(512, 512),
        mean_std_norm=True,
        use_trivial_augment=False
        ):
    
    """
    Returns a transform function that applies several image augmentations and pre-processing steps 
    to both images and their corresponding targets (including masks and bounding boxes).
    
    Args:
        train (bool): Whether the transformation is for training or evaluation. Default is True.
        img_size (tuple): The target size to resize the image to, should be a tuple (H, W). Default is (512, 512).
        mean_std_norm (bool): Whether to normalize the image using the mean and standard deviation of ImageNet. Default is True.
        use_trivial_augment (bool): Whether to apply trivial augmentation or not. Default is False.
    
    Returns:
        transform (function): A function that applies transformations to both the image and its corresponding target.
    
    Raises:
        ValueError: If `img_size` is not a tuple or does not contain two integers.
        TypeError: If the input parameters are of an incorrect type.
    """    

    # Validate parameters
    if not isinstance(img_size, tuple) or len(img_size) != 2 or not all(isinstance(i, int) for i in img_size):
        raise ValueError("img_size must be a tuple of two integers representing (H, W).")
    
    if not isinstance(train, bool):
        raise TypeError("train must be a boolean value.")
    
    if not isinstance(mean_std_norm, bool):
        raise TypeError("mean_std_norm must be a boolean value.")
    
    if not isinstance(use_trivial_augment, bool):
        raise TypeError("use_trivial_augment must be a boolean value.")

    def transform(img, target):
        # Ensure img is a PIL image at the start
        if not isinstance(img, Image.Image):
            img = Ftv.to_pil_image(img)

        # Resize both image and mask to the same size
        img = Ftv.resize(img, (512, 512), interpolation=Ftv.InterpolationMode.BILINEAR)
        target["masks"] = Ftv.resize(target["masks"], (512, 512), interpolation=Ftv.InterpolationMode.NEAREST)

        if train:
            if use_trivial_augment:
                img = T.TrivialAugmentWide(num_magnitude_bins=31)(img)
                # No direct trivial augment for masks, so they are left unchanged.
            else:
                # Apply spatial transformations to both image and mask
                if torch.rand(1) < 0.5:
                    img = Ftv.hflip(img)
                    target["masks"] = Ftv.hflip(target["masks"])
                
                if torch.rand(1) < 0.5:
                    img = Ftv.vflip(img)
                    target["masks"] = Ftv.vflip(target["masks"])
                
                if torch.rand(1) < 0.5:
                    angle = torch.randint(-20, 20, (1,)).item()
                    img = Ftv.rotate(img, angle)
                    target["masks"] = Ftv.rotate(target["masks"], angle)
                
                if torch.rand(1) < 0.5:
                    img = Ftv.gaussian_blur(img, kernel_size=3, sigma=(0.1, 2.0))
                
                if torch.rand(1) < 0.2:
                    # Custom zoom-out that applies to both image and mask in sync
                    scale_factor = torch.rand(1).item() * 1.0 + 1.0  # Random zoom factor from 1x to 2x
                    img, target['masks'] = random_zoom_out(img, target['masks'], scale_factor)

                # Color-based augmentations only for the image
                img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)

        # Convert image to tensor only if it is still a PIL image
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)

        if mean_std_norm:
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, target  # Return both transformed image and modified target

    return transform


class SegmentationTransforms:
    
    """
    A class to apply transformations on both images and masks for segmentation tasks.

    Args:
        train (bool): Whether the transformation is for training or evaluation. Default is True.
        img_size (tuple): The target size to resize the image to, should be a tuple (H, W). Default is (512, 512).
        mean_std_norm (bool): Whether to normalize the image using the mean and standard deviation of ImageNet. Default is True.
        use_trivial_augment (bool): Whether to apply trivial augmentation or not. Default is False.

    Raises:
        ValueError: If `img_size` is not a tuple or does not contain two integers.
        TypeError: If the input parameters are of an incorrect type.
    """

    def __init__(self, train=True, img_size=(512, 512), mean_std_norm=True, use_trivial_augment=False):
        # Validate parameters
        if not isinstance(img_size, tuple) or len(img_size) != 2 or not all(isinstance(i, int) for i in img_size):
            raise ValueError("img_size must be a tuple of two integers representing (H, W).")
        
        if not isinstance(train, bool):
            raise TypeError("train must be a boolean value.")
        
        if not isinstance(mean_std_norm, bool):
            raise TypeError("mean_std_norm must be a boolean value.")
        
        if not isinstance(use_trivial_augment, bool):
            raise TypeError("use_trivial_augment must be a boolean value.")

        self.train = train
        self.img_size = img_size
        self.mean_std_norm = mean_std_norm
        self.use_trivial_augment = use_trivial_augment

    def __call__(self, img, target):
        """
        Apply transformations to the given image and its corresponding target.

        Args:
            img (PIL.Image or Tensor): The input image.
            target (dict): The corresponding target containing segmentation masks.

        Returns:
            Tuple[Tensor, dict]: The transformed image and updated target.
        """

        # Ensure img is a PIL image at the start
        if not isinstance(img, Image.Image):
            img = Ftv.to_pil_image(img)

        # Resize both image and mask to the same size
        img = Ftv.resize(img, self.img_size, interpolation=Ftv.InterpolationMode.BILINEAR)
        target["masks"] = Ftv.resize(target["masks"], self.img_size, interpolation=Ftv.InterpolationMode.NEAREST)

        if self.train:
            if self.use_trivial_augment:
                img = T.TrivialAugmentWide(num_magnitude_bins=31)(img)
                # No direct trivial augment for masks, so they are left unchanged.
            else:
                # Apply spatial transformations to both image and mask
                if torch.rand(1) < 0.5:
                    img = Ftv.hflip(img)
                    target["masks"] = Ftv.hflip(target["masks"])
                
                if torch.rand(1) < 0.5:
                    img = Ftv.vflip(img)
                    target["masks"] = Ftv.vflip(target["masks"])
                
                if torch.rand(1) < 0.5:
                    angle = torch.randint(-20, 20, (1,)).item()
                    img = Ftv.rotate(img, angle)
                    target["masks"] = Ftv.rotate(target["masks"], angle)
                
                if torch.rand(1) < 0.5:
                    img = Ftv.gaussian_blur(img, kernel_size=3, sigma=(0.1, 2.0))
                
                if torch.rand(1) < 0.2:
                    # Custom zoom-out that applies to both image and mask in sync
                    scale_factor = torch.rand(1).item() * 1.0 + 1.0  # Random zoom factor from 1x to 2x
                    img, target['masks'] = self.random_zoom_out(img, target['masks'], scale_factor)

                # Color-based augmentations only for the image
                img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)

        # Convert image to tensor only if it is still a PIL image
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)

        if self.mean_std_norm:
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, target  # Return both transformed image and modified target

    @staticmethod
    def random_zoom_out(img, mask, scale_factor):
        """
        Performs random zoom out transformation on both the image and mask.

        Args:
            img (PIL.Image): Input image.
            mask (PIL.Image): Corresponding segmentation mask.
            scale_factor (float): Factor by which to zoom out.

        Returns:
            Tuple[PIL.Image, PIL.Image]: Zoomed-out image and mask.
        """
        # Placeholder function, should be implemented based on dataset requirements
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
        transforms (callable, optional): Optional transformations to apply to the images and targets.
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
        self.num_classes = len(class_dictionary)  # Max num of classes

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
                - 'classes': Mapping {channel_index: class_name}
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

        # Get unique object ids ignoring background (0)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        #num_objs = len(obj_ids)

        # Map ids in class_dictionary to channel indexes for the one-hot mask (one class per channel)
        class_mapping = {orig_id: channel for channel, (orig_id, _) in enumerate(self.class_dictionary.items())}

        # Initialize the one-hot encoded mask
        H, W = mask.shape[-2:]
        multi_channel_mask = torch.zeros((self.num_classes, H, W), dtype=torch.uint8)

        # Fill the one-hot mask with class indexes
        class_map = {}
        #boxes = []
        for obj_id in obj_ids:

            # Get the channel index
            ch_idx = class_mapping.get(obj_id.item(), None)
            if ch_idx is not None:

                # Map object ID to a class index
                multi_channel_mask[ch_idx] = (mask == obj_id).to(dtype=torch.uint8)  # Zero-based index
               
                # Store class name mapping for each object in the mask
                class_map[ch_idx] = self.class_dictionary[obj_id.item()]  # Store class name mapping

                # Get the bounding box for the current mask
                #obj_mask = multi_channel_mask[ch_idx].unsqueeze(0)
                #box = masks_to_boxes(obj_mask)
                #boxes.append(box)

        # Convert boxes to a tensor
        #boxes = torch.cat(boxes, dim=0) if len(boxes) > 0 else torch.empty(0, 4)

        # There is only one class (the remapped class)
        #labels = torch.arange(1, num_objs + 1, dtype=torch.int64)

        # Compute area
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # And image id
        #image_id = idx

        # Suppose all instances are not crowd
        #iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        # Wrap sample and targets into torchvision tv_tensors:
        target = {}
        #target["boxes"] = boxes
        target["masks"] = multi_channel_mask  # Store multi-class mask
        target["class_map"] = class_map
        #target["labels"] = labels
        target["image_id"] = idx
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        
        # Apply transformations
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
