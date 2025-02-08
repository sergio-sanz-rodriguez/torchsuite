"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from collections import defaultdict
from typing import List, Tuple, Optional, Union
from PIL import Image


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        train_transform: transforms.Compose, 
        test_transform: transforms.Compose,
        batch_size: int, 
        num_train_samples: int = None, 
        num_test_samples: int = None,
        num_workers: int=NUM_WORKERS
    ):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        train_transform: torchvision transforms to perform on training data.
        test_transform: torchvision transforms to perform on test data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_train_samples: Number of samples to include in the training dataset (None for all samples).
        num_test_samples: Number of samples to include in the test dataset (None for all samples).
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    # Resample training data if num_train_samples is specified
    if num_train_samples is not None:
        if num_train_samples > len(train_data):
            # Oversample by repeating indices
            indices = random.choices(range(len(train_data)), k=num_train_samples)
        else:
            # Undersample by selecting a subset of indices
            indices = random.sample(range(len(train_data)), k=num_train_samples)
        train_data = Subset(train_data, indices)

    # Resample testing data if num_test_samples is specified
    if num_test_samples is not None:
        if num_test_samples > len(test_data):
            # Oversample by repeating indices
            indices = random.choices(range(len(test_data)), k=num_test_samples)
        else:
            # Undersample by selecting a subset of indices
            indices = random.sample(range(len(test_data)), k=num_test_samples)
        test_data = Subset(test_data, indices)

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, #enables fast data transfer to CUDA-enable GPU
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True, #enables fast data transfer to CUDA-enable GPU
    )

    return train_dataloader, test_dataloader, class_names

def resample_data(data, labels, samples_0=None, samples_1=None):
    """
    Resamples the dataset by specifying the number of samples for class 0 and class 1.
    
    Args:
        data: The dataset (e.g., ImageFolder).
        labels: The labels corresponding to the dataset.
        samples_0: The desired number of samples for class 0 (can be None to keep all).
        samples_1: The desired number of samples for class 1 (can be None to keep all).
        
    Returns:
        A subset of the dataset with the specified number of samples for each class.
    """
    # Create a DataFrame for easier indexing
    df = pd.DataFrame({"index": range(len(labels)), "label": labels})
    sampled_indices = []
    
    # Process class 0
    indices_0 = df[df["label"] == 0]["index"]
    if samples_0 is not None:
        sampled_indices.extend(
            np.random.choice(indices_0, size=samples_0, replace=(len(indices_0) < samples_0))
        )
    else:
        sampled_indices.extend(indices_0)
    
    # Process class 1
    indices_1 = df[df["label"] == 1]["index"]
    if samples_1 is not None:
        sampled_indices.extend(
            np.random.choice(indices_1, size=samples_1, replace=(len(indices_1) < samples_1))
        )
    else:
        sampled_indices.extend(indices_1)
    
    # Shuffle sampled indices for randomness
    np.random.shuffle(sampled_indices)
    
    # Return a subset of the dataset
    return torch.utils.data.Subset(data, sampled_indices)


def create_dataloaders_with_resampling(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    train_class_0_samples: int = None,
    train_class_1_samples: int = None,
    test_class_0_samples: int = None,
    test_class_1_samples: int = None,
    num_workers: int = 0,
    combine_train_test: bool = False,
):
    """
    Creates training and testing DataLoaders with optional class balancing. It works for binary classification.

    Args:
        train_dir (str): Path to the training directory.
        test_dir (str): Path to the testing directory.
        train_transform (transforms.Compose): Transformations for training data.
        test_transform (transforms.Compose): Transformations for test data.
        batch_size (int): Number of samples per batch in DataLoaders.
        train_class_0_samples (int, optional): Target number of samples for class 0 in training data.
        train_class_1_samples (int, optional): Target number of samples for class 1 in training data.
        test_class_0_samples (int, optional): Target number of samples for class 0 in testing data.
        test_class_1_samples (int, optional): Target number of samples for class 1 in testing data.
        num_workers (int): Number of subprocesses for data loading.
        combine_train_test (bool): Whether to combine train and test into a single DataLoader.

    Returns:
        tuple: (train_dataloader, test_dataloader, class_names) 
               or (combined_dataloader, class_names) if combine_train_test=True.
    """
    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Extract labels
    train_labels = train_data.targets
    test_labels = test_data.targets

    # Get class names
    class_names = train_data.classes

    # Resample data
    train_data_resampled = resample_data(train_data, train_labels, train_class_0_samples, train_class_1_samples)
    test_data_resampled = resample_data(test_data, test_labels, test_class_0_samples, test_class_1_samples)

    # Combine datasets if required
    if combine_train_test:
        combined_data = torch.utils.data.ConcatDataset([train_data_resampled, test_data_resampled])
        combined_dataloader = DataLoader(
            combined_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return combined_dataloader, class_names

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data_resampled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data_resampled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


class CustomImageDataset(Dataset):
    
    """
    Custom dataset for loading images from a list of paths.

    Args:
        image_paths (List[str]): List of file paths to images.
        labels (List[int]): List of corresponding labels.
        transform (transforms.Compose, optional): Transformations to apply.
    """

    def __init__(
            self,
            image_paths: List[str],
            labels: Union[List[int], List[str]],
            transform: Optional[transforms.Compose] = None
            ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

def create_dataloaders_list(
    train_image_paths: List[str],
    train_labels: Union[List[int], List[str]],
    val_image_paths: List[str],
    val_labels: Union[List[int], List[str]],
    test_image_paths: List[str],
    test_labels: Union[List[int], List[str]],
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 0,
):
    """
    Creates training and testing DataLoaders with optional class balancing using image paths instead of directories.
    Instead a folder, lists of image paths and labels are passed. It is usefull to select a subset of samples in the
    folder.

    Args:
        train_image_paths (List[str]): List of training image file paths.
        train_labels (List[int]): List of corresponding labels for training images.
        test_image_paths (List[str]): List of test image file paths.
        test_labels (List[int]): List of corresponding labels for test images.
        train_transform (transforms.Compose): Transformations for training data.
        test_transform (transforms.Compose): Transformations for test data.
        batch_size (int): Number of samples per batch in DataLoaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_dataloader, test_dataloader, class_names) 
               or (combined_dataloader, class_names) if combine_train_test=True.
    """
    # Create datasets
    train_data = CustomImageDataset(train_image_paths, train_labels, transform=train_transform)
    val_data = CustomImageDataset(val_image_paths, val_labels, transform=val_transform)
    test_data = CustomImageDataset(test_image_paths, test_labels, transform=test_transform)

    # Get class names
    class_names = list(set(train_labels))  # Unique class labels

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names



def create_dataloaders_vit(
        vit_model: str = "vit_b_16_224",
        train_dir: str = "./",
        test_dir: str = "./",
        batch_size: int = 64,
        num_train_samples: int = None, 
        num_test_samples: int = None,
        aug: bool = True,
        num_workers: int = os.cpu_count()
    ):

    """
    Creates data loaders for training and testing datasets tailored for Vision Transformers
    according to https://pytorch.org/vision/main/models/vision_transformer.html. Applies 
    necessary preprocessing transformations, including optional data augmentation. 
    using v2.TrivialAugmentWide().

    Args:
        vit_model (str): The ViT model variant to use. Default is "vit_b_16_224".
            Options:
            - 'vit_b_16_224': ViT-Base/16-224
            - 'vit_b_16_384': ViT-Base/16-384
            - 'vit_b_32_224': ViT-Base/32-224
            - 'vit_l_16_224': ViT-Large/16-224
            - 'vit_l_16_384': ViT-Large/16-384
            - 'vit_l_32_224': ViT-Large/32-224
            - 'vit_h_14_224': ViT-Huge/14-224
            - 'vit_h_14_518': ViT-Huge/14-518
        train_dir (str): Path to the training dataset directory. Default is "./".
        test_dir (str): Path to the test dataset directory. Default is "./".
        batch_size (int): Batch size for the data loaders. Default is 64.
        num_train_samples (int, optional): Number of samples to include in the training dataset (None for all samples).
        num_test_samples (int, optional): Number of samples to include in the test dataset (None for all samples).
        aug (bool): Whether to apply data augmentation. Default is True.
        num_workers (int): Number of workers for data loading. Default is os.cpu_count().

    Returns:
        tuple:
            train_dataloader (torch.utils.data.DataLoader): Data loader for the training dataset.
            test_dataloader (torch.utils.data.DataLoader): Data loader for the test dataset.
            class_names (list): List of class names in the dataset.
    """

    # Mapping ViT model names to image sizes (resize and crop)
    vit_model_sizes = {
        'vit_b_16_224': (256, 224),
        'vit_b_16_384': (384, 384),
        'vit_b_32_224': (256, 224),
        'vit_l_16_224': (242, 224),
        'vit_l_16_384': (512, 512),
        'vit_l_32_224': (256, 224),
        'vit_h_14_224': (224, 224),
        'vit_h_14_518': (518, 518)
    }

    # Validate model selection
    if vit_model not in vit_model_sizes:
        raise ValueError(f"[ERROR] Invalid ViT model '{vit_model}'. Available options: {list(vit_model_sizes.keys())}")

    # Get image sizes for the selected ViT model
    IMG_SIZE_RESIZE, IMG_SIZE_CROP = vit_model_sizes[vit_model]

    # Define training transformations
    if aug:
        train_transforms = v2.Compose([
            v2.TrivialAugmentWide(),
            v2.Resize((IMG_SIZE_RESIZE, IMG_SIZE_RESIZE)),
            v2.RandomCrop((IMG_SIZE_CROP, IMG_SIZE_CROP)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = v2.Compose([
            v2.Resize((IMG_SIZE_RESIZE, IMG_SIZE_RESIZE)),
            v2.CenterCrop((IMG_SIZE_CROP, IMG_SIZE_CROP)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Define test transformations
    test_transforms = v2.Compose([
        v2.Resize((IMG_SIZE_RESIZE, IMG_SIZE_RESIZE)),
        v2.CenterCrop((IMG_SIZE_CROP, IMG_SIZE_CROP)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transforms,
        test_transform=test_transforms,
        batch_size=batch_size,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names


def create_dataloaders_swin(
    swin_model: str = "swin_t",
    train_dir: str = "./train",
    test_dir: str = "./test",
    batch_size: int = 64,
    num_train_samples: int = None,
    num_test_samples: int = None,
    aug: bool = True,
    num_workers: int = os.cpu_count()
):
    """
    Creates data loaders for training and testing datasets tailored for Swin Transformers
    according to https://pytorch.org/vision/main/models/swin_transformer.html. Applies 
    necessary preprocessing transformations, including optional data augmentation with
    v2.TrivialAugmentWide().

    Args:
        swin_model (str): The specific Swin Transformer model to use. Default is "swin_t".
            Options include:
            - "swin_t_224": Swin-Tiny
            - "swin_s_224": Swin-Small
            - "swin_b_224": Swin-Base
            - "swin_v2_t_256": Swin-V2-Tiny
            - "swin_v2_s_256": Swin-V2-Small
            - "swin_v2_b_256": Swin-V2-Base
        train_dir (str): Path to the training dataset directory. Default is "./train".
        test_dir (str): Path to the testing dataset directory. Default is "./test".
        batch_size (int): Batch size for the data loaders. Default is 64.
        num_train_samples (int, optional): Number of samples to include in the training dataset (None for all samples).
        num_test_samples (int, optional): Number of samples to include in the testing dataset (None for all samples).
        aug (bool): Whether to apply data augmentation. Default is True.
        num_workers (int): Number of subprocesses to use for data loading. Default is the number of CPU cores.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (torch.utils.data.DataLoader): Data loader for the training dataset.
            - test_dataloader (torch.utils.data.DataLoader): Data loader for the testing dataset.
            - class_names (list): List of class names.
    """

    # Define input size and normalization parameters based on the model
    model_params = {
        'swin_t_224': (232, 224),
        'swin_s_224': (246, 224),
        'swin_b_224': (238, 224),
        'swin_v2_t_256': (260, 256),
        'swin_v2_s_256': (260, 256),
        'swin_v2_b_256': (272, 256)
    }

    if swin_model not in model_params:
        raise ValueError(f"[ERROR] The specified model '{swin_model}' is not among the supported options.")

    input_size = model_params[swin_model]['input_size']

    # Get image sizes for the selected ViT model
    IMG_SIZE_RESIZE, IMG_SIZE_CROP = vit_model_sizes[vit_model]

    # Define training transformations
    if aug:
        train_transforms = v2.Compose([
            v2.TrivialAugmentWide(),
            v2.Resize((IMG_SIZE_RESIZE, IMG_SIZE_RESIZE)),
            v2.RandomCrop((IMG_SIZE_CROP, IMG_SIZE_CROP)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = v2.Compose([
            v2.Resize((IMG_SIZE_RESIZE, IMG_SIZE_RESIZE)),
            v2.CenterCrop((IMG_SIZE_CROP, IMG_SIZE_CROP)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Define test transformations
    test_transforms = v2.Compose([
        v2.Resize((IMG_SIZE_RESIZE, IMG_SIZE_RESIZE)),
        v2.CenterCrop((IMG_SIZE_CROP, IMG_SIZE_CROP)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transforms,
        test_transform=test_transforms,
        batch_size=batch_size,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names
