"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from collections import defaultdict

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
        pin_memory=True, #enables fast data transfre to CUDA-enable GPU
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True, #enables fast data transfre to CUDA-enable GPU
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

    Returns:
        tuple: train_dataloader, test_dataloader, class_names
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


def create_dataloaders_for_vit(
        vit_model: str="bitbase16",
        train_dir: str="./",
        test_dir: str="./",
        batch_size: int=64,
        num_train_samples: int = None, 
        num_test_samples: int = None,
        aug: bool=True,
        num_workers: int=os.cpu_count()
        ):
    
    """
    Creates data loaders for the training and test datasets to be used to traing visiton transformers.

    Args:
        vit_model (str): The name of the ViT model to use. Default is "bitbase16".
            -bitbase16: ViT-Base/16-224
            -bitbase16_2: ViT-Base/16-384
            -bitlarge16: ViT-Large/16-224
            -bitlarge32: ViT-Large/32-224
        train_dir (str): The path to the training dataset directory. Default is TRAIN_DIR.
        test_dir (str): The path to the test dataset directory. Default is TEST_DIR.
        batch_size (int): The batch size for the data loaders. Default is BATCH_SIZE.
        num_train_samples: Number of samples to include in the training dataset (None for all samples).
        num_test_samples: Number of samples to include in the test dataset (None for all samples).
        aug (bool): Whether to apply data augmentation or not. Default is True.
        display_imgs (bool): Whether to display sample images or not. Default is True.

    Returns:
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
        class_names (list): A list of class names.
    """

    IMG_SIZE = 224
    IMG_SIZE_2 = 384
    IMG_SIZE_3 = 518

    # Manual transforms for the training dataset
    manual_transforms = v2.Compose([           
        v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),    
    ])

    # ViT-Base/16 transforms
    if vit_model == "vitbase16":

        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitb = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((256)),
                v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitb = v2.Compose([
                v2.Resize((256)),
                v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitb = v2.Compose([    
            v2.Resize((256)),
            v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Base
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitb,
            test_transform=manual_transforms_test_vitb,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
            )
    
    if vit_model == "vitbase16_2":

        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitb = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((384)),
                v2.CenterCrop((IMG_SIZE_2, IMG_SIZE_2)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitb = v2.Compose([
                v2.Resize((384)),
                v2.CenterCrop((IMG_SIZE_2, IMG_SIZE_2)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitb = v2.Compose([    
            v2.Resize((384)),
            v2.CenterCrop((IMG_SIZE_2, IMG_SIZE_2)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Base
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitb,
            test_transform=manual_transforms_test_vitb,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
            )

    # ViT-Large/16 transforms
    elif vit_model == "vitlarge16":

        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitl = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((242)),
                v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitl = v2.Compose([
                v2.Resize((242)),
                v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitl = v2.Compose([    
            v2.Resize((242)),
            v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Large/16
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitl,
            test_transform=manual_transforms_test_vitl,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
        )

    # ViT-Large/32 transforms
    elif vit_model == "vitlarge32":
        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitl = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((256)),
                v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitl = v2.Compose([
                v2.Resize((256)),
                v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitl = v2.Compose([    
            v2.Resize((256)),
            v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Large/32
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitl,
            test_transform=manual_transforms_test_vitl,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
        )
    # Vit-Huge/14 transforms
    else:
        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitl = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((518)),
                v2.RandomCrop((IMG_SIZE_3, IMG_SIZE_3)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitl = v2.Compose([
                v2.Resize((518)),
                v2.CenterCrop((IMG_SIZE_3, IMG_SIZE_3)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitl = v2.Compose([    
            v2.Resize((518)),
            v2.CenterCrop((IMG_SIZE_3, IMG_SIZE_3)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Large/32
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitl,
            test_transform=manual_transforms_test_vitl,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
        )

    return train_dataloader, test_dataloader, class_names
