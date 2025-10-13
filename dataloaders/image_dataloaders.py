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
from typing import List, Optional, Union
from PIL import Image
from pathlib import Path

NUM_WORKERS = os.cpu_count()

class DualTransformDataset(datasets.ImageFolder):
    def __init__(self, root, transform_aug, transform_noaug, use_aug=True):
        super().__init__(root, transform=transform_aug)
        self.transform_aug = transform_aug
        self.transform_noaug = transform_noaug
        self.use_aug = use_aug

    def set_augmentation(self, use_aug):
        self.use_aug = use_aug

    def __getitem__(self, index):
        #img, label = super().__getitem__(index)
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')

        # Reapply the chosen transform:
        if self.use_aug:
            img = self.transform_aug(image)
        else:
            img = self.transform_noaug(image)
        return img, label

  
def create_classification_dataloaders(
        train_dir: str, 
        test_dir: str, 
        train_transform: transforms.Compose, 
        test_transform: transforms.Compose,
        batch_size: int, 
        num_train_samples: int = None, 
        num_test_samples: int = None,
        num_workers: int=NUM_WORKERS
    ):
    """Creates training and testing DataLoaders for classification.

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
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    # Create custom dataset
    #train_data = DualTransformDataset(root=train_dir, transform_aug=train_transform, transform_noaug=test_transform, use_aug=True)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

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

class DistillationDataset(Dataset):
    def __init__(self, image_folder, transform_student, transform_teacher):
        self.image_folder = image_folder
        self.transform_student = transform_student
        self.transform_teacher = transform_teacher

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        
        # Load image and label from the dataset
        img, label = self.image_folder[idx]

        # Convert Tensor back to PIL Image if necessary (depends on dataset output)
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Generate a per-sample seed so that both transforms see the same randomness.
        seed = torch.randint(0, 2**32, (1,)).item()

        # Save current RNG states so we can restore them later
        # This prevents altering the global RNG state of the worker
        state_py = random.getstate()
        state_torch = torch.get_rng_state()

        # Apply the student transform with the given seed
        random.seed(seed)
        torch.manual_seed(seed)
        img_student = self.transform_student(img)

        # Restore RNG state before doing the teacher transform
        random.setstate(state_py)
        torch.set_rng_state(state_torch)

        # Apply the teacher transform with the same seed
        random.seed(seed)
        torch.manual_seed(seed)
        img_teacher = self.transform_teacher(img)

        return img_student, img_teacher, label
    
def create_distillation_dataloaders(
        train_dir: str, 
        test_dir: str, 
        transform_student_train, 
        transform_teacher_train,
        transform_student_test,
        transform_teacher_test,
        batch_size: int, 
        num_train_samples: int = None, 
        num_test_samples: int = None,
        num_workers: int = NUM_WORKERS
    ):
    """
    Creates DataLoaders for knowledge distillation using DistillationDataset.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform_student_train: Transform for student inputs.
        transform_teacher_train: Transform for teacher inputs.
        transform_student_test: Transform for student test set.
        transform_teacher_test: Transform for student test set.
        batch_size: Batch size for DataLoaders.
        num_train_samples: Optionally limit number of training samples.
        num_test_samples: Optionally limit number of test samples.
        num_workers: Number of DataLoader workers.

    Returns:
        train_dataloader, test_dataloader, class_names
    """

    # Base datasets
    base_train_dataset = datasets.ImageFolder(root=train_dir)
    base_test_dataset = datasets.ImageFolder(root=test_dir)

    # Optional sampling (train)
    if num_train_samples is not None:
        indices = (random.choices if num_train_samples > len(base_train_dataset)
                   else random.sample)(range(len(base_train_dataset)), k=num_train_samples)
        base_train_dataset = Subset(base_train_dataset, indices)

    # Optional sampling (test)
    if num_test_samples is not None:
        indices = (random.choices if num_test_samples > len(base_test_dataset)
                   else random.sample)(range(len(base_test_dataset)), k=num_test_samples)
        base_test_dataset = Subset(base_test_dataset, indices)

    # Distillation-aware training dataset
    train_dataset = DistillationDataset(
        image_folder=base_train_dataset,
        transform_student=transform_student_train,
        transform_teacher=transform_teacher_train
    )

    # Distillation-aware testing dataset
    test_dataset = DistillationDataset(
        image_folder=base_test_dataset,
        transform_student=transform_student_test,
        transform_teacher=transform_teacher_test
    )

    # Extract class names
    class_names = (
        train_dataset.image_folder.classes
        if isinstance(train_dataset.image_folder, datasets.ImageFolder)
        else train_dataset.image_folder.dataset.classes
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names


class RegressionDataset(Dataset):

    """
    PyTorch regression dataset for loading images and their associated scores.

    Args:
        data (pd.DataFrame): DataFrame with at least two columns:
            - 1st column: path or filename of the image (with or without extension)
            - 2nd column: float value to predict.
        image_folder (str | Path, optional): Root directory where image files are stored.
            Used if 'image_path' entries are relative paths.
        transform (callable, optional): Transform applied to each image (e.g., resizing, normalization).
            Should accept and return a PIL image or tensor.

    Returns:
        tuple:
            img (torch.Tensor): Image tensor of shape (C, H, W)
            score (torch.Tensor): Regression target as a float tensor
    """

    def __init__(self, data, image_folder=None, transform=None):
        self.data = data.reset_index(drop=True)
        self.image_folder = Path(image_folder) if image_folder else None
        self.transform = transform

    def __len__(self):

        """
        Returns the total number of samples in the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        
        """
        Retrieve image tensor and corresponding score.
        """

        # Handle tensor indices by converting to list
        if torch.is_tensor(idx):
            idx = idx.item()

        # Rename columsn
        #cols = list(self.data.columns)
        #self.data = self.data.rename(columns={cols[0]: "image_path", cols[1]: "score"})

        # Build image file path
        image_path = Path(self.data.iloc[idx, 0])
        if not image_path.is_absolute() and not image_path.exists():
            if self.image_folder:
                image_path = self.image_folder / image_path
            else:
                raise FileNotFoundError(
                    f"Image '{image_path}' not found and no image_folder was provided."
                )

        # If file does not exist, try to guess common extensions
        if not image_path.exists():
            common_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            found = False
            for ext in common_exts:
                candidate = image_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"Image file not found: {image_path} (also tried extensions {common_exts})"
                )

        # Open image and convert to RGB mode
        img = Image.open(image_path).convert('RGB')

         # Apply transformation pipeline if provided (expects PIL Image input)
        if self.transform:
            img = self.transform(img)
        else:
            # Basic normalization and tensor conversion
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)

        # Retrieve score from the DataFrame and convert to float
        score = torch.tensor(float(self.data.iloc[idx, 1]), dtype=torch.float32)

        return img, score


def create_regression_dataloaders(
        train_data,
        test_data,
        train_image_folder=None,
        test_image_folder=None,
        train_transform=None,
        test_transform=None,
        batch_size: int = 32,
        num_workers: int = 4
):
    """
    Creates PyTorch DataLoaders for regression datasets.

    Args:
        train_data (pd.DataFrame | str | Path): DataFrame or path to CSV for training.
        test_data (pd.DataFrame | str | Path): DataFrame or path to CSV for validation/testing.
        train_image_folder (str | Path, optional): Directory containing training images.
        test_image_folder (str | Path, optional): Directory containing test images.
        train_transform (callable, optional): Transform pipeline for training images.
        test_transform (callable, optional): Transform pipeline for test images.
        batch_size (int): Batch size for both loaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    
    # Helper function
    def _load_regression_data(data_input):

        """
        Load and validate a regression dataset from CSV or DataFrame.
        """

        # Case 1: Already a DataFrame
        if isinstance(data_input, pd.DataFrame):
            df = data_input.copy()
        
         # Case 2: Path to CSV
        elif isinstance(data_input, (str, Path)):
            data_input = Path(data_input)
            if not data_input.exists():
                raise FileNotFoundError(f"File not found: {data_input}")
            if data_input.suffix.lower() != ".csv":
                raise ValueError(f"Expected CSV file, got: {data_input.suffix}")
            df = pd.read_csv(data_input)
        else:
            raise TypeError("Input must be a DataFrame or a path to a CSV file.")

        # Validate column count
        if df.shape[1] < 2:
            raise ValueError("The dataset must have at least two columns: 1st column with image_paths, 2nd column with scores.")

        # Rename columns anyway
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "image_path", cols[1]: "score"})

        # Drop missing or invalid entries
        df = df.dropna(subset=["image_path", "score"])
        df = df[df["image_path"].astype(str).str.strip() != ""]

        if not pd.api.types.is_string_dtype(df["image_path"]):
            raise TypeError("All 'image_path' entries must be strings.")
        if not pd.api.types.is_numeric_dtype(df["score"]):
            raise TypeError("All 'score' entries must be numeric.")

        return df[["image_path", "score"]]

    # Load and validate datasets
    train_df = _load_regression_data(train_data)
    test_df = _load_regression_data(test_data)

    # Create dataset objects
    train_dataset = RegressionDataset(train_df, image_folder=train_image_folder, transform=train_transform)
    test_dataset = RegressionDataset(test_df, image_folder=test_image_folder, transform=test_transform)

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader

    
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


def create_classification_dataloaders_with_resampling(
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


class CustomImageDatasetList(Dataset):
    
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
    

def create_classification_dataloaders_list(
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
    train_data = CustomImageDatasetList(train_image_paths, train_labels, transform=train_transform)
    val_data = CustomImageDatasetList(val_image_paths, val_labels, transform=val_transform)
    test_data = CustomImageDatasetList(test_image_paths, test_labels, transform=test_transform)

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


def create_classification_dataloaders_vit(
        model: str = "vit_b_16_224",
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
        model (str): The ViT model variant to use. Default is "vit_b_16_224".
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
        aug (bool): Whether to apply data augmentation to the training dataset. Default is True.
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
    if model not in vit_model_sizes:
        raise ValueError(f"[ERROR] Invalid ViT model '{model}'. Available options: {list(vit_model_sizes.keys())}")

    # Get image sizes for the selected ViT model
    IMG_SIZE_RESIZE, IMG_SIZE_CROP = vit_model_sizes[model]

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
    train_dataloader, test_dataloader, class_names = create_classification_dataloaders(
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


def create_classification_dataloaders_swin(
    model: str = "swin_v2_t",
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
        model (str): The specific Swin Transformer model to use. Default is "swin_t".
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
        aug (bool): Whether to apply data augmentation to the training dataset. Default is True.
        num_workers (int): Number of subprocesses to use for data loading. Default is the number of CPU cores.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (torch.utils.data.DataLoader): Data loader for the training dataset.
            - test_dataloader (torch.utils.data.DataLoader): Data loader for the testing dataset.
            - class_names (list): List of class names.
    """

    # Define input size and normalization parameters based on the model
    swin_model_sizes = {
        'swin_t_224':    (232, 224),
        'swin_s_224':    (246, 224),
        'swin_b_224':    (238, 224),
        'swin_v2_t_256': (260, 256),
        'swin_v2_s_256': (260, 256),
        'swin_v2_b_256': (272, 256)
    }

    # Validate model selection
    if model not in swin_model_sizes:
        raise ValueError(f"[ERROR] Invalid ViT model '{model}'. Available options: {list(swin_model_sizes.keys())}")

    # Get image sizes for the selected ViT model
    IMG_SIZE_RESIZE, IMG_SIZE_CROP = swin_model_sizes[model]

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
    train_dataloader, test_dataloader, class_names = create_classification_dataloaders(
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


class YUVBlockDataset(Dataset):
    def __init__(self, yuv_paths, width, height, block_size=32, stride=16):
        self.block_size = block_size
        self.yuv_paths = yuv_paths
        self.width = width
        self.height = height
        self.stride = stride
        self.coords = []  # Store (file_idx, top, left)

        for file_idx, path in enumerate(yuv_paths):
            with open(path, 'rb') as f:
                y, u, v = self.read_yuv420_frame(f, width, height)
                for top in range(0, height - block_size + 1, stride):
                    for left in range(0, width - block_size + 1, stride):
                        self.coords.append((file_idx, top, left))

        self.frames = [self.read_yuv420_frame(open(p, 'rb'), width, height) for p in yuv_paths]

    @staticmethod
    def read_yuv420_frame(file, width, height):
        frame_size = width * height
        uv_size = frame_size // 4

        y = np.frombuffer(file.read(frame_size), dtype=np.uint8).reshape((height, width))
        u = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
        v = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))

        return y, u, v

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        file_idx, top, left = self.coords[idx]
        y, u, v = self.frames[file_idx]

        # Crop 32×32 from Y
        luma = y[top:top+32, left:left+32]  # shape (32, 32)

        # Crop 16×16 from U and V (because of 4:2:0 subsampling)
        uv_top, uv_left = top // 2, left // 2
        cb = u[uv_top:uv_top+16, uv_left:uv_left+16]
        cr = v[uv_top:uv_top+16, uv_left:uv_left+16]

        # Convert to torch tensors
        luma_tensor = torch.from_numpy(luma).unsqueeze(0).float() / 255.0  # [1, 32, 32]
        chroma_tensor = torch.stack([
            torch.from_numpy(cb).float(),
            torch.from_numpy(cr).float()
        ]) / 255.0  # [2, 16, 16]

        return luma_tensor, chroma_tensor

#dataset = YUVBlockDataset(yuv_paths=['video1.yuv'], width=128, height=128)
#loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
#loss = F.mse_loss(predicted_chroma, target_chroma)
# loss = torch.mean(torch.sqrt((predicted - target)**2 + 1e-6))
#loss_cb = F.mse_loss(predicted[:,0,:,:], target[:,0,:,:])
#loss_cr = F.mse_loss(predicted[:,1,:,:], target[:,1,:,:])
#loss = 0.5 * loss_cb + 0.5 * loss_cr
#MSE + perceptual loss:
#Use MSE as main loss.
#Add a small perceptual loss (e.g. on a low-level VGG feature layer) if artifacts are an issue.
#This is rare for YUV block prediction but useful in high-quality upsampling tasks.