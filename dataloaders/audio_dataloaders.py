"""
Contains functionality for creating PyTorch DataLoaders for 
audio classification data.
"""

import os
import random
import torch
import torchaudio
from torchvision.transforms.v2 import Lambda
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn.utils.rnn import pad_sequence

NUM_WORKERS = os.cpu_count()


# Function to load an audio file and convert it to a tensor
def load_audio(file_path, target_sample_rate=None, convert_to_mono=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")

    waveform, sample_rate = torchaudio.load(file_path)

    # Convert stereo to mono by averaging channels
    if convert_to_mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if target_sample_rate and sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate


# Padding function
def pad_waveform(waveform, target_length):
    """Pads or truncates the waveform to ensure a fixed length."""
    current_length = waveform.shape[-1]

    if current_length < target_length:
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))  # Pad at the end
    elif current_length > target_length:
        waveform = waveform[:, :target_length]  # Truncate

    return waveform

import torch
import torch.nn as nn


# Padding class
class PadWaveform(torch.nn.Module):
    """A PyTorch module to pad or truncate waveforms to a fixed length."""

    def __init__(self, target_length=16000):
        
        """
        Initializes the PadWaveform module.

        Args:
            target_length (int): The desired length of the waveform after padding/truncation.
        """
        
        super(PadWaveform, self).__init__()  # Properly initializes the parent nn.Module.
        self.target_length = target_length  # Stores the target length as an instance attribute.

    def forward(self, waveform):
        
        """
        Pads or truncates the input waveform to ensure it has a fixed length.

        Args:
            waveform (Tensor): A PyTorch tensor representing the audio waveform. 
                               Expected shape: (channels, time) or (batch, channels, time).

        Returns:
            Tensor: The processed waveform with the specified target length.
        """

        waveform = pad_waveform(waveform, self.target_length)
       
        return waveform 


# Function to pad sequences to the same length
def pad_collate_fn(batch):

    """ 
    This function pad sequences in a batch to the same length
    """
    X, y = zip(*batch)

    # Process each sequence x from the batch
    X_tensors = []
    for x in X:
        # Case 1: 1D tensor (length,)
        if len(x.shape) == 1:
            x = x.view(-1, 1)  # Reshape to (length, 1) for mono audio
        
        # Case 2: 2D tensor (length, channels) or (channels, length)
        # We need to permute so that pad_sequence works
        elif len(x.shape) == 2:
            if x.shape[0] < x.shape[1]:  # Check if it's (channels, length)
                x = x.T  # Transpose to (length, channels)
        
        X_tensors.append(torch.tensor(x))

    # Use pad_sequence to pad the sequences, ensuring they are in the correct shape
    X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=0)
    X_padded = X_padded.permute(0, 2, 1)  # Ensure the final shape is (batch_size, length, channels)
    
    y = torch.tensor(y)  # Ensure y is a tensor
    
    return X_padded, y

#class PadSequenceTransform(torch.nn.Module):
#    def __init__(self):
#        super(PadSequenceTransform, self).__init__()

#    def forward(self, batch):
#        # Apply padding across the batch
#        return pad_sequence(batch, batch_first=True, padding_value=0)


# Custom Dataset for Audio Classification
class AudioDataset(Dataset):
    
    """
    A PyTorch Dataset class for loading and preprocessing audio files for classification.

    Args:
        root_dir (str): The root directory containing subdirectories of audio files for each class.
        transform (callable, optional): A function/transform to apply to the audio waveform.
    """

    def __init__(self, root_dir, transform=None):
    
        """
        Initializes the dataset, setting the root directory, transform, and loading file paths and labels.

        Args:
            root_dir (str): The root directory where class subdirectories containing audio files reside.
            transform (callable, optional): A transformation to apply to the audio data.
        """
    
        self.root_dir = root_dir  # Root directory containing subdirectories for each class.
        self.transform = transform  # Optional transformation to apply to the audio data.
        self.files = []  # List to store file paths of all audio files.
        self.labels = []  # List to store corresponding labels for each audio file.
        
        # Retrieve class names by listing all subdirectories in the root directory.
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        # Create a label map, assigning each class a unique integer index.
        label_map = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Loop over each class and collect file paths for audio files (.wav).
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)  # Get path of the current class subdirectory.
            for file in os.listdir(class_path):  # Loop over all files in the class directory.
                if file.endswith(".wav"):  # Filter only .wav files.
                    self.files.append(os.path.join(class_path, file))  # Store the full file path.
                    self.labels.append(label_map[class_name])  # Assign the class label to the file.

    def __len__(self):
     
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of audio files in the dataset.
        """
     
        return len(self.files)

    def __getitem__(self, idx):
     
        """
        Returns the audio waveform and corresponding label for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio waveform and its label.
        """
     
        file_path = self.files[idx]  # Get the file path for the sample at the specified index.
        label = self.labels[idx]  # Get the label for the sample at the specified index.
        
        # Load the audio data (waveform and sample rate) from the file.
        waveform, sample_rate = load_audio(file_path)

        # Apply the transformation, if any, to the waveform.
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


# Custom Dataset for Audio Classification
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        # Assign label indices based on sorted directory names
        label_map = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(class_path, file))
                    self.labels.append(label_map[class_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = load_audio(file_path)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

# Function to create DataLoaders
def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform=None,
    test_transform=None,
    num_train_samples: int = None,
    num_test_samples: int = None,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42
):
    """ 
    Function to create DataLoaders for training and testing audio data. This function already aligns the lengths
    of the audio waveforms.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        train_transform (callable, optional): Transformation function for training data.
        test_transform (callable, optional): Transformation function for testing data.
        num_train_samples (int, optional): Number of training samples to randomly select. If None, use all.
        num_test_samples (int, optional): Number of testing samples to randomly select. If None, use all.
        batch_size (int): Batch size for the DataLoader. Default is 32.
        num_workers (int): Number of worker threads for data loading. Default is 4.
        random_seed (int): Seed for random operations to ensure reproducibility. Default is 42.

    Returns:
        tuple: (train_dataloader, test_dataloader, class_names)
            - train_dataloader (DataLoader): DataLoader for training data.
            - test_dataloader (DataLoader): DataLoader for testing data.
            - class_names (list): List of class labels present in the training dataset.
    """

    random.seed(random_seed)

    # Create datasets
    train_data = AudioDataset(root_dir=train_dir, transform=train_transform)
    test_data = AudioDataset(root_dir=test_dir, transform=test_transform)

    # Extract class names
    class_names = train_data.classes  

    # Resample training data if num_train_samples is specified
    if num_train_samples is not None:
        train_indices = random.sample(range(len(train_data)), k=min(num_train_samples, len(train_data)))
        train_data = Subset(train_data, train_indices)

    # Resample testing data if num_test_samples is specified
    if num_test_samples is not None:
        test_indices = random.sample(range(len(test_data)), k=min(num_test_samples, len(test_data)))
        test_data = Subset(test_data, test_indices)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_collate_fn  
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_fn  
    )

    return train_dataloader, test_dataloader, class_names

