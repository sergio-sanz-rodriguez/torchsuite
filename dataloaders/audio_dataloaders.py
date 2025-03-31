"""
Contains functionality for creating PyTorch DataLoaders for 
audio classification data.
"""

import os
import random
import torch
import sys
import librosa
import torchaudio
from torchaudio.transforms import Resample, FrequencyMasking, TimeMasking, AmplitudeToDB, PitchShift
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
sys.path.append(os.path.abspath("../engines"))
from engines.common import Logger
sys.path.append(os.path.abspath("../utils"))
from utils.classification_utils import set_seeds

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
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate

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

        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []

        # Retrieve class names
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        # Assign label indices based on sorted directory names
        label_map = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Loop over each class and collect file paths for audio files (.wav).
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(class_path, file))
                    self.labels.append(label_map[class_name])

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

        file_path = self.files[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = load_audio(file_path)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


# Padding class
class PadWaveform(torch.nn.Module):
    
    """
    A PyTorch module to pad or truncate waveforms to a fixed length.
    """

    def __init__(self, target_length=16000):
        
        """
        Initializes the PadWaveform module.

        Args:
            target_length (int): The desired length of the waveform after padding/truncation.
        """
        
        super(PadWaveform, self).__init__()  # Properly initializes the parent nn.Module.
        self.target_length = target_length  # Stores the target length as an instance attribute.
    
    # Padding function
    def pad_waveform(self, waveform):

        """
        Pads or truncates the waveform to ensure a fixed length.
        """

        current_length = waveform.shape[-1]

        if current_length < self.target_length:
            pad_amount = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))  # Pad at the end
        elif current_length > self.target_length:
            waveform = waveform[:, :self.target_length]  # Truncate

        return waveform

    def forward(self, waveform):
        
        """
        Pads or truncates the input waveform to ensure it has a fixed length.

        Args:
            waveform (Tensor): A PyTorch tensor representing the audio waveform. 
                               Expected shape: (channels, time) or (batch, channels, time).

        Returns:
            Tensor: The processed waveform with the specified target length.
        """

        waveform = self.pad_waveform(waveform)
       
        return waveform 


# Augmentation class
class AudioAugmentations(Logger, torch.nn.Module):

    """
    A module for applying audio augmentations in the time domain (waveform) or 
    frequency domain (spectrogram). Supports background noise addition, 
    frequency masking, and time masking.
    """
    
    def __init__(
        self,
        apply_augmentation: bool=True,
        signal: str="waveform",
        sample_rate: Optional[int]=None, # For pitch shifting
        n_mels: Optional[int]=None, # For frequency masking
        target_length: Optional[int]=None, # For time masking
        hop_length: Optional[int]=None, # For time masking
        seed: Optional[int]=None,
        augment_magnitude=2):

        """
        Initializes the AudioAugmentations module.

        Args:
            apply_augmentation (bool): Whether to apply augmentation or not.
            signal (str): Choose between 'waveform' (time domain) or 'spectrogram' (frequency domain).
            sample_rate (int, optional): Sample rate of audio (used for pitch shifting).
            n_mels (int, optional): Number of mel bands (used for frequency masking).
            target_length (int, optional): Target length for time masking.
            hop_length (int, optional): Hop length used for spectrogram.
            seed (int): The seed for random number generation.
            augment_magnitude (int): Strength of augmentation.
        """

        torch.nn.Module.__init__(self)
        Logger.__init__(self)        
        super().__init__()
                
        if not isinstance(apply_augmentation, bool):
            self.error("'apply_augmentation' must be a boolean.")
        if not isinstance(sample_rate, int) and sample_rate is not None:
            self.error("'sample_rate' must be an integer.")
        if not isinstance(n_mels, int) and n_mels is not None:
            self.error("'n_mels' must be an integer.")
        if not isinstance(target_length, int) and target_length is not None:
            self.error("'target_length' must be an integer.")
        if not isinstance(hop_length, int) and hop_length is not None:
            self.error("'hop_length' must be an integer.")
        if not isinstance(seed, int) and seed is not None:
            self.error("'seed' must be an integer.")
        if not isinstance(augment_magnitude, int):
            self.error("'augment_magnitude' must be an integer.")
        
        signal_options = ["waveform", "spectrogram"]
        if not isinstance(signal, str):
            self.error("'signal' must be a string.")
        elif signal not in signal_options:
            self.error(f"'signal' must be one of {signal_options}.")
        
        self.apply_augmentation = apply_augmentation
        self.signal = signal
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        self.hop_length = hop_length
        self.seed = seed
        self.augment_magnitude = augment_magnitude

    
    def apply_pitch_shift(self, waveform, sample_rate, n_steps=1):

        """
        Applies pitch shifting to a waveform.

        Args:
            waveform: The input audio waveform tensor.
            sample_rate: The sample rate of the waveform.
            n_steps: The number of semitones to shift the pitch.

        Returns:
            The pitch-shifted waveform.
        """

        #waveform = waveform.numpy()  # Convert from tensor to numpy array
        #waveform_shifted = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)
        #return torch.tensor(waveform_shifted)
        #waveform_np = waveform.detach().cpu().numpy()
        #if waveform_np.ndim > 1:
        #    waveform_shifted = np.stack([librosa.effects.pitch_shift(ch, sr=sample_rate, n_steps=n_steps) for ch in waveform_np])
        #else:
        #    waveform_shifted = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)
        #return torch.tensor(waveform_shifted, dtype=waveform.dtype, device=waveform.device)

        # Ensure waveform is [channel, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # make it [1, time]

        pitch_shift = PitchShift(sample_rate=sample_rate, n_steps=n_steps)

        return pitch_shift(waveform)

    
    def add_background_noise(self, waveform, noise_level=0.005):

        """
        Adds background noise to a waveform.

        Args:
            waveform: The input audio waveform tensor.
            noise_level: The level of background noise to add.

        Returns:
            The waveform with added background noise.
        """

        if self.seed is not None:
            set_seeds(self.seed)

        noise = torch.randn_like(waveform) * noise_level

        return waveform + noise
    
    def apply_time_augmentation(self, waveform):

        """
        Applies augmentation in the time domain (waveform).
        Currently, only background noise is added.

        Returns:
            Tensor: Augmented waveform.
        """

        #pitch_shift = 1 * self.augment_magnitude
        background_noise = 0.0025 * self.augment_magnitude

        #waveform = self.apply_pitch_shift(waveform, sample_rate=self.sample_rate, n_steps=pitch_shift)
        waveform = self.add_background_noise(waveform, noise_level=background_noise)

        return waveform
    
    def apply_freq_augmentation(self, spectrogram):

        """
        Applies augmentation in the frequency domain (spectrogram).
        Adds Frequency Masking and Time Masking.

        Returns:
            Tensor: Augmented spectrogram.
        """

        freq_mask_param = int(0.05 * self.n_mels * self.augment_magnitude)
        total_time_steps = (self.target_length // self.hop_length) + 1
        time_mask_param = int(0.05 * total_time_steps * self.augment_magnitude)

        spectrogram = FrequencyMasking(freq_mask_param=freq_mask_param)(spectrogram)
        spectrogram = TimeMasking(time_mask_param=time_mask_param)(spectrogram)
        
        return spectrogram
    
    def forward(self, x):

        """
        Applies augmentations to the input waveform.

        Args:
            waveform: The input audio waveform tensor.

        Returns:
            The augmented waveform.
        """

        if not self.apply_augmentation:
            return x
        elif self.signal == "waveform":
            return self.apply_time_augmentation(x)
        else:
            return self.apply_freq_augmentation(x)


# Waveform-based audio transforms
class AudioWaveformTransforms(Logger, torch.nn.Module):

    """
    Initializes the AudioWaveformTransforms class with the given parameters.
    
    Args:
        augmentation (bool): Whether to apply augmentation to the audio.        
        sample_rate (int): The original sample rate of the audio.
        new_sample_rate (int): The resampled audio sample rate.
        target_length (int): The target length of the waveform after padding.        
        seed (int): The seed for random number generation.
        augment_magnitude (int): The magnitude of augmentation applied to the audio.
    """

    def __init__(
        self,
        augmentation: bool = True,        
        sample_rate: int = 16000,
        new_sample_rate: int = 8000,
        target_length: int = 8000, 
        seed: Optional[int] = None,       
        augment_magnitude: int = 2,
        ):

        """
        Initializes the AudioWaveformTransforms class with the given parameters.
        
        Args:
            augmentation (bool): Whether to apply augmentation to the audio.        
            sample_rate (int): The original sample rate of the audio.
            new_sample_rate (int): The resampled audio sample rate.
            target_length (int): The target length of the waveform after padding.        
            seed (int): The seed for random number generation.
            augment_magnitude (int): The magnitude of augmentation applied to the audio.
        """

        torch.nn.Module.__init__(self)
        Logger.__init__(self)
        super().__init__()

        self.validate_inputs(
            augmentation, sample_rate, new_sample_rate,
            target_length, augment_magnitude)
        
        self.augmentation = augmentation        
        self.sample_rate = sample_rate
        self.new_sample_rate = new_sample_rate
        self.target_length = target_length        
        self.seed = seed
        self.augment_magnitude = augment_magnitude
        
        # Time-domain processing
        self.time_transforms = torch.nn.Sequential(

            # Resample audio
            Resample(
                orig_freq=self.sample_rate,
                new_freq=self.new_sample_rate),   
            
            # Pad waveform to target length
            PadWaveform(
                target_length=self.target_length),

            # Apply time augmentation
            AudioAugmentations(
                apply_augmentation=self.augmentation,
                signal="waveform",
                seed=self.seed,
                augment_magnitude=self.augment_magnitude)
        )        
                
    def validate_inputs(
        self, augmentation, sample_rate, new_sample_rate,
        target_length, augment_magnitude):

        """
        Validates the input parameters for the AudioWaveformTransforms class.
        
        Ensures that the inputs are of the correct type and within acceptable ranges. If any input is invalid,
        an error message is logged.
        
        Args:
            augmentation (bool): Whether to apply augmentation to the audio.            
            sample_rate (int): The original sample rate of the audio.
            new_sample_rate (int): The resampled audio sample rate.
            target_length (int): The target length of the waveform after padding.            
            augment_magnitude (int): The magnitude of augmentation applied to the audio..

        """

        # Type and value range checks
        if not isinstance(augmentation, bool):
            self.error("'augmentation' must be a boolean.")        
        if not isinstance(sample_rate, int):
            self.error("'sample_rate' must be an integer.")
        if not isinstance(new_sample_rate, int):
            self.error("'new_sample_rate' must be an integer.")
        if not isinstance(target_length, int):
            self.error("'target_length' must be an integer.")
        if not isinstance(augment_magnitude, int):
            self.error("'augment_magnitude' must be an integer.")

        # Augment magnitude
        if augment_magnitude < 1 or augment_magnitude > 5:
            self.error("'augment_magnitude' must be between 1 and 5.")

    
    def forward(self, x):

        """
        Applies time-domain transformations: resampling, padding, augmentation (optional)

        Returns:
            A transformed waveform
        """

        return self.time_transforms(x)


class AudioSpectrogramTransforms(Logger, torch.nn.Module):

    """
    Initializes the AudioSpectrogramTransforms class with the given parameters.
    
    Args:
        augmentation (bool): Whether to apply augmentation to the audio.
        mean_std_norm (bool): Whether to normalize the spectrogram using mean and std.
        fft_analysis_method (str): The type of FFT analysis to perform ("none", "time_freq", or "freq_band"):
            - "none": One spectrogram fo the whole signal.
            - "time_freq": Three spectrograms with different time-frequency-resolution trade-offs.
            - "freq_band": Three spectrograms analyzing different (low, mid, high) frequency bands.
            - "all": Creates an image with three channels, one channel for each fft analysis method
        fft_analysis_concat (str): Only applicable to "time_freq" and "freq_band", specifies how the spectrograms should be concatenated:
            - "freq": Concatenation along the frequency axis (default for "freq_band").
            - "time": Concatenation along the time axis.
            - "channel": One spectrogram per image channel (RGB-like) (default for "time_freq")
            - "default": "freq" for method "freq_band", "channel" for method "time_freq"
        sample_rate (int): The original sample rate of the audio.
        new_sample_rate (int): The resampled audio sample rate.
        target_length (int): The target length of the waveform after padding.
        n_fft (int): The number of FFT bins.
        win_length (Optional[int]): The window length for FFT.
        hop_length (Optional[int]): The hop length for FFT.
        n_mels (Optional[int]): The number of mel bands.
        power (float): The power of the spectrogram.
        img_size (tuple): The desired size of the output image (height, width).
        seed (int): The seed for random number generation.
        augment_magnitude (int): The magnitude of augmentation applied to the audio.
    """

    def __init__(
        self,
        augmentation: bool = True,
        mean_std_norm: bool = True,
        fft_analysis_method: str = "none",
        fft_analysis_concat: str = "default",
        sample_rate: int = 16000,
        new_sample_rate: int = 8000,
        target_length: int = 8000,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: Optional[int] = None,
        power: float = 2.0,
        img_size: tuple = (384, 384), #(W, W)
        seed: Optional[int] = None,
        augment_magnitude: int = 2,
        ):

        """
        Initializes the AudioSpectrogramTransforms class with the given parameters.
        
        Args:
            augmentation (bool): Whether to apply augmentation to the audio.
            mean_std_norm (bool): Whether to normalize the spectrogram using mean and std.
            fft_analysis_method (str): The type of FFT analysis to perform ("none", "time_freq", or "freq_band"):
                - "none": One spectrogram fo the whole signal.
                - "time_freq": Three spectrograms with different time-frequency-resolution trade-offs.
                - "freq_band": Three spectrograms analyzing different (low, mid, high) frequency bands.
                - "all": Creates an image with three channels, one channel for each fft analysis method
            fft_analysis_concat (str): Only applicable to "time_freq" and "freq_band", specifies how the spectrograms should be concatenated:
                - "freq": Concatenation along the frequency axis (default for "freq_band").
                - "time": Concatenation along the time axis.
                - "channel": One spectrogram per image channel (RGB-like) (default for "time_freq")
                - "default": "freq" for method "freq_band", "channel" for method "time_freq"
            sample_rate (int): The original sample rate of the audio.
            new_sample_rate (int): The resampled audio sample rate.
            target_length (int): The target length of the waveform after padding.
            n_fft (int): The number of FFT bins.
            win_length (Optional[int]): The window length for FFT.
            hop_length (Optional[int]): The hop length for FFT.
            n_mels (Optional[int]): The number of mel bands.
            power (float): The power of the spectrogram.
            img_size (tuple): The desired size of the output image (height, width).
            seed (int): Specifies the seed
            augment_magnitude (int): The magnitude of augmentation applied to the audio.
        """

        torch.nn.Module.__init__(self)
        Logger.__init__(self)
        super().__init__()

        self.validate_inputs(
            augmentation, mean_std_norm, fft_analysis_method, fft_analysis_concat, sample_rate, new_sample_rate,
            target_length, n_fft, win_length, hop_length, n_mels, power, img_size, augment_magnitude)
        
        self.augmentation = augmentation
        self.mean_std_norm = mean_std_norm
        self.fft_analysis_method = fft_analysis_method
        self.fft_analysis_concat = fft_analysis_concat
        self.sample_rate = sample_rate
        self.new_sample_rate = new_sample_rate
        self.target_length = target_length
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = round(target_length / (img_size[0] - 1)) if hop_length is None else hop_length
        self.n_mels = img_size[0] if n_mels is None else n_mels
        self.power = power
        self.img_size = img_size
        self.seed = seed
        self.augment_magnitude = augment_magnitude
        self.overlap_ratio = 0.2

        # Time-domain processing
        self.time_transforms = torch.nn.Sequential(

            # Resample audio
            Resample(
                orig_freq=self.sample_rate,
                new_freq=self.new_sample_rate),   
            
            # Pad waveform to target length
            PadWaveform(
                target_length=self.target_length),

            # Apply time augmentation
            AudioAugmentations(
                apply_augmentation=self.augmentation,
                signal="waveform",
                seed=self.seed,
                augment_magnitude=self.augment_magnitude)
        )        

        # Frequency-domain processing
        self.freq_transforms = torch.nn.Sequential(

            # Convert audio to Mel spectrogram
            self.mel_spectrogram(),

            # Apply frequency augmentation
            AudioAugmentations(
                apply_augmentation=self.augmentation,
                signal="spectrogram",
                n_mels=self.n_mels,
                target_length=self.target_length,
                hop_length=self.hop_length,
                seed=self.seed,
                augment_magnitude=self.augment_magnitude),

            # Convert amplitude to decibels
            AmplitudeToDB(stype="power", top_db=80),        
        )

        # Image transformations
        self.image_transforms = v2.Compose([
            v2.Resize(img_size, antialias=True),
            v2.CenterCrop(img_size),    
            v2.Lambda(self.expand_channel), # Convert from [1, H, W] to [3, H, W] if needed        
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(self.min_max_normalize)  # Normalize to [0, 1]
        ])

        # Optional normalization
        if mean_std_norm:
            self.image_transforms.transforms.append(
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                    )

    def validate_inputs(
        self, augmentation, mean_std_norm, fft_analysis_method, fft_analysis_concat, sample_rate, new_sample_rate,
        target_length, n_fft, win_length, hop_length, n_mels, power, img_size, augment_magnitude):

        """
        Validates the input parameters for the AudioSpectrogramTransforms class.
        
        Ensures that the inputs are of the correct type and within acceptable ranges. If any input is invalid,
        an error message is logged.
        
        Args:
            The parameters to validate.

        """

        # Type and value range checks
        if not isinstance(augmentation, bool):
            self.error("'augmentation' must be a boolean.")
        if not isinstance(mean_std_norm, bool):
            self.error("'mean_std_norm' must be a boolean.")
        if not isinstance(sample_rate, int):
            self.error("'sample_rate' must be an integer.")
        if not isinstance(new_sample_rate, int):
            self.error("'new_sample_rate' must be an integer.")
        if not isinstance(target_length, int):
            self.error("'target_length' must be an integer.")
        if not isinstance(n_fft, int):
            self.error("'n_fft' must be an integer.")
        if not isinstance(power, (int, float)):
            self.error("'power' must be an integer or float.")
        if not isinstance(img_size, tuple):
            self.error("'img_size' must be a tuple.")
        if not isinstance(augment_magnitude, int):
            self.error("'augment_magnitude' must be an integer.")

        # Analyisis options
        analysis_options = ["single", "time_freq", "freq_band"]
        if not isinstance(fft_analysis_method, str):
            self.error("'fft_analysis_method' must be a string.")
        elif fft_analysis_method not in analysis_options:
            self.error(f"'fft_analysis_method' must be one of {analysis_options}.")

        concat_options = ["freq", "time", "channel", "default"]
        if not isinstance(fft_analysis_concat, str):
            self.error("'fft_analysis_concat' must be a string.")
        elif fft_analysis_concat not in concat_options:
            self.error(f"'fft_analysis_concat' must be one of {concat_options}.")

        # Optional parameter checks
        if win_length is not None and not isinstance(win_length, int):
            self.error("'win_length' must be either None or an integer.")
        if hop_length is not None and not isinstance(hop_length, int):
            self.error("'hop_length' must be either None or an integer.")
        if n_mels is not None and not isinstance(n_mels, int):
            self.error("'n_mels' must be either None or an integer.")
        
        # Augment magnitude
        if augment_magnitude < 1 or augment_magnitude > 5:
            self.error("'augment_magnitude' must be between 1 and 5.")


    @staticmethod
    def min_max_normalize(x):

        """
        Normalizes tensor to [0, 1] range.
        """

        return (x - x.min()) / (x.max() - x.min() + 1e-6)  # Avoid division by zero

    @staticmethod
    def abs_normalize(x):

        """
        Normalizes tensor to [-1, 1] range
        """

        return x / x.abs().max()
        
    @staticmethod
    def expand_channel(x):

        """
        Expands the input tensor to have 3 channels.
        """

        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        
        return x
    
    @staticmethod
    def squeeze_channel(x):

        """
        Removes the channel dimension from the input tensor.
        """

        return torch.squeeze(x, dim=1)

    @staticmethod
    def log_freq_band_split(f_min, f_max):

        """
        Split the frequency range [f_min, f_max] into three logarithmic subbands
        similar to Mel scale.
        
        Args:
            f_min (float or tensor): Minimum frequency in Hz.
            f_max (float or tensor): Maximum frequency in Hz.
            
        Returns:
            tuple: Three frequency bands (low, mid, high) in Hz.
        """

        # Convert to tensor anyway
        f_min = torch.tensor(f_min)
        f_max = torch.tensor(f_max)

        def hz_to_mel(f):
            """Convert Hz to Mel scale."""
            return 2595 * torch.log10(1 + f / 700)

        def mel_to_hz(m):
            """Convert Mel scale back to Hz."""
            return 700 * (10**(m / 2595) - 1)
        
        # Convert to Mel scale
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        # Split into 3 equal Mel intervals
        mel_1 = mel_min + (mel_max - mel_min) / 3
        mel_2 = mel_min + 2 * (mel_max - mel_min) / 3
        mel_3 = mel_max  # Highest frequency remains unchanged

        # Convert back to Hz
        f1 = mel_to_hz(mel_1)
        f2 = mel_to_hz(mel_2)
        f3 = mel_to_hz(mel_3)

        return f1, f2, f3

    def mel_spectrogram_default(self):

        """
        Creates a sequential transformation pipeline to generate a Mel spectrogram 
        from an audio waveform. The output spectrogram has a shape of [1, H, W], 
        where H represents the number of Mel filter banks, and W represents the 
        number of time frames.

        Returns:
            torch.nn.Sequential: A sequential module that applies the transformations 
            in order, returning a Mel spectrogram tensor.
        """

        compositor =  torch.nn.Sequential(

            # Apply normalization to [-1, 1] range
            v2.Lambda(self.abs_normalize),

            # Compute Mel spectrogram
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.new_sample_rate,  # Use instance attributes
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=self.power
            ),

            # Convert (1, H, W) to (3, H, W)
            #v2.Lambda(self.expand_channels),
        )

        return compositor

    # Generate Mel Spectrogram for the different channels
    def mel_spectrogram_time_freq(self):

        """
        Generates a Mel spectrogram using multiple FFT sizes to capture different 
        frequency and temporal resolutions. The generated spectrograms are then 
        concatenated along a specified dimension.

        Returns:
            torch.nn.Sequential: A sequential transformation.

        Notes:
            - self.fft_analysis_concat defines how the spectrograms are concatenated:
                - "freq": Stacks spectrograms along the frequency dimension, reducing 
                the number of Mel bands per spectrogram.
                - "time": Stacks spectrograms along the time dimension, reducing temporal 
                resolution while preserving frequency information
                - "channel"/"default": Stacks spectrograms along the channel dimension, preserving 
                the original number of Mel bands.
            - The spectrograms are computed on the CPU.
            - This method is useful for improving model robustness by combining 
            multiple frequency-temporal representations.
        """

        # Spectrograms to be computed on the "cpu"
        device = "cpu"

        # Spectrogram concatenation by frequency
        if self.fft_analysis_concat == "freq":
            hop_length = self.hop_length
            n_mels = round(self.n_mels / 3)
            dim = 1
            squeeze = False

        # Spectrogram concatenation by time
        elif self.fft_analysis_concat == "time":
            hop_length = self.hop_length * 3
            n_mels = self.n_mels
            dim = 2
            squeeze = False

        # Spectrogram concatenation by channel
        elif self.fft_analysis_concat == "channel" or self.fft_analysis_concat == "default":
            hop_length = self.hop_length
            n_mels = self.n_mels
            dim = 0
            squeeze = True

        # High frequency resolution, low temporal resolution (large n_fft)
        spec_1 = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.new_sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=self.power
        )

        # Medium frequency resolution, medium temporal resolution (half n_fft)
        spec_2 = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.new_sample_rate,
            n_fft=int(self.n_fft // 2),
            win_length=int(self.n_fft // 2),
            hop_length=hop_length,
            n_mels=n_mels,
            power=self.power
        )

        # Low frequency resolution, high temporal resolution (small n_fft)
        spec_3 = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.new_sample_rate,
            n_fft=int(self.n_fft // 4),
            win_length=int(self.n_fft // 4),
            hop_length=hop_length,
            n_mels=n_mels,
            power=self.power
        )

        # Move to device
        spec_1 = spec_1.to(device)
        spec_2 = spec_2.to(device)
        spec_3 = spec_3.to(device)

        # Define the transformation function
        compositor = torch.nn.Sequential(
            
            # Apply normalization to [-1, 1] range
            v2.Lambda(self.abs_normalize),

            # Compute Mel spectrogram
            v2.Lambda(lambda x: torch.cat([
                spec_1(x.to(device)),
                spec_2(x.to(device)),
                spec_3(x.to(device)),
            ], dim=dim)),
            
            # Convert (1, H, W) to (3, H, W)
            v2.Lambda(self.squeeze_channel) if squeeze else v2.Lambda(lambda x: x)
        )

        return compositor

    def mel_spectrogram_freq_band(self):

        """
        Generates a Mel spectrogram using the frequency band method.

        This method splits the frequency range into three overlapping bands: 
        low, medium, and high frequencies. Separate Mel spectrograms are 
        computed for each band and then stacked together along a specified 
        dimension.

        Returns:
            torch.nn.Sequential: A sequential transformation.

        Notes:
            - self.fft_analysis_concat defines how the spectrograms are concatenated:
                - "freq"/"default": Stacks spectrograms along the frequency axis.
                - "time": Stacks spectrograms along the time axis with 
                increased hop length for lower temporal resolution.
                - "channel": Stacks spectrograms along the channel axis.
            - The overlap between frequency bands is controlled by `self.overlap_ratio`.
            - Spectrograms are computed on the CPU by default.
        """

        # Spectrograms to be computed on the "cpu"
        device = "cpu"

        # Spectrogram concatenation by frequency
        if self.fft_analysis_concat == "freq" or self.fft_analysis_concat == "default":
            hop_length = self.hop_length
            n_mels = round(self.n_mels / 3)
            dim = 1
            squeeze = False

        # Spectrogram concatenation by time
        elif self.fft_analysis_concat == "time":
            hop_length = self.hop_length * 3
            n_mels = self.n_mels
            dim = 2
            squeeze = False

        # Spectrogram concatenation by channel
        elif self.fft_analysis_concat == "channel":
            hop_length = self.hop_length
            n_mels = self.n_mels
            dim = 0
            squeeze = True

        # Compute frequency limits
        f_max1, f_max2, f_max3 = self.log_freq_band_split(0, self.new_sample_rate / 2)

        # Compute overlap amount
        overlap1 = (f_max1 - 0) * self.overlap_ratio
        overlap2 = (f_max2 - f_max1) * self.overlap_ratio
        overlap3 = (f_max3 - f_max2) * self.overlap_ratio

        # Define overlapping frequency ranges
        f_min1, f_max1 = 0, f_max1 + overlap1
        f_min2, f_max2 = f_max1 - overlap1, f_max2 + overlap2
        f_min3, f_max3 = f_max2 - overlap2, f_max3

        # Low-frequence band
        spec_1 = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.new_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            power=self.power,
            f_min=f_min1,
            f_max=f_max1
        )

        # Medium-frequency band
        spec_2 = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.new_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            power=self.power,
            f_min=f_min2,
            f_max=f_max2
        )

        # High-frequency band
        spec_3 = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.new_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            power=self.power,
            f_min=f_min3,
            f_max=f_max3
        )

        # Move to device
        spec_1 = spec_1.to(device)
        spec_2 = spec_2.to(device)
        spec_3 = spec_3.to(device)

        # Define the transformation function
        compositor = torch.nn.Sequential(
            
            # Apply normalization to [-1, 1] range
            v2.Lambda(self.abs_normalize),

            # Compute Mel spectrogram
            v2.Lambda(lambda x: torch.cat([
                spec_1(x.to(device)),
                spec_2(x.to(device)),
                spec_3(x.to(device)),
            ], dim=dim)),
            
            # Convert (1, H, W) to (3, H, W)
            v2.Lambda(self.squeeze_channel) if squeeze else v2.Lambda(lambda x: x)

            # Convert (1, H, W) to (3, H, W) for image channels
            #v2.Lambda(self.expand_channels)
        )

        return compositor
    

    def mel_spectrogram(self):

        """
        Determines the type of Mel spectrogram transformation to apply based on the 'fft_analysis_method' parameter.

        Returns:
            A Lambda function for the chosen Mel spectrogram transformation.
        """

        # Default spectrogram transformation
        if self.fft_analysis_method == "single":
            return self.mel_spectrogram_default()
        
        # Spectrogram concatenation by time-frequency trade-offs
        elif self.fft_analysis_method == "time_freq":
            return self.mel_spectrogram_time_freq()
        
        # Spectrogram concatenation by frequency bands
        elif self.fft_analysis_method == "freq_band":
            return self.mel_spectrogram_freq_band()
        

    def forward(self, x):

        """
        Applies transformations in three distinct stages.

        Returns:
            An RGB-like image showing an spectrogram
        """

        # Time-domain processing: resampling, padding, augmentation (optional)
        x = self.time_transforms(x)

        # Conversion from waveform to spectrogram including augmentation (optional)
        x = self.freq_transforms(x)

        # Conversion from spectrogram to image
        x = self.image_transforms(x)

        return x



# Function to create DataLoaders
def create_dataloaders_waveform(
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



    # Function to create DataLoaders
def create_dataloaders_spectrogram(
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

    # Create dataloaders
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

