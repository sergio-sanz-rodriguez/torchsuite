<div align="center">
  <img src="images/logo/logo_tochsuite_1_resized_1.jpg" alt="Into Picture" width="450"/>
</div>

# TorchSuite 
TorchSuite is a versatile and feature-rich PyTorch-based library designed to simplify and accelerate deep learning development. It provides essential modules and utilities for training, inference, data handling, and model optimization, making it an invaluable tool for researchers, machine learning professionals, and practitioners.  

## Features

The main highlights of the library are listed next:

- Simplified model training and evaluation
- CPU and GPU-accelerated computations
- Support for a variety of deep learning architectures
- Easy-to-use API for seamless experimentation

Currently, TorchSuite is fully optimized for the following machine learning areas:
- Image and audio classification and distillation
- Image regression 
- Object detection
- Image segmentation

Future versions will include support for other data types, such as video and text.

## Modules  
- **CPU and GPU-accelerated computation:** optimized for both CPU and GPU, enabling flexible model training and inference
- **Training and inference engines:** `classification.py`, `regression.py` (new feature), `obj_detection.py`, `segmentation.py` for training and evaluation workflows
- **Flexible data loading:** `image/audio/obj_dect/segmentation_dataloaders.py` to streamline dataset preparation and augmentation.
- **Utility functions:** `classification/obj_detection/segmentation_utils.py` offering a collection of utility functions to enhance productivity.
- **Vision Transformer (ViT) support:** `vision_transformer.py` for implementing ViTs with PyTorch.
- **ConvNeXT support:**  `convnext.py`, the [original implementation](https://github.com/facebookresearch/ConvNeXt) by Meta Platforms, Inc.
- **Wav2Vec2 support:** `wav2vec2.py` includes a Transformer-based acoustic model for audio classification.
- **Region-based CNN (R-CNN) support:** `faster_rcnn.py` to create deep learning networks for object detection.
- **U-Net support:** `unet.py` for a flexible design of U-Net models tailored to image segmentation. Pretrained U-Net modeling is taken from [mberkay0](https://github.com/mberkay0/pretrained-backbones-unet). The library includes the canonical vanilla U-Net architecture as well.
- **Learning rate scheduling:** `schedulers.py` provides adaptive learning rate strategies. Some classes have been taken from [kamrulhasanrony](https://github.com/kamrulhasanrony/Vision-Transformer-based-Food-Classification/tree/master). 
- **Custom loss functions:** `loss_functions.py` including various loss formulations for different tasks.
- **Code examples:** a series of notebooks demonstrating Python code for training deep learning models.

## ToDo
- Code example for regression taks.

## Installation

To install and set up this project, follow these steps:
1. Clone the repository: 
```bash
 git clone https://github.com/sergio-sanz-rodriguez/TorchSuite
```

2. Navigate into the project directory:
```bash
cd TorchSuite
```

3. Create a virtual environment with GPU suppport:
```bash
conda create --name torchsuite_gpu python=3.11.10
conda activate torchsuite_gpu

(or using venv)

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install additional dependencies:
```bash
pip install -r requirements.txt
```

5. Install PyTorch with GPU support (modify CUDA version as needed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

6. Verify installation
```bash
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

## Dependencies

The required libraries are listed in `requirements.txt`:

```bash
ipython==8.12.3
matplotlib==3.10.1
numpy==1.26.4
opencv_python_headless==4.9.0.80
pandas==2.2.3
Pillow==10.4.0
Requests==2.32.3
scikit_learn==1.5.2
timm==1.0.12
torch==2.5.0
torchaudio==2.5.0
torchvision==0.20.0
tqdm==4.66.6
transformers==4.48.3
```

## Code Examples  
The following notebooks demonstrate how to implement and train deep learning models using the modules described above:

- `image_classification.ipynb` shows the implementation of a transformer-based image classification model using the library.
- `image_distillation.ipynb` shows how to implement model distillation for image-based tasks.
- `audio_waveform_classification.ipynb` and `audio_spectrogram_classification.ipynb` focuses on training and evaluating models for audio classification based on raw waveforms and spectrograms.
- `object_detection_custom.ipynb` and `object_detection_standard.ipynb` to train custom and PyTorch-standard R-CNN models for object detection and segmentation taks.

These notebooks provide hands-on examples of the core functionality of the library.

## Best Practices for Deep Learning Training

### Image Data Augmentation
Data augmentation is of paramount importance to ensure the model's generalization. `TrivialAugmentWide()` is an efficient method that applies diverse image transformations with a single command. This method should be applied during preprocessing of the image dataset to adjust its format (e.g., image resolution, torch.tensor format, color normalization, etc.) to match the network's requirements.

```bash
# Specify transformations
from torchvision.transforms import v2
transform_train = v2.Compose([    
    v2.TrivialAugmentWide(), # Data augmentation
    v2.Resize(256),
    v2.RandomCrop((224, 224)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])
```

### Audio Data Augmentation
Similar to image recognition tasks, generalization in audio recognition can be improved by applying augmentation techniques to the audio signal. Whether pattern recognition is performed in the time domain (waveform) or the frequency domain (spectrograms), the following transformations can be applied:
* Time domain

 ```bash
import torch
import torchaudio
import librosa
# Load audio file
waveform, sample_rate = torchaudio.load("wave.wav", normalize=True)
# Apply pitch shifting
waveform = librosa.effects.pitch_shift(
    waveform.numpy(),           # Audio waveform
    sr=sample_rate,             # Sample rate of the waveform
    n_steps=2                   # Number of semitones to shift the pitch
)
# Add random noise
waveform = waveform + torch.randn_like(waveform) * 0.005

 ```
 * Frequency domain
 ```bash
import torchaudio
# Compute spectrogram to get an image with shape [1, 384, 381]
stride = round(len(waveform) / (384 - 1))
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,    # Sample rate of the waveform
    n_fft=1024,                 # Number of FFT bins       	
    win_length=1024,            # Window length of the waveform for FFT analysis
    hop_length=stride,          # Stride in samples between two analysis windows
    n_mels=384,                 # Number of mel filter banks
    power=2                     # Exponent for the magnitude spectrogram
)
spectrogram = transform(waveform)
# Define 10% masking in the frequency domain
freq_mask_param = int(0.10 * 384)
spectrogram = torchaudio.transforms.FrequencyMasking(
    freq_mask_param=freq_mask_param)(spectrogram)
# Define 10% masking in the time domain
time_mask_param = int(0.10 * 381)
spectrogram = torchaudio.transforms.TimeMasking(
    time_mask_param=time_mask_param)(spectrogram)
 ```

### Training a Classifier
The AdamW optimizer has been shown to improve generalization. Additionally, `CrossEntropyLoss` is the most commonly used loss function in classification tasks, where a certain level of label smoothing (e.g., 0.1) can further enhance generalization. Adding a scheduler for learning rate regulation is also a good practice to optimize parameter updates. An initial learning rate between `1e-4` and `1e-5` and a final learning rate up to `1e-7` are recommended. Optionally, the custom `FixedLRSchedulerWrapper` scheduler can be used to maintain a fixed learning rate in the final epochs, helping stabilize the model parameters.

```bash
# Create AdamW optimizer
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Create loss function
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# Set scheduler: from epoch #1 to #10 use CosinAnnealingRL, from epoch #11 to #20 a fixed learning rate
scheduler = FixedLRSchedulerWrapper(
    scheduler=CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6),
    fixed_lr=1e-6,
    fixed_epoch=10)
```

### Training a Classifier Using Distillation
Distillation is a technique where a smaller, lightweight model (the "student") is trained to mimic the behavior of a larger, pre-trained model (the "teacher"). This approach can significantly reduce complexity and speed up inference while maintaining comparable accuracy.

A custom cross-entropy-based distillation loss function has been created. This loss function consists of a weighted combination of two components:
* Soft Loss (KL divergence): Encourages the student model to match the teacher model’s probability distribution, allowing it to learn fine-grained relationships between classes.
* Hard Loss (cross-entropy): Ensures the student model learns directly from the ground truth labels for correct classification.

A good starting point for configuring this loss function is:

```bash
# Create loss function
loss_fn = DistillationLoss(alpha=0.4, temperature=2, label_smoothing=0.1)
```
where `alpha` controls the weighting between soft and hard losses, `temperature` smooths the teacher’s probability distribution, making it easier for the student to learn from, and `label_smoothing` prevents overconfidence by redistributing a small portion of the probability mass to all classes.


## Contributing
If you want to contribute to this project, contact me via email at **sergio.sanz.rodriguez@gmail.com**.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions, feel free to contact me via email at **sergio.sanz.rodriguez@gmail.com** or connect with me on LinkedIn: [linkedin.com/in/sergio-sanz-rodriguez/](https://www.linkedin.com/in/sergio-sanz-rodriguez/).  

