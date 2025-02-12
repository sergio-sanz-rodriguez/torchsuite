<div align="center">
  <img src="images/logo_tochsuite_1_resized_1.jpg" alt="Into Picture" width="450"/>
</div>

# TorchSuite 
TorchSuite is a versatile and feature-rich PyTorch-based library designed to simplify and accelerate deep learning development.  

It provides essential modules and utilities for training, inference, data handling, and model optimization, making it an invaluable tool for researchers, machine learning professionals, and practitioners.  

Currently, TorchSuite is fully optimized for **image classification tasks**. Future versions will extend support to object detection, segmentation, and other unstructured data types such as audio.  

## Features

- Simplified model training and evaluation
- GPU-accelerated computations
- Support for various deep learning architectures
- Easy-to-use API for experimentation

## Modules  

- **Training and inference engine:** A robust `engine.py` module for seamless training and evaluation workflows.  
- **Flexible data loading:** `dataloaders.py` simplifies dataset preparation and augmentation.  
- **Utility functions:** `helper_functions.py` offers a collection of tools to enhance productivity.  
- **Learning rate scheduling:** `schedulers.py` provides adaptive learning rate strategies.  
- **Vision Transformer (ViT) support:** `vision_transformer.py` enables ViT implementations with PyTorch.  
- **Custom loss functions:** `loss_functions.py` includes various loss formulations for different tasks.  

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
matplotlib==3.10.0
numpy==2.2.2
pandas==2.2.3
Pillow==11.1.0
Requests==2.32.3
scikit_learn==1.6.1
torch==2.5.0
torchvision==0.20.0
tqdm==4.66.6
scikit-image==0.23.1
tifffile>=2022.8.12
```

## Code Examples  
The notebook `classification_example.ipynb` demonstrates how to implement and train a deep learning model using some of the modules described above.

## Best Practices for Deep Learning Training

### Data Augmentation
Data augmentation is of paramount importance to ensure the model's generalization. `TrivialAugmentWide()` is an efficient method that applies diverse image transformations with a single command. This method should be applied during preprocessing of the image dataset to adjust its format (e.g., image resolution, torch.tensor format, color normalization, etc.) to match the network's requirements.

```bash
# Specify transformations
from torchvision.transforms import v2
transform_train = v2.Compose([    
    v2.TrivialAugmentWide(),
    v2.Resize(256),
    v2.RandomCrop((224, 224)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])
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

### ### Training a Classifier Using Distillation


## Contributing
1. Fork the repository.
2. Create a new branch: 
```bash
git checkout -b feature-branch
```
3. Make your changes.
4. Commit your changes:
```bash
git commit -am 'Add new feature'
```
5. Push to the branch:
```bash
git push origin feature-branch
```
6. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions, feel free to contact me via email at **sergio.sanz.rodriguez@gmail.com** or connect with me on LinkedIn: [linkedin.com/in/sergio-sanz-rodriguez/](https://www.linkedin.com/in/sergio-sanz-rodriguez/).  

