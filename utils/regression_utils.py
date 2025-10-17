"""
Provides utility functions for deep learning workflows in PyTorch.  
Includes dataset handling, visualization, model saving/loading, evaluation metrics, and statistical analysis.  

Functions:
- **File & Directory Management**: `walk_through_dir`, `zip_folder`, `download_data`
- **Visualization**: `plot_decision_boundary`, `plot_predictions`, `display_random_images`, `pred_and_plot_image`, `pred_and_plot_image_imagenet`, `plot_loss_curves`, `plot_confusion_matrix`, `plot_class_distribution` 
- **Training & Evaluation**: `print_train_time`, `save_model`, `load_model`, `accuracy_fn`, `get_most_wrong_examples`
- **Reproducibility**: `set_seeds`
- **ROC & AUC Analysis**: `find_roc_threshold_tpr`, `find_roc_threshold_fpr`, `find_roc_threshold_f1`, `find_roc_threshold_accuracy`, `partial_auc_score`, `cross_val_partial_auc_score`
"""

import torch
import random
import matplotlib.pyplot as plt
from typing import Literal, Dict, Union
from .common_utils import theme_presets

def display_random_images_regression(
    dataset: torch.utils.data.dataset.Dataset, # or torchvision.datasets.ImageFolder?
    n: int = 10,
    display_shape: bool = True,
    rows: int = 5,
    cols: int = 5,
    seed: int = None,
    theme: Union[Literal["light", "dark"], Dict[str, str]] = "light"):

   
    """Displays a number of random images from a given dataset for regression.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Dataset to select random images from.
        classes (List[str], optional): Names of the classes. Defaults to None.
        n (int, optional): Number of images to display. Defaults to 10.
        display_shape (bool, optional): Whether to display the shape of the image tensors. Defaults to True.
        rows: number of rows of the subplot
        cols: number of columns of the subplot
        seed (int, optional): The seed to set before drawing random images. Defaults to None.
        theme (str or dict): "light", "dark", or a custom dict with keys 'bg' and 'text'.
    
    Usage:
    display_random_images(
        train_data, 
        n=16, 
        classes=class_names,
        rows=4,
        cols=4,
        display_shape=False,
        seed=None,
        theme='dark')
    """

    # Resolve theme
    if isinstance(theme, dict):
        figure_color_map = theme
    elif theme in theme_presets:
        figure_color_map = theme_presets[theme]
    else:
        raise ValueError(f"Unknown theme '{theme}'. Use 'light', 'dark', or a dict with 'bg' and 'text'.")

    # Setup the range to select images
    n = min(n, len(dataset))

    # Adjust display if n too high
    if n > rows*cols:
        n = rows*cols
        #display_shape = False
        print(f"For display purposes, n shouldn't be larger than {rows*cols}, setting to {n} and removing shape display.")
    
    # Set random seed
    if seed:
        random.seed(seed)

    # Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # Setup plot
    fig = plt.figure(figsize=(cols*8, rows*8))
    fig.patch.set_facecolor(figure_color_map['bg'])

    # Loop through samples and display random samples 
    for i, sample_idx  in enumerate(random_samples_idx):
        img, score  = dataset[sample_idx]

        # Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        #targ_image_adjust = img.permute(1, 2, 0)
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            targ_image_adjust = img.permute(1, 2, 0).cpu().numpy()
        else:
            targ_image_adjust = img

        # Plot adjusted samples
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(targ_image_adjust)
        ax.set_title(f"Score: {score:.2f}", )
        ax.axis("off")
        ax.set_facecolor(figure_color_map['bg'])
        if display_shape:
            print(f"[{i}] shape: {targ_image_adjust.shape}, score: {score:.2f}")
    
    plt.show()
