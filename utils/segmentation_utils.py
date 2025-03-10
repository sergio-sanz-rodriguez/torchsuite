import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
sys.path.append(os.path.abspath("../engines"))
from engines.common import Logger

def num_to_rgb(num_arr, color_map: Dict=None):
    
    """
    Converts a numpy array of class indices to an RGB image.

    Parameters:
    - num_arr (np.ndarray): The input array of class indices.
    - color_map (Dict, optional): A dictionary mapping class indices to RGB colors. If None, a default color map is used.

    Returns:
    - np.ndarray: The RGB image as a numpy array.
    """

    if not isinstance(num_arr, np.ndarray):
        Logger().error("The input array is not a numpy array. Please provide a numpy array.")

    if len(num_arr.shape) != 2:
        Logger().error("The input array is not 2D. Please provide a 2D numpy array.")

    if color_map == None:
        Logger().error("The color map for the classes is not defined. Please define {idx: [R, G, B], ...}")

    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
 
    for k in color_map.keys():
        output[single_layer == k] = color_map[k]
 
    # return a floating point array in range [0.0, 1.0]
    return np.float32(output) / 255.0

# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, mask, alpha=1.0, beta=0.5, gamma=0.0):
    
    """
    Adds an overlay to the image:
    - image (Tensor): Original image
    - mask (Tensor): The segmented image in form of mask
    - alpha (Float): Transparency for the original image.
    - beta (Float):  Transparency for the segmentation map.
    - gamma (Float): Scalar added to each sum.

    Returns:
    - Overlaid image
    """

    # Convert Tensors to color images
    mask =  cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply the overlay
    image = cv2.addWeighted(image, alpha, mask, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, 0.0, 1.0)


def display_image_with_mask(image, mask, ax=None, alpha=1.0, beta=0.5, gamma=0.0, color_map=None, title=None):
    
    """
    Displays an image with its corresponding mask overlaid.

    Parameters:
    - image (Tensor): The image tensor to be displayed (C, H, W)
    - mask (Tensor): The mask tensor to be overlaid on the image (H, W)
    - ax (matplotlib.axes.Axes, optional): The subplot axis to plot on. If None, creates a new figure.
    - alpha (Float): Transparency for the original image.
    - beta (Float):  Transparency for the segmentation map.
    - gamma (Float): Scalar added to each sum.
    - title (Str, optional): Title of the subplot.
    """

    if color_map == None:
        Logger().error("The color map for the classes is not defined. Please define {idx: [R, G, B], ...}")

    # Convert tensors to numpy arrays for displaying
    image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

    # Ensure mask has correct shape (H, W)
    mask = mask.squeeze(0).cpu().numpy()  # Remove the first dimension if it's 1
    if mask.ndim == 1:
        mask = mask.reshape(image.shape[0], image.shape[1])

    # Convert mask to RGB for overlay
    mask = num_to_rgb(mask, color_map=color_map)
    image = image_overlay(image, mask, alpha, beta, gamma)

    # If no axis is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))  # Create a new figure if no axis is given

    # Plot the image
    ax.imshow(image)

    # Set title if provided
    if title:
        ax.set_title(title)

    # Turn off axis for clarity
    ax.axis('off')

    # If ax was None (new figure), show the plot
    if ax is None:
        plt.show()


def visualize_original_and_transformed(img_nt, target_nt, img_t, target_t, alpha):
    
    """
    Visualize original and transformed images with their corresponding masks.
    
    Parameters:
    - img_nt (Tensor): Original image from non-transformed dataset.
    - target_nt (dict): Target containing masks and other info for the original image.
    - img_t (Tensor): Transformed image from the transformed dataset.
    - target_t (dict): Target containing masks and other info for the transformed image.
    - alpha (float): Overlay to plot img and mask in the same image
    """
    
    # Convert tensors to numpy arrays for displaying
    img_nt = img_nt.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    img_t = img_t.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

    # Extract the masks (assuming binary masks with shape [num_objs, H, W])
    # Adjust this if you need multiple object masks
    mask_nt = target_nt["masks"][0].squeeze(0).cpu().numpy()  # Remove the first dimension if it's 1 (num_objs, H, W)
    mask_t = target_t["masks"][0].squeeze(0).cpu().numpy()  # Same here for the transformed mask

    # Ensure mask has correct shape (H, W)
    if mask_nt.ndim == 1:
        mask_nt = mask_nt.reshape(img_nt.shape[0], img_nt.shape[1])
    if mask_t.ndim == 1:
        mask_t = mask_t.reshape(img_t.shape[0], img_t.shape[1])

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image and mask with overlay
    axes[0].imshow(img_nt)
    axes[0].imshow(mask_nt, alpha=0.5, cmap='jet')
    axes[0].set_title('Original Image with Mask')
    axes[0].axis('off')  # Turn off axis for clarity

    # Plot the transformed image and mask
    axes[1].imshow(img_t)
    axes[1].imshow(mask_t, alpha=0.5, cmap='jet')
    axes[1].set_title('Transformed Image with Mask')
    axes[1].axis('off')  # Turn off axis for clarity

    # Show the plot
    plt.show()


def collapse_one_hot_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Converts a one-hot encoded mask (C, H, W) to a single-channel mask (H, W).
    
    Args:
        mask (torch.Tensor): One-hot encoded mask of shape (C, H, W),
                             where C is the number of classes.
    
    Returns:
        torch.Tensor: Single-channel mask of shape (H, W), where each pixel 
                      contains the class index.
    """

    # Add an extra channel of zeros to the mask
    #extra_channel = torch.zeros(1, mask.shape[1], mask.shape[2])  # (1, H, W)
    #mask_with_extra = torch.cat((extra_channel, mask), dim=0) # Concatenate along the channel dimension (C -> C+1)
        
    # Perform argmax along the class dimension to collapse the one-hot encoded mask
    #collapsed_mask = torch.argmax(mask_with_extra, dim=0)

    collapsed_mask = torch.argmax(mask, dim=0)
    
    return collapsed_mask


def create_label_class_dict(mask, target_categories):
    """
    Creates a dictionary mapping label values to class names from a segmentation mask.

    Args:
        mask (Tensor): Segmentation mask (1D or 2D Tensor).
        target_categories (dict): A dictionary that maps label values to class names.

    Returns:
        dict: A dictionary with label as keys and class names as values.
    """
    
    # Pass subplot axes to the function
    mask = collapse_one_hot_mask(mask)    
    
    # Identify nonzero categories
    unique_labels = mask.unique()
    #unique_labels = unique_labels[unique_labels != 0]
    
    # Create the dictionary mapping labels to class names
    label_class_dict = {label.item(): target_categories.get(label.item(), 'Unknown') for label in unique_labels}
    
    return label_class_dict
