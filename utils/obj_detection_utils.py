"""
Provides utility functions for deep learning object detection workflows in PyTorch.  
Some functions are based on https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py"
"""

import datetime
import errno
import os
import time
import cv2
import torch
import random
import hashlib
import numpy as np
import torch.distributed as dist
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Literal, Dict, Union
from collections import defaultdict, deque
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F
from torchvision.transforms import v2 as T
from .common_utils import theme_presets

def collate_fn(batch):
    return tuple(zip(*batch))


# Function to remove redundant boxes and masks
def prune_predictions(
    pred,
    score_threshold=0.66,
    iou_threshold=0.5,
    best_candidate="area",
    remove_large_boxes=None
    ):

    """
    Filters and refines predictions by:
    1. Removing unusually large bounding boxes if specified
    2. Removing low-confidence detections based on the score threshold.
    3. Applying a binary mask threshold to filter out weak segmentation masks.
    4. Using Non-Maximum Suppression (NMS) to eliminate overlapping predictions.
    6. Ensuring the highest-scoring prediction is always included.
    7. Selecting the best-confident bounding box based on a criterion: largest area or highest score.

    Args:
        pred: The raw predictions containing "boxes", "scores", "labels", and "masks".
        score_threshold: The minimum confidence score required to keep a prediction (default: 0.66).
        iou_threshold: The Intersection over Union (IoU) threshold for NMS (default: 0.5).
        best_candidate: Selects, from the final set of bounding boxes, the best one based on a criterion:
            -"area": the bounding box with the largest area is chosen
            -"score": the bounding boxe with the highest score is chosen
            -None: no criteion is used, maining the pruning method may contain one or more best bounding box candidates

    Returns:
        A dictionary with filtered and refined predictions:
            "boxes": Tensor of kept bounding boxes.
            "scores": Tensor of kept scores.
            "labels": Tensor of kept labels.
    """
    
    # Validate score_threshold
    if not isinstance(score_threshold, (int, float)) or not (0 <= score_threshold <= 1):
        raise ValueError("'score_threshold' must be a float between 0 and 1")

    # Validate iou_threshold
    if not isinstance(iou_threshold, (int, float)) or not (0 <= iou_threshold <= 1):
        raise ValueError("'iou_threshold' must be a float between 0 and 1")

    # Validate best_candidate
    if best_candidate not in ("area", "score", None):
        raise ValueError("'best_candidate' must be one of: 'area', 'score', or None")
    
    # Validate remove_large_boxes
    if remove_large_boxes is not None and not isinstance(remove_large_boxes, numbers.Number):
        raise ValueError("'remove_large_boxes' must be a numeric value or None")

    # Filter big boxes
    if remove_large_boxes is not None:
        areas = (pred["boxes"][:, 2] - pred["boxes"][:, 0]) * (pred["boxes"][:, 3] - pred["boxes"][:, 1])
        keep_idx = areas < remove_large_boxes
        pred["boxes"] = pred["boxes"][keep_idx]
        pred["scores"] = pred["scores"][keep_idx]
        pred["labels"] = pred["labels"][keep_idx]


    # Filter predictions based on confidence score threshold
    scores = pred["scores"]

    if len(scores) == 0:
        return {
            "boxes": [],
            "scores": [],
            "labels": []
            }

    best_idx = scores.argmax()
    high_conf_idx = scores > score_threshold

    # Extract the best bounding box, score, and label
    best_pred = {
        "boxes": pred["boxes"][best_idx].unsqueeze(0).long(), 
        "scores": pred["scores"][best_idx].unsqueeze(0),
        "labels": pred["labels"][best_idx].unsqueeze(0),
    }

    filtered_pred = {
        "boxes":  pred["boxes"][high_conf_idx].long(),
        "scores": pred["scores"][high_conf_idx],
        "labels": pred["labels"][high_conf_idx], #[f"roi: {s:.3f}" for s in scores[high_conf_idx]]
    }

    # Apply Non-Maximum Suppression (NMS) to remove overlapping predictions
    if len(filtered_pred["boxes"]) == 0:
        if len(best_pred["boxes"]) > 0:
            return best_pred
        else:
            return filtered_pred 
    
    keep_idx = ops.nms(filtered_pred["boxes"].float(), filtered_pred["scores"], iou_threshold)

    # Return filtered predictions
    keep_preds = {
        "boxes": filtered_pred["boxes"][keep_idx],
        "scores": filtered_pred["scores"][keep_idx],
        "labels": filtered_pred["labels"][keep_idx], #[i] for i in keep_idx],
    }

    # Ensure the best prediction is always included
    best_box = best_pred["boxes"][0]
    if not any(torch.equal(best_box, box) for box in keep_preds["boxes"]):
        keep_preds["boxes"] = torch.cat([keep_preds["boxes"], best_pred["boxes"]])
        keep_preds["scores"] = torch.cat([keep_preds["scores"], best_pred["scores"]])
        keep_preds["labels"] = torch.cat([keep_preds["labels"], best_pred["labels"]])
    
    # Now we have a set of good candidates. Let's take the best one based on a criterion
    if keep_preds["boxes"].shape[0] > 1:

        # Return only the one with the highest score
        if best_candidate == "score":            
            idx = keep_preds['scores'].argmax().item()
            final_pred = {
                "boxes": keep_preds["boxes"][idx].unsqueeze(0),
                "scores": keep_preds["scores"][idx].unsqueeze(0),
                "labels": keep_preds["labels"][idx].unsqueeze(0),
            }
            return final_pred

        # Compute area of each box and return the one with the largest area
        elif best_candidate == "area":
            areas = (keep_preds["boxes"][:, 2] - keep_preds["boxes"][:, 0]) * (keep_preds["boxes"][:, 3] - keep_preds["boxes"][:, 1])
            idx = areas.argmax().item()            
            final_pred = {
                "boxes": keep_preds["boxes"][idx].unsqueeze(0),
                "scores": keep_preds["scores"][idx].unsqueeze(0),
                "labels": keep_preds["labels"][idx].unsqueeze(0),
            }
            return final_pred
        
    return keep_preds
       
# Prune predictions v2 with support for multi-class prunning
def prune_predictions_v2(
    pred,
    class_id_to_name={1: 'roi'},
    score_threshold=0.66,
    iou_threshold=0.5,
    best_candidate="area",
    remove_large_boxes=None,
    remove_small_boxes=None,
    apply_cc_prunning=False,
    cc_iou_threshold=0.5,
    keep_best_scoring_box=True
    ):

    """
    Filters and refines object detection predictions.

    Steps performed:
    1. (Optional) Removes boxes that are too large or too small.
    2. For each class:
       - Filters boxes by confidence score.
       - Optionally ensures the single highest-scoring box is kept.
       - Applies Non-Maximum Suppression (NMS) per class.
       - Optionally selects only one box (based on highest score or largest area).
    3. (Optional) Applies cross-class pruning if multiple class predictions overlap.
    4. Returns pruned predictions in the same dict format.

    Args:
        pred (dict): Dictionary with keys "boxes", "scores", and "labels".
        class_id_to_name (dict): Maps class IDs to class names.
        score_threshold (float): Minimum confidence score for keeping predictions.
        iou_threshold (float): IoU threshold for NMS.
        best_candidate (str|None): Strategy for selecting a single box:
            - "area": Keep the largest box.
            - "score": Keep the highest-scoring box.
            - None: Keep all boxes.
        remove_large_boxes (float|None): Remove boxes larger than this area.
        remove_small_boxes (float|None): Remove boxes smaller than this area.
        apply_cc_prunning (bool): If True, apply cross-class NMS.
        cc_iou_threshold (float): IoU threshold for cross-class NMS.
        keep_best_scoring_box (bool): Always keep the highest-scoring box per class.

    Returns:
        dict: {
            "boxes": Tensor of shape [N,4],
            "scores": Tensor of shape [N],
            "labels": Tensor of shape [N]
        }
    """
    
    # Validate score_threshold
    if not isinstance(score_threshold, (int, float)) or not (0 <= score_threshold <= 1):
        raise ValueError("'score_threshold' must be a float between 0 and 1")

    # Validate iou_threshold
    if not isinstance(iou_threshold, (int, float)) or not (0 <= iou_threshold <= 1):
        raise ValueError("'iou_threshold' must be a float between 0 and 1")

    # Validate best_candidate
    if best_candidate not in ("area", "score", None):
        raise ValueError("'best_candidate' must be one of: 'area', 'score', or None")
    
    # Validate remove_large_boxes
    if remove_large_boxes is not None and not isinstance(remove_large_boxes, numbers.Number):
        raise ValueError("'remove_large_boxes' must be a numeric value or None")
    
    if remove_small_boxes is not None and not isinstance(remove_small_boxes, numbers.Number):
        raise ValueError("'remove_large_boxes' must be a numeric value or None")

    # --- Optional box size filtering ---
    # Removes boxes that are too large or too small.
    if remove_large_boxes is not None or remove_small_boxes is not None:
        boxes = pred["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep_idx = torch.ones_like(areas, dtype=torch.bool)
        if remove_large_boxes is not None:
            keep_idx &= areas < remove_large_boxes
        if remove_small_boxes is not None:
            keep_idx &= areas > remove_small_boxes

        for key in ["boxes", "scores", "labels"]:
            pred[key] = pred[key][keep_idx]

    # If only one class exists, cross-class pruning is meaningless.
    if len(class_id_to_name) == 1:
        apply_cc_prunning = False

    final_boxes = []
    final_scores = []
    final_labels = []
   
    final_predictions = []

    # Process each class separately
    for class_id, class_name in class_id_to_name.items():

        # Find indices for this class
        #indexes = [i for i, label in enumerate(pred["labels"]) if label == key]
        idxs = (pred["labels"] == class_id).nonzero(as_tuple=True)[0]
        
        if idxs.numel() == 0:
            continue

        class_scores = pred["scores"][idxs]
        class_boxes = pred["boxes"][idxs]
        class_labels = pred["labels"][idxs]

        # Filter by score threshold
        keep_scores_mask = class_scores > score_threshold

        # Boxes and scores that pass threshold
        filtered_boxes = class_boxes[keep_scores_mask]
        filtered_scores = class_scores[keep_scores_mask]
        filtered_labels = class_labels[keep_scores_mask]

        # Always keep best scoring box (even if below threshold)
        if keep_best_scoring_box:

            best_score_idx = class_scores.argmax().item()

            # If no boxes pass threshold, fallback to best scoring box only
            if filtered_boxes.shape[0] == 0:
                filtered_boxes = class_boxes[best_score_idx].unsqueeze(0)
                filtered_scores = class_scores[best_score_idx].unsqueeze(0)
                filtered_labels = class_labels[best_score_idx].unsqueeze(0)

        # Non-Maximum Suppression (NMS)
        keep_nms_idx = ops.nms(filtered_boxes.float(), filtered_scores, iou_threshold)
        nms_boxes = filtered_boxes[keep_nms_idx]
        nms_scores = filtered_scores[keep_nms_idx]
        nms_labels = filtered_labels[keep_nms_idx]

        # Ensure best box is included even if NMS removed it
        if keep_best_scoring_box:
            best_box = class_boxes[best_score_idx]
            if not any(torch.all(best_box == b) for b in nms_boxes):
                nms_boxes = torch.cat([nms_boxes, best_box.unsqueeze(0)])
                nms_scores = torch.cat([nms_scores, class_scores[best_score_idx].unsqueeze(0)])
                nms_labels = torch.cat([nms_labels, class_labels[best_score_idx].unsqueeze(0)])

        # Select best candidate if multiple remain
        if nms_boxes.shape[0] > 1 and best_candidate is not None:

            if best_candidate == "score":
                best_idx = nms_scores.argmax().item()
            elif best_candidate == "area":
                areas = (nms_boxes[:, 2] - nms_boxes[:, 0]) * (nms_boxes[:, 3] - nms_boxes[:, 1])
                best_idx = areas.argmax().item()
            else:
                best_idx = 0  # fallback to first box

            nms_boxes = nms_boxes[best_idx].unsqueeze(0)
            nms_scores = nms_scores[best_idx].unsqueeze(0)
            nms_labels = nms_labels[best_idx].unsqueeze(0)

        # Collect final results per class
        final_boxes.append(nms_boxes)
        final_scores.append(nms_scores)
        final_labels.append(nms_labels)

    # Return if no boxes remain
    if len(final_boxes) == 0:
        return {"boxes": [], "scores": [], "labels": []}
    
    # Apply Cross-class pruning (optional)
    if apply_cc_prunning:
        # Combine final results
        all_boxes = torch.cat(final_boxes, dim=0)
        all_scores = torch.cat(final_scores, dim=0)
        all_labels = torch.cat(final_labels, dim=0)

        # Run cross-class NMS to remove duplicates with lower scores
        keep = ops.nms(all_boxes, all_scores, cc_iou_threshold)  # same threshold can work

        return {
            "boxes": all_boxes[keep],
            "scores": all_scores[keep],
            "labels": all_labels[keep],
        }
    
    # No cross-class pruning
    return {
        "boxes": torch.cat(final_boxes, dim=0),
        "scores": torch.cat(final_scores, dim=0),
        "labels": torch.cat(final_labels, dim=0),
    }


# Function to display images with masks and boxes on the ROIs
def display_and_save_predictions(
    preds: List=None,
    dataloader: torch.utils.data.Dataset | torch.utils.data.DataLoader = None,
    box_color: str='white',
    mask_color: str='blue',
    width: int=1,
    font_type: str=None,
    font_size: int=8,
    print_classes: bool=True,
    print_scores: bool=True,
    label_to_class_dict={1: 'roi'},
    save_dir: str = None,
    theme: Union[Literal["light", "dark"], Dict[str, str]] = "light"):

    """
    This function displays images with predicted bounding boxes and segmentation masks.
    Arguments:
        preds (List): A list of predictions, each containing 'boxes', 'labels', 'scores', and optionally 'masks'.
        dataloader (torch.utils.data.DataLoader): A DataLoader object containing the images.
        box_color (str): Color of the bounding boxes drawn on the image.
        mask_color (str): Color of the segmentation masks drawn on the image.
        width (int): The width of the bounding box lines.
        print_classes (bool): If True, the labels will be printed on the bounding boxes.
        print_scores (bool): If True, the confidence scores will be printed on the bounding boxes.
        label_to_class_dict (dict): Dictionary mapping label indices to class names.
        save_dir (str, optional): Path to save images. If None, images will not be saved.
        theme (str or dict): "light", "dark", or a custom dict with keys 'bg' and 'text'.
    """

    plt.close("all")

    # Resolve theme
    if isinstance(theme, dict):
        figure_color_map = theme
    elif theme in theme_presets:
        figure_color_map = theme_presets[theme]
    else:
        raise ValueError(f"Unknown theme '{theme}'. Use 'light', 'dark', or a dict with 'bg' and 'text'.")

    # Convert dataset to DataLoader if needed
    if isinstance(dataloader, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Create save directory if saving images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    # Number of images
    num_images = len(preds)
    cols = 3 
    rows = (num_images + cols - 1) // cols

    # Set up the grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.patch.set_facecolor(figure_color_map['bg'])
    axes = axes.flatten()

    # Loop through the predictions and process each image
    for idx, (data, filtered_pred) in enumerate(zip(dataloader.dataset, preds)):  

        # Get the image from the dataset
        image, _ = data
        
        # Prepare the image
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)

        # Taking the first 3 channels if it's RGB
        image = image[:3, ...]  

        # Replace labels with strings for proper display
        labels = [
            f"{label_to_class_dict[l.item()] if print_classes else ''}{': ' + f'{s.item():.3f}' if print_scores else ''}".strip(": ")
            for l, s in zip(filtered_pred["labels"], filtered_pred["scores"])
        ]

        # Draw bounding boxes
        if len(filtered_pred["boxes"]) > 0:
            output_image = draw_bounding_boxes(
                image=image,
                boxes=filtered_pred["boxes"],
                labels=labels if print_classes or print_scores else None,
                colors=box_color,
                width=width,
                font=font_type,
                font_size=font_size)
        else:
            output_image = image

        # Save Image (if save_dir is provided)
        if save_dir:
            image_pil = to_pil_image(output_image)  # Convert tensor to PIL Image
            image_path = os.path.join(save_dir, f"prediction_{idx+1}.png")
            image_pil.save(image_path)

        # Plot on the grid
        ax = axes[idx]
        ax.imshow(output_image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.set_title(f"Prediction {idx + 1}", color=figure_color_map['text'])
        ax.axis("off")

    # Hide unused subplots (if any)
    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_transformed_data(
        img,
        target,
        transformed_img,
        transformed_target,
        color_conversion=None,
        theme: Union[Literal["light", "dark"], Dict[str, str]] = "light"):

    """
    Visualizes the original and transformed image along with bounding boxes and masks.
    
    Args:
        img: Original image tensor.
        target: Original target dictionary (contains boxes, masks, labels).
        transformed_img: Transformed image tensor.
        transformed_target: Transformed target dictionary.
        color_conversion: Optional OpenCV color conversion code (e.g., cv2.COLOR_HSV2RGB).
                        https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
                        If None, assumes image is already in RGB.
        theme (str or dict): "light", "dark", or a custom dict with keys 'bg' and 'text'.
    """

    # Resolve theme
    if isinstance(theme, dict):
        figure_color_map = theme
    elif theme in theme_presets:
        figure_color_map = theme_presets[theme]
    else:
        raise ValueError(f"Unknown theme '{theme}'. Use 'light', 'dark', or a dict with 'bg' and 'text'.")

    # Visualize original image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(figure_color_map['bg'])

    # Convert tensors to numpy and apply color conversion if needed
    def convert_for_plot(tensor_img):
        img_np = (tensor_img * 255).byte().permute(1, 2, 0).cpu().numpy() # Convert CHW to HWC for plotting
        if color_conversion is not None:
            img_np = cv2.cvtColor(img_np, color_conversion)
        return img_np
    
    img = convert_for_plot(img)
    transformed_img = convert_for_plot(transformed_img)
    
    # Original Image
    axes[0].imshow(img)
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[0].add_patch(rect)
    axes[0].set_title('Original Image', color=figure_color_map['text'])
    axes[0].axis('off')

    # Transformed Image
    axes[1].imshow(transformed_img)
    for box in transformed_target['boxes']:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        axes[1].add_patch(rect)
    axes[1].set_title('Transformed Image', color=figure_color_map['text'])
    axes[1].axis('off')

    plt.show()


class RandomCircleOcclusion(T.RandomErasing):

    """
    Applies random circular occlusions to an image tensor. Inherits from torchvision.transforms.RandomErasing.

    This transform overlays multiple randomly placed circles of various sizes and predefined colors onto 
    the image. It is useful for data augmentation to simulate occlusions or visual noise.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), num_elems=6):

        """
        Initializes the RandomCircleOcclusion class.

        Args:
            p (float, optional): Probability of applying the occlusion. Defaults to 0.5.
            scale (tuple, optional): Range of scale factors for the occlusion. Defaults to (0.02, 0.2).
            ratio (tuple, optional): Range of aspect ratios for the occlusion. Defaults to (0.3, 3.3).

        Note:
            The occlusion color is set to a forest green color.
        """

        # Define the forest green color in a tuple format (R, G, B)
        forest_green = (34/255, 139/255, 34/255)  # Normalized RGB values
        self.colors = [
            (0, 0, 0),          # Black
            (53, 94, 59),       # Dark Green
            (211, 211, 211),    # Light Gray
            (34, 139, 34),      # Forest Green
        ]
        
        super().__init__(p=p, scale=scale, ratio=ratio, value=forest_green)
        self.num_elems = num_elems

    def forward(self, img, target=None):

        """
        Applies the random circular occlusion to the input image tensor.

        Args:
            img (Tensor): Image tensor with shape (C, H, W).
            target (optional): Target corresponding to the image, returned unchanged.

        Returns:
            Tuple[Tensor, Any]: Transformed image and original target.
        """

        if torch.rand(1).item() > self.p:
            return img, target

        img_np = img.mul(255).byte().permute(1, 2, 0).numpy()  # Convert tensor to uint8 numpy
        h, w, _ = img_np.shape

        # Generate a random occlusion mask
        num_blobs = np.random.randint(2, self.num_elems)
        for _ in range(num_blobs):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            size = np.random.randint(h // 20, h // 5)
            selected_color = np.array(self.colors[np.random.randint(len(self.colors))], dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), size, 255, -1)

            # Apply selected color
            img_np[mask > 0] = selected_color

        img_tensor = torch.from_numpy(img_np).float().div(255).permute(2, 0, 1)  # Convert back to tensor
        return img_tensor, target


class RandomTextureOcclusion:

    """
    Applies random occlusions to an image using RGBA texture objects (e.g., plants) 
    to simulate natural occlusions. Occlusions are applied stochastically with a 
    user-defined probability.

    This class is useful for data augmentation in computer vision tasks such as object 
    detection or classification, where robustness to occlusion is desired.
    """

    def __init__(self, obj_path, scale=(0.2, 0.5), transparency=0.5, p=0.5):

        """
        Initializes the RandomTextureOcclusion class.

        Args:
            obj_path (list): List of paths to plant images.
            scale (tuple, optional): Range of scale factors for the occlusion. Defaults to (0.2, 0.5).
            transparency (float, optional): Transparency of the plant image. Defaults to 0.5.
            p (float, optional): Probability of applying the occlusion. Defaults to 0.5.
        """

        #obj_path = ["T_Bush_Falcon.png"]
        obj_images = [Image.open(path).convert("RGBA") for path in obj_path]
        self.scale = scale
        self.obj_images = obj_images 
        self.transparency = transparency
        self.p = p

    def __call__(self, img, target=None):

        """
        Applies a randomly placed, scaled, and rotated texture occlusion to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image.
            target: Optional target data (e.g., labels or bounding boxes).

        Returns:
            Tuple: Transformed image as torch.Tensor, and the unmodified target.
        """

        if random.random() > self.p or not self.obj_images:
            return img, target

        # Convert to PIL Image if necessary
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Select a random plant image        
        obj_img = random.choice(self.obj_images)
        
        # Resize plant to a random size
        w, h = img.size       
        scale = random.uniform(self.scale[0], self.scale[1])  # Random scale factor
        new_h, new_w = int(w * scale), int(w * scale)
        new_h, new_w = min(new_w, w), min(new_h, h)
        obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random rotation
        angle = random.uniform(0, 360)
        obj_img = obj_img.rotate(angle, expand=True)
        # Make sure it fits into base image (w, h)
        new_w, new_h = obj_img.size
        if new_w >= w or new_h >= h:
            scale = min(w / new_w, h / new_h) * 0.8  # a bit smaller than max
            new_w, new_h = int(new_w * scale), int(new_h * scale)
            obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random placement
        x_offset = random.randint(0, w - new_w)
        y_offset = random.randint(0, h - new_h)        

        # Convert plant image to numpy for blending
        obj_np = np.array(obj_img).astype(float)
        img_np = np.array(img).astype(float)

        # Alpha channel
        obj_alpha = obj_np[:, :, 3] > 128
        
        # Apply transparency
        # Blend the RGB channels with the alpha channel
        for c in range(3):  # Iterate over RGB channels
            img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                obj_alpha * obj_np[:, :, c] + (1 - obj_alpha) * img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )

        # Convert back to PIL
        img_occluded = Image.fromarray(img_np.astype(np.uint8))
        
        return F.pil_to_tensor(img_occluded), target


class RandomTextureOcclusionDeterministic:

    """
    Applies deterministic occlusions using RGBA texture objects (e.g., plants) based on 
    the hash of the input image or its filename. This ensures reproducibility of the 
    augmentation, which is useful for validation or consistent visual testing.
    """

    def __init__(self, obj_path, scale=(0.2, 0.5), transparency=0.5, p=0.5):
        
        """
        Initializes the RandomTextureOcclusionDeterministic class.

        Args:
            obj_path (list): List of paths to RGBA plant images to be used for occlusion.
            scale (tuple, optional): Range of scale factors for the occlusion. Defaults to (0.2, 0.5).
            transparency (float): Transparency level for occlusions (not currently used).
            p (float): Probability of applying the occlusion.
        """

        self.obj_images = [Image.open(path).convert("RGBA") for path in obj_path]
        self.scale = scale
        self.transparency = transparency
        self.p = p

    def __call__(self, img, target=None):

        """
        Applies a reproducible texture occlusion to the image using a hash-based seed 
        (from the filename or image bytes).

        Args:
            img (PIL.Image or torch.Tensor): Input image.
            target: Optional target data (e.g., labels or bounding boxes).

        Returns:
            Tuple: Transformed image as torch.Tensor, and the unmodified target.
        """
                
        # Convert to PIL if needed
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Try to extract filename-based seed
        try:
            filename = img.filename  # Works if loaded via PIL.Image.open(path)
            seed = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % (2**32)
        except Exception:
            # If filename not available, hash first 1000 bytes of image content
            img_bytes = np.array(img).tobytes()[:1000]
            seed = int(hashlib.sha256(img_bytes).hexdigest(), 16) % (2**32)

        rng = random.Random(seed)

        if rng.random() > self.p or not self.obj_images:
            if isinstance(img, Image.Image):
                img = F.pil_to_tensor(img)
            return img, target

        # Choose plant
        obj_img = rng.choice(self.obj_images)

        # Resize randomly
        w, h = img.size
        scale = random.uniform(self.scale[0], self.scale[1])  # Random scale factor
        new_w, new_h = int(w * scale), int(w * scale)
        new_w, new_h = min(new_w, w), min(new_h, h)
        obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random rotation
        angle = rng.uniform(0, 360)
        obj_img = obj_img.rotate(angle, expand=True)
        # Make sure it fits into base image (w, h)
        new_w, new_h = obj_img.size
        if new_w >= w or new_h >= h:
            scale = min(w / new_w, h / new_h) * 0.8  # a bit smaller than max
            new_w, new_h = int(new_w * scale), int(new_h * scale)
            obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random placement
        x_offset = rng.randint(0, w - new_w)
        y_offset = rng.randint(0, h - new_h)

        # Convert to NumPy
        obj_np = np.array(obj_img).astype(float)
        img_np = np.array(img).astype(float)

        # Alpha channel
        obj_alpha = obj_np[:, :, 3] > 128

        # Apply transparency
        # Blend the RGB channels with the alpha channel
        for c in range(3):  # Iterate over RGB channels
            img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                obj_alpha * obj_np[:, :, c] + (1 - obj_alpha) * img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )

        #img_occluded = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        # Convert back to PIL
        img_occluded = Image.fromarray(img_np.astype(np.uint8))

        return F.pil_to_tensor(img_occluded), target