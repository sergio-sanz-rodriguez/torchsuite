# Backend functions
import os
import torch
import numpy as np
from PIL import Image
import torchvision.ops as ops
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2, ToPILImage
from constants import (
    MAX_DIM, DEFAULT_CONFIDENCE, DEFAULT_DIMM, DEFAULT_DOWNSCALE,
    DEFAULT_BOX_COLOR, LABEL_TO_CLASS_DICT, IOU_THRESHOLD,
    PRINT_CLASSES, PRINT_SCORES
)
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)
from ultralytics import YOLO

# Initialze state
def fresh_state():

    """
    Creates and returns a new empty state dictionary.

    This state object stores:
    - original image
    - transformed image
    - detection boxes/scores/labels
    - index of the current bounding box to display
    - any other metadata needed across callbacks
    """
    
    return {
        "image_orig": None,
        "image_prev": None,
        "image_tr":   None,
        "image_out":  None,
        "downscale":  DEFAULT_DOWNSCALE,
        "all_boxes":  [],
        "all_scores": [],
        "all_labels": [],
        "boxes":      [],
        "scores":     [],
        "labels":     [],        
        "splits":     [],
        "n_items":    0,
        "index":      0,
        "conf":       None,
        "color":      None,
        "alpha":      None,
    }

# Prune predictions
def prune_predictions(
        boxes,
        scores,
        score_th=DEFAULT_CONFIDENCE):
    
    """
    Filters predictions by confidence and IoU threshold.

    Parameters:
        boxes (torch.Tensor): [N,4] bounding boxes (xyxy)
        scores (torch.Tensor): [N] confidence scores        
        conf (float): Default confidence or score threshold

    Returns:
        keep_mask (torch.BoolTensor): mask of boxes to keep
    """

    # Define thresholds based on strictness level
    #thresholds = {
    #    "low":    {"score_threshold": 0.05,  "iou_threshold": 0.90},
    #    "medium": {"score_threshold": 0.30,  "iou_threshold": 0.50},
    #    "high":   {"score_threshold": 0.85,  "iou_threshold": 0.05}
    #}

    iou_th = IOU_THRESHOLD

    # Filter by score
    high_conf_mask = scores > score_th
    if high_conf_mask.sum() == 0:
        return torch.zeros_like(scores, dtype=torch.bool)

    filtered_boxes = boxes[high_conf_mask].float()
    filtered_scores = scores[high_conf_mask]

    # Apply class-agnostic NMS
    keep_idx = ops.nms(filtered_boxes, filtered_scores, iou_th)

    # Build final mask in original size
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    orig_idx = high_conf_mask.nonzero(as_tuple=True)[0]
    keep_mask[orig_idx[keep_idx]] = True

    return keep_mask

# Image comparison function
def same_images(
        image1: Image,
        image2: Image):
    
    """
    Compare two images pixel-wise.

    Args:
        image1 (PIL.Image): The first image.
        image2 (PIL.Image): The second image.

    Returns:
        bool: True if the images are identical, False otherwise.
    """

    # Convert images to NumPy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Compare pixel arrays
    return np.array_equal(img1_array, img2_array)

# Specify manual transforms
transform_yolo = v2.Compose([    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.ToPureTensor()
    ])

# Initialize ToPILImage transform
to_pil = ToPILImage()


# Load YOLO model
def load_yolo(model_path):
    """
    Loads a YOLO model from the given path and set it to evaluation mode.

    Args:
        model_path (str): Path to the YOLO .pt model file.

    Returns:
        YOLO: Loaded YOLO model ready for inference.

    Raises:
        ValueError: If the model cannot be loaded (e.g., invalid path or file).
    """

    try:
        print(f"Loading YOLO model from '{model_path}' ...")
        model = YOLO(model_path)
        model.eval()  # set to evaluation mode
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        raise ValueError(f"Model file not found at '{model_path}'. Check the path.")
    except RuntimeError as e:
        raise ValueError(f"Error loading the YOLO model: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error while loading model: {e}")

# Function to dimm regions outside the bounding boxes
def dimm_background(image: torch.Tensor,
                    boxes: torch.Tensor,
                    alpha: float):
    
    """
    Dimms background by masking only the areas defined by bounding boxes.

    Parameters:
        image: (C, H, W) PyTorch tensor representing the image.
        boxes: (N, 4) Tensor with bounding boxes [x1, y1, x2, y2].
        alpha: background dim factor in [0, 1].

    Returns:
        Masked image as a PyTorch tensor and float
        Mask as a PyTorch tensor and float
    """

    # Convert image if needed
    if image.max() > 1:
        image = image / 255.0

    _, H, W = image.shape

    # Create the base mask filled with alpha
    mask = torch.full((H, W), alpha, dtype=image.dtype, device=image.device)

    if boxes is not None and len(boxes) > 0:

        # Round/clip coordinates
        boxes = boxes.clamp(min=0)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=H)
        boxes = boxes.to(torch.int64)

        # Build index grids
        y = torch.arange(H, device=image.device)
        x = torch.arange(W, device=image.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # For each box produce a boolean mask and combine them
        box_masks = []
        for (x1, y1, x2, y2) in boxes:
            m = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)
            box_masks.append(m)

        if box_masks:
            combined_mask = torch.stack(box_masks).any(dim=0)
            mask[combined_mask] = 1.0

    # Align the dimensions of the mask with the image
    mask = mask.unsqueeze(0)

    # Return dimmed image and mask
    return (image * mask).float(), mask.float()

def split_items(
        image_in,
        image_out,
        state):

    """
    Returns a list of PIL images for each detected box.
    Parameters:
        image_in: PIL.Image.Image: input image
        image_out: PIL.Image.Image: output image with the bounding boxes
        state: dict containing a list or tensor of boxes [x1, y1, x2, y2]

    Returns:
        A list of split images + the image_out
    """

    # Nothing to split
    if image_in is None or 'all_boxes' not in state or len(state['all_boxes']) == 0:
        return []

    splits = []
    W, H = image_in.size
    boxes = state['all_boxes']
    mask = state['mask']
    for box in boxes[mask]:
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            crop = image_in.crop((x1, y1, x2, y2))
            splits.append(crop)
    splits.append(image_out)

    # Returns the list of PIL images
    return splits

def prev_split_(
        image_in,
        image_out,
        state):
    
    """
    Shows previous bounding box in the list
    Parameters:
        image_in: PIL.Image.Image: input image
        image_out: PIL.Image.Image: output image with the bounding boxes (bypass)
        state: dict containing 'preds_boxes' as a list or tensor of boxes [x1, y1, x2, y2]
        split_state: dict containing the list of bounding-box images and indexes

    Returns:
        The image associated with actual index and the state dict
    """

    splits = state["splits"]
    index =  state["index"]
    n_items = len(splits)

    # Nothing to show
    if n_items == 0:
        splits = split_items(image_in, image_out, state)
        n_items = len(splits)
        if n_items == 0:
            return image_out, fresh_state()
        index = 1
    
    # Move to prev index (wrap around)
    index = (index - 1) % n_items
    
    return splits[index], {"splits": splits, "index": index}

def next_split_(
        image_in,
        image_out,
        state):
    
    """
    Shows next bounding box in the list
    Parameters:
        image_in: PIL.Image.Image: input image
        image_out: PIL.Image.Image: output image with the bounding boxes (bypass)
        state: dict containing 'preds_boxes' as a list or tensor of boxes [x1, y1, x2, y2]
        split_state: dict containing the list of bounding-box images and indexes

    Returns:
        The image associated with actual index and the state dict
    """

    splits = state["splits"]
    index =  state["index"]
    n_items = len(splits)

    # Nothing to show
    if n_items == 0:
        splits = split_items(image_in, image_out, state)
        n_items = len(splits)
        if n_items == 0:
            return image_out, fresh_state()
        index = -1
    
    # Move to next index (wrap around)
    index = (index + 1) % n_items

    return splits[index], {"splits": splits, "index": index}

def prev_split(state):

    """
    Shows previous bounding box in the list
    Parameters:
        state: dict containing 'preds_boxes' as a list or tensor of boxes [x1, y1, x2, y2]

    Returns:
        Updated state
    """

    state["index"] = (state["index"] - 1) % (state["n_items"] + 1)

    return state

def next_split(state):

    """
    Shows next bounding box in the list
    Parameters:
        state: dict containing 'preds_boxes' as a list or tensor of boxes [x1, y1, x2, y2]

    Returns:
        Updated state
    """

    state["index"] = (state["index"] + 1) % (state["n_items"] + 1)

    return state

def downscale_status(enable, state):

    """
    Updates the 'downscale' status flag in the given state dictionary.

    Parameters:
        enable (bool): Whether downscaling should be enabled or disabled.
        state (dict): A dictionary representing the current state.

    Returns:
        dict: The updated state dictionary with the new 'downscale' value.
    """

    state["downscale"] = enable
    
    return state

def downscale(image: Image):

    """
    Downscales the input PIL image so that its width and height
    do not exceed MAX_DIM, preserving aspect ratio.

    Parameters:
        image (PIL.Image): The image to resize.

    Returns:
        PIL.Image: The resized image if downscaling is needed,
                   otherwise the original image.
    """

    w, h = image.size
    scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
    if scale < 1.0:
        return image.resize((round(w*scale), round(h*scale)), Image.LANCZOS)
    
    return image

def show_analyzing(image, state):

    """
    Returns the input image together with an 'Analyzing...' message
    and the current state. Used to immediately update the UI to
    indicate that processing has started.

    Parameters:
        image (any): The input image (or object expected by the UI).
        state (any): The current application state.

    Returns:
        tuple: (image, "Analyzing...", state)
    """

    # Nothing to show when clearing
    if image is None:
        return None, "", "", fresh_state()

    return image, "Analyzing...", "", state

def compute_confidence(score):

    """
    Computes the confidence percentage based on an score.
    
    Parameters:
        score (Tensor): confidence score
    
    Returns:
        Confidence in string format
    """

    if len(score) == 1:
        confidence = f"{score.item()*100:.1f}%"
    else:
        confidence = "-"
        
    return confidence

def compose_labels(labels, scores):

    """
    Composes text labels for bounding boxes, including class names and/or scores.

    Each label is created based on the corresponding class ID and score. The class name
    is retrieved from LABEL_TO_CLASS_DICT if PRINT_CLASSES is True. The score is included 
    if PRINT_SCORES is True.

    Parameters:
        labels (list of torch.Tensor): Class IDs corresponding to each bounding box.
        scores (list of torch.Tensor): Confidence scores corresponding to each bounding box.

    Returns:
        list of str: A list of formatted label strings, one per box.
    """

    return [
        f"{LABEL_TO_CLASS_DICT[l.item()] if PRINT_CLASSES else ''}"
        f"{': ' + f'{s.item():.3f}' if PRINT_SCORES else ''}".strip(": ")
        for l, s in zip(labels, scores)
    ] 

def detect_items(
        image,
        conf=DEFAULT_CONFIDENCE,
        color=DEFAULT_BOX_COLOR,
        dimm=DEFAULT_DIMM,
        state=None,
        model=None):        

    """
    Transforms and performs a prediction on the image and returns prediction details.
    Args:
        image (torch.Tensor): Input tensor representing the image.
        preset (str): Model preset balancing speed and accuracy.
        color (str): Color of the bounding box.
        dimm (float): Set the dimm level of the background
        state (dict): Contains previously stored predictions to avoid redundant model inference.
        state (YOLO): YOLO model object
    Returns:
        Tuple[image, int, dict]: A tuple containing the output image with bounding boxes,
                               count of detected objects, and the prediction time.
    """
    
    # Check model
    if model is None:
        raise ValueError("No model provided for detection.")
    
    # Nothing to show for the first time
    if state is None:        
        state = fresh_state()

    try:

        # Nothing to show if image is None
        if image is None:        
            return None, "", "", fresh_state()

        # Downscale large images to avoid performance or memory issues        
        if state["downscale"]:            
            image = downscale(image)
        
        #  Compute drawing parameters
        alpha = 1 - dimm
        thickness1 = round(max(image.size[0], image.size[1])/175)
        thickness2 = int(thickness1*0.5)
        color_str, fill_str = color.split("|")
        fill_bool = fill_str == "True"

        # Detect items if new image        
        if (state["image_orig"] is None) or (not same_images(image, state["image_orig"])):
        
            # Convert image to numpy
            img_np = np.array(image)
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]  # drop alpha channel

            # Run the YOLO model                      
            results = model.predict(image, imgsz=640, device='cpu', verbose=False, conf=0.0, iou=1.0)
            yolo_preds = results[0]

            # Extract boxes, scores, class IDs
            boxes = yolo_preds.boxes.xyxy.cpu().clone()        # [N,4]
            scores = yolo_preds.boxes.conf.cpu().clone()       # [N]
            labels = yolo_preds.boxes.cls.cpu().long().clone() # [N]

            # Store context
            state = fresh_state()
            state["image_orig"] = image
            state["image_tr"] = transform_yolo(image)
            state["all_boxes"] = boxes
            state["all_scores"] = scores
            state["all_labels"] = labels            
            state["index"] = 0

        # Apply pruning
        if conf != state["conf"]:
            boxes, scores, labels, index = state["all_boxes"], state["all_scores"], state["all_labels"], state["index"]
            item_idx = prune_predictions(state["all_boxes"], state["all_scores"], conf)
            boxes, scores, labels = boxes[item_idx].long(), scores[item_idx], labels[item_idx]

            # Store the number of items found
            num_items = len(boxes)

            # Sort bboxes from left to right
            if num_items > 0:
                boxes_np = boxes.cpu().numpy()
                sorted_indices = np.lexsort((boxes_np[:,1], boxes_np[:,0]))  # sort by y1 then x1
                boxes = boxes[sorted_indices]
                scores = scores[sorted_indices]
                labels = labels[sorted_indices]
            
            # Update context                       
            state["index"] = 0
            state["n_items"] = num_items
            state["boxes"] = boxes
            state["scores"] = scores
            state["labels"] = labels

        # Get context        
        index = state["index"]
        boxes = state["boxes"]
        scores = state["scores"]
        labels = state["labels"]
        num_items = state["n_items"]

        # Convert normalized tensor [0,1] to 8-bit RGB image [0,255]
        image_tr = state["image_tr"].clone()[:3, ...] #.mul(255.0).to(torch.uint8).clone()[:3, ...]

        # Update overlay image if the parameters have changed
        if conf != state["conf"] or color != state["color"] or alpha != state["alpha"]:
            
            # Build label strings for all boxes (used for full-image drawing) 
            # label_strings = compose_labels(labels, scores)          

            # Dimm background
            image_out, _ = dimm_background(
                image=image_tr,
                boxes=boxes,
                alpha=alpha
                )
            
            # Draw bounding boxes
            if boxes is not None and len(boxes) > 0:
                image_out = draw_bounding_boxes(
                    image=image_out,
                    boxes=boxes,
                    #labels=label_compose,
                    colors=color_str,
                    width=thickness2,
                    fill=fill_bool
                    )
            else:
                # No boxes to draw
                image_out = image_out

            # Update context
            state["conf"] = conf
            state["color"] = color
            state["alpha"] = alpha
            state["image_out"] = image_out
        
        # Select which bounding box(es) to display
        if index > 0 and index <= num_items:
            box_i = boxes[index-1].unsqueeze(0)
            score_i = scores[index-1].unsqueeze(0)
            label_i = labels[index-1].unsqueeze(0)
        else:
            box_i = boxes
            score_i = scores
            label_i = labels

        # Build human-readable label strings for the selected box(es)
        #label_string_i = compose_labels(labels, scores)
        
        # Dim the whole image if there are no detected boxes
        if num_items == 0:
            image_out, _ = dimm_background(
                image=image_tr,
                boxes=None,
                alpha=alpha
            )

        else:

            # Create a version of the image where only the current box is cut out 
            image_i, mask = dimm_background(
                image=image_tr,
                boxes=box_i,
                alpha=0.0
                )
            
            if box_i is not None and len(box_i) > 0:
                image_i = draw_bounding_boxes(
                        image=image_i,
                        boxes=box_i,
                        #labels=label_string_i,
                        colors=color_str,
                        width=thickness1,
                        fill=True,
                    )

            # Blend the highlighted single box (image_i) with the full-image drawing (image_out)
            image_out = image_i + (state["image_out"] * (1 - mask))
        
        # Convert image to PIL
        image_boxes = to_pil(image_out.cpu())
        
        # Update context
        state["image_prev"] = state["image_orig"]

        # Compute confidence
        confidence = compute_confidence(score_i)

        return image_boxes, num_items, confidence, state

    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        return None, error_message, "", state
