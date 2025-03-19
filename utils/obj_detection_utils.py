"""
Provides utility functions for deep learning object detection workflows in PyTorch.  
Some functions are based on https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py"
"""

import datetime
import errno
import os
import time
import torch
import torch.distributed as dist
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List
from collections import defaultdict, deque
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

# Function to remove redundant boxes and masks
def prune_predictions(
    pred,
    score_threshold=0.66,
    mask_threshold=0.5,
    iou_threshold=0.5,
    ):

    """
    Filters and refines predictions by:
    1. Removing low-confidence detections based on the score threshold.
    2. Applying a binary mask threshold to filter out weak segmentation masks.
    3. Using Non-Maximum Suppression (NMS) to eliminate overlapping predictions.

    Parameters:
    pred : dict
        The raw predictions containing "boxes", "scores", "labels", and "masks".
    score_threshold : float, optional
        The minimum confidence score required to keep a prediction (default: 0.7).
    mask_threshold : float, optional
        The threshold for binarizing the segmentation masks (default: 0.5).
    iou_threshold : float, optional
        The Intersection over Union (IoU) threshold for NMS (default: 0.5).

    Returns:
    dict
        A dictionary with filtered and refined predictions:
        - "boxes": Tensor of kept bounding boxes.
        - "scores": Tensor of kept scores.
        - "labels": List of label strings.
        - "masks": Tensor of kept segmentation masks. [OPTIONAL]
    """
    
    # Filter predictions based on confidence score threshold
    scores = pred["scores"]
    high_conf_idx = scores > score_threshold

    filtered_pred = {
        "boxes":  pred["boxes"][high_conf_idx].long(),
        "scores": pred["scores"][high_conf_idx],
        "labels": pred["labels"][high_conf_idx], #[f"roi: {s:.3f}" for s in scores[high_conf_idx]]
    }

    # Only add "masks" if present in prediction output
    if "masks" in pred:
        filtered_pred["masks"] = (pred["masks"] > mask_threshold).squeeze(1)[high_conf_idx]

    # Apply Non-Maximum Suppression (NMS) to remove overlapping predictions
    if len(filtered_pred["boxes"]) == 0:
        return filtered_pred  # No boxes to process
    keep_idx = ops.nms(filtered_pred["boxes"].float(), filtered_pred["scores"], iou_threshold)

    # Return filtered predictions
    return {
        "boxes": filtered_pred["boxes"][keep_idx],
        "scores": filtered_pred["scores"][keep_idx],
        "labels": filtered_pred["labels"][keep_idx], #[i] for i in keep_idx],
        **({"masks": filtered_pred["masks"][keep_idx]} if "masks" in filtered_pred else {})  # Only add "masks" if present
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
    save_dir: str = None
    ):

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
    """

    plt.close("all")

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
        #labels  = [
        #    f"{label_to_class_dict[l.item()]}: {s:.3f}"
        #    for l, s in zip(filtered_pred["labels"], filtered_pred["scores"])
        #]

        # Draw bounding boxes
        output_image = draw_bounding_boxes(
            image=image,
            boxes=filtered_pred["boxes"],
            labels=labels if print_classes or print_scores else None,
            colors=box_color,
            width=width,
            font=font_type,
            font_size=font_size)
        
        # Draw segmentation masks (if any)
        if "masks" in filtered_pred and len(filtered_pred["masks"]) > 0:

            # Merge all masks into one by taking the maximum along the mask dimension
            masks = filtered_pred["masks"].sum(dim=0).clamp(0, 1).bool()

            # Draw masks
            output_image = draw_segmentation_masks(output_image, masks.unsqueeze(0), alpha=0.5, colors=mask_color)

        # Save Image (if save_dir is provided)
        if save_dir:
            image_pil = to_pil_image(output_image)  # Convert tensor to PIL Image
            image_path = os.path.join(save_dir, f"prediction_{idx+1}.png")
            image_pil.save(image_path)

        # Plot on the grid
        ax = axes[idx]
        ax.imshow(output_image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.set_title(f"Prediction {idx + 1}")
        ax.axis("off")

    # Hide unused subplots (if any)
    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_transformed_data(img, target, transformed_img, transformed_target):

    """
    Visualizes the original and transformed image along with bounding boxes and masks.
    
    Parameters:
    - img: Original image tensor.
    - target: Original target dictionary (contains boxes, masks, labels).
    - transformed_img: Transformed image tensor.
    - transformed_target: Transformed target dictionary.
    """
    
    # Visualize original image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original Image
    axes[0].imshow(img.permute(1, 2, 0))  # Convert CHW to HWC for plotting
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[0].add_patch(rect)
    axes[0].set_title('Original Image')

    # Transformed Image
    axes[1].imshow(transformed_img.permute(1, 2, 0))  # Convert CHW to HWC for plotting
    for box in transformed_target['boxes']:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        axes[1].add_patch(rect)
    axes[1].set_title('Transformed Image')

    plt.show()