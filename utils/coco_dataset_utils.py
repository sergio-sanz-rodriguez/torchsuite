"""
Provides utility functions for processing the COCO dataset for object detection and segmentation.
Includes functions to convert COCO metadata into folder structures: `images` for the original 
images and `masks` for the corresponding segmentation masks.
"""

# Generic libraries
import os
import json
import shutil
import random
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw

# Torchvision libraries
from torchvision import datasets
from torchvision.transforms import v2 as T

# Import custom libraries
from .classification_utils import set_seeds

# Warnings
import warnings
os.environ['TORCH_USE_CUDA_DSA'] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")
warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript.converter")

# Create target model directory
MODEL_DIR = Path("outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Set seeds
set_seeds(42)


def COCO_2_PennFundanPed(coco_images_path, coco_annotations_path, output_images_dir, output_masks_dir, label=""):
    
    """
    Converts the COCO dataset to match the PennFudanPed dataset format.
    - Saves only pedestrian images.
    - Creates binary segmentation masks for pedestrians.
    - Ensures a 1:1 correspondence between images and masks.
    - Optionally adds a label to the filename.

    Parameters:
    - coco_images_path (str): Path to the COCO images directory.
    - coco_annotations_path (str): Path to the COCO annotation JSON file.
    - output_images_dir (str): Directory to save converted images.
    - output_masks_dir (str): Directory to save corresponding masks.
    - label (str): Optional label to append to filenames.
    """

    # Ensure output directories exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Load COCO dataset
    dataset = datasets.CocoDetection(root=coco_images_path, annFile=coco_annotations_path)

    # Load COCO category mapping
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Find pedestrian category ID
    pedestrian_category_id = next(k for k, v in category_mapping.items() if "person" in v.lower())

    image_count = 0  # Counter to ensure consistent filenames

    for idx in range(len(dataset)):
        img, annotations = dataset[idx]

        # Filter only pedestrian annotations
        pedestrian_annotations = [ann for ann in annotations if ann["category_id"] == pedestrian_category_id]

        if not pedestrian_annotations:
            continue  # Skip if no pedestrians are present

        # Define consistent filenames with the optional label
        base_filename = f"{image_count:06d}"
        filename = f"{base_filename}_{label}.png" if label else f"{base_filename}.png"

        # Convert image to PNG if it's not already in that format
        img = img.convert("RGBA")  # Ensure the image is in a format compatible with PNG
        img.save(os.path.join(output_images_dir, filename))

        # Create segmentation mask
        mask = Image.new("L", img.size, 0)  # Initialize blank mask
        draw = ImageDraw.Draw(mask)

        has_valid_segmentation = False

        for ann in pedestrian_annotations:
            if not ann["segmentation"]:
                continue  

            for seg in ann["segmentation"]:
                if isinstance(seg, list) and len(seg) >= 6:  
                    polygon = [tuple(map(int, seg[i:i + 2])) for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, outline=255, fill=255)
                    has_valid_segmentation = True

        # Save only if a valid mask exists
        if has_valid_segmentation:
            mask.save(os.path.join(output_masks_dir, filename))
            image_count += 1  # Increment only when a valid image-mask pair is saved
        else:
            os.remove(os.path.join(output_images_dir, filename))  # Remove image if mask is invalid

    print(f"Dataset conversion completed: {image_count} images and masks saved.")


def COCO_2_ImgMsk(coco_images_path, coco_annotations_path, output_images_dir, output_masks_dir, 
                  class_dictionary="all", label=""):
    """
    Converts the COCO dataset to a folder format with images and segmentation masks.
    Allows filtering specific categories via a dictionary, or 'all' to use all categories.

    Parameters:
    - coco_images_path (str): Path to the COCO images directory.
    - coco_annotations_path (str): Path to the COCO annotation JSON file.
    - output_images_dir (str): Directory to save converted images.
    - output_masks_dir (str): Directory to save corresponding masks.    
    - class_dictionary (dict or "all"): A dictionary mapping category IDs to class names, or "all" for all categories.
    - label (str): Optional label to append to filenames.
    """

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    dataset = datasets.CocoDetection(root=coco_images_path, annFile=coco_annotations_path)

    # Load COCO category mapping
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    # If "all" is selected, include all available categories
    if class_dictionary  == "all":
        class_dictionary = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    # Check if class_dictionary is actually a dictionary
    if not isinstance(class_dictionary, dict):
        raise ValueError("`class_dictionary` must be a dictionary or 'all'.")

    # Map the ids n class_dictionary with the original ids in the coco dataset
    selected_category_ids = {cat["id"]: cat["name"] for cat in coco_data["categories"] if cat["name"] in class_dictionary.values()}
    category_id_mapping = {
        coco_id: key
        for key, value in class_dictionary.items()
        for coco_id, coco_name in selected_category_ids.items()        
        if coco_name == value
    }

    print(f"Category ID Mapping: {category_id_mapping}")

    # Initialize counter for filenames
    image_count = 0  

    for idx in range(len(dataset)):
        img, annotations = dataset[idx]

        # Filter only selected categories
        valid_annotations = [ann for ann in annotations if ann["category_id"] in category_id_mapping]

        # Skip if no selected categories are present
        if not valid_annotations:
            continue  

        base_filename = f"{image_count:06d}"
        filename = f"{base_filename}_{label}.png" if label else f"{base_filename}.png"

        img = img.convert("RGBA")  
        img.save(os.path.join(output_images_dir, filename))

        # Create segmentation mask
        mask = Image.new("L", img.size, 0)  
        draw = ImageDraw.Draw(mask)

        has_valid_segmentation = False

        for ann in valid_annotations:
            if not ann["segmentation"]:
                continue
            
            # Get old category_id from the dataset
            category_id = ann["category_id"]
            
            # Check if category_id exists in the new mapping
            if category_id not in category_id_mapping:
                print(f"Warning: Category ID {category_id} not found in the new mapping. Skipping this annotation.")
                continue  # Skip if category_id is not in the new mapping
 
            # Get the new category ID using the mapping
            internal_key = category_id_mapping[category_id]

            for seg in ann["segmentation"]:
                if isinstance(seg, list) and len(seg) >= 6:  
                    polygon = [tuple(map(int, seg[i:i + 2])) for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, outline=internal_key, fill=internal_key)  # Use remapped ID for mask

                    has_valid_segmentation = True

        if has_valid_segmentation:
            mask.save(os.path.join(output_masks_dir, filename))
            image_count += 1  
        else:
            os.remove(os.path.join(output_images_dir, filename))  

    print(f"Dataset conversion completed: {image_count} images and masks saved.")

    return category_id_mapping # Return the mapping of COCO category IDs to internal category IDs




def select_and_copy_samples(input_images_dir, input_masks_dir, output_images_dir, output_masks_dir, num_samples, seed=42):
    
    """
    Randomly selects a specified number of image-mask pairs and copies them to new directories.
    
    Parameters:
    - input_images_dir (str): Directory containing original images.
    - input_masks_dir (str): Directory containing original masks.
    - output_images_dir (str): Directory to save sampled images.
    - output_masks_dir (str): Directory to save sampled masks.
    - num_samples (int): Number of image-mask pairs to select.
    - seed (int): Random seed for reproducibility.
    """

    # Ensure output directories exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Get list of image filenames
    images = sorted(os.listdir(input_images_dir))
    masks = sorted(os.listdir(input_masks_dir))

    # Ensure image-mask pairing consistency (without file extensions)
    images_set = {os.path.splitext(img)[0] for img in images}
    masks_set = {os.path.splitext(mask)[0] for mask in masks}

    # Ensure enough samples exist
    if len(images_set) < num_samples or len(masks_set) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but fewer images are available.")

    # Set random seed and select samples
    random.seed(seed)
    selected_files = random.sample(list(images_set), num_samples)  # Convert set to list

    # Copy selected images and masks to new directories
    for file in selected_files:
        img_path = os.path.join(input_images_dir, file + ".png")
        mask_path = os.path.join(input_masks_dir, file + ".png")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            shutil.copy(img_path, os.path.join(output_images_dir, file + ".png"))
            shutil.copy(mask_path, os.path.join(output_masks_dir, file + ".png"))

    print(f"Successfully copied {num_samples} image-mask pairs to new folders.")

def move_samples(input_images_dir, input_masks_dir, output_val_dir, output_test_dir, num_samples_to_move, seed=42):

    """
    Move a specified number of samples from the validation dataset to the testing dataset.

    - input_images_dir: Directory containing the images.
    - input_masks_dir: Directory containing the masks.
    - output_val_dir: Output directory for the validation dataset.
    - output_test_dir: Output directory for the testing dataset.
    - num_samples_to_move: The number of samples to move from validation to testing.
    - seed: Random seed for reproducibility.
    """

    # List image and mask files
    images = sorted(os.listdir(input_images_dir))
    masks = sorted(os.listdir(input_masks_dir))

    # Ensure they match in length
    assert len(images) == len(masks), "Number of images and masks do not match."

    # Shuffle the dataset
    random.seed(seed)
    combined = list(zip(images, masks))
    random.shuffle(combined)
    
    # Ensure that there are enough samples to move
    if num_samples_to_move > len(combined):
        raise ValueError(f"Requested {num_samples_to_move} samples, but only {len(combined)} are available.")

    # Split the dataset into validation and test
    val_samples = combined[:-num_samples_to_move]
    test_samples = combined[-num_samples_to_move:]

    # Create the output directories if they do not exist
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    # Move the images and masks to their respective directories
    for img_name, mask_name in val_samples:
        shutil.move(os.path.join(input_images_dir, img_name), os.path.join(output_val_dir, img_name))
        shutil.move(os.path.join(input_masks_dir, mask_name), os.path.join(output_val_dir, mask_name))

    for img_name, mask_name in test_samples:
        shutil.move(os.path.join(input_images_dir, img_name), os.path.join(output_test_dir, img_name))
        shutil.move(os.path.join(input_masks_dir, mask_name), os.path.join(output_test_dir, mask_name))

    print(f"Moved {num_samples_to_move} samples to the test dataset. {len(val_samples)} samples remain in the validation dataset.")


def copy_files(src, dst):
    
    """
    Copy files and directories from src to dst, adding new files to the existing folder.
    
    Parameters:
    - src: Path to the source directory containing files to copy.
    - dst: Path to the destination directory where files will be copied.
    """
    
    if not os.path.exists(dst):
        os.makedirs(dst)  # Make sure the destination folder exists
    
    # Get all files in the source folder
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        
        # If it's a file, copy it to the destination
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)
        
        # If it's a folder, recursively copy its contents
        elif os.path.isdir(src_item):
            copy_files(src_item, dst_item)

def split_dataset(src_images, src_masks, dst_train_images, dst_train_masks, dst_val_images, dst_val_masks, dst_test_images, dst_test_masks, train_pct, val_pct, test_pct, seed=42):
    
    """
    Split the dataset into training, validation, and test sets, and move the corresponding images and masks to their respective directories.
    It is required that the image and the associated mask have the same name.
    
    Parameters:
    - src_images: Path to the source images.
    - src_masks: Path to the source mask files.
    - dst_train_images: Destination folder for training images.
    - dst_train_masks: Destination folder for training masks.
    - dst_val_images: Destination folder for validation images.
    - dst_val_masks: Destination folder for validation masks.
    - dst_test_images: Destination folder for test images.
    - dst_test_masks: Destination folder for test masks.
    - train_pct: Percentage of the dataset to be used for training.
    - val_pct: Percentage of the dataset to be used for validation.
    - test_pct: Percentage of the dataset to be used for testing.
    - seed: Random seed for reproducibility.
    """

    # Ensure directories exist
    os.makedirs(dst_train_images, exist_ok=True)
    os.makedirs(dst_train_masks, exist_ok=True)
    os.makedirs(dst_val_images, exist_ok=True)
    os.makedirs(dst_val_masks, exist_ok=True)
    os.makedirs(dst_test_images, exist_ok=True)
    os.makedirs(dst_test_masks, exist_ok=True)
    
    # Get list of all image files
    image_files = [f for f in os.listdir(src_images) if os.path.isfile(os.path.join(src_images, f))]
    random.seed(seed)
    random.shuffle(image_files)  # Shuffle the files

    # Calculate the number of files for each set
    total_files = len(image_files)
    train_end = int(total_files * train_pct)
    val_end = train_end + int(total_files * val_pct)

    # Split into train, val, and test sets
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Move files while ensuring masks exist
    for file in train_files:        
        shutil.copy2(os.path.join(src_images, file), os.path.join(dst_train_images, file))
        shutil.copy2(os.path.join(src_masks, file), os.path.join(dst_train_masks, file))

    for file in val_files:        
        shutil.copy2(os.path.join(src_images, file), os.path.join(dst_val_images, file))
        shutil.copy2(os.path.join(src_masks, file), os.path.join(dst_val_masks, file))

    for file in test_files:
        shutil.copy2(os.path.join(src_images, file), os.path.join(dst_test_images, file))
        shutil.copy2(os.path.join(src_masks, file), os.path.join(dst_test_masks, file))

    print("Dataset split successfully.")

