import os, sys

# Constants and default parameters
MAX_DIM = 1920
MAX_ALPHA = 1.0
MIN_ALPHA = 0.0
DEFAULT_STEP = 0.2
DEFAULT_ALPHA = 0.6
IOU_THRESHOLD = 0.70
DEFAULT_DIMM = 1 - DEFAULT_ALPHA
DEFAULT_DOWNSCALE = False
LABEL_TO_CLASS_DICT={0: 'human'}
PRINT_CLASSES = False
PRINT_SCORES = False
CONF_LOW = 0.40
CONF_MED = 0.10
CONF_HIGH = 0.025
CONF_CHOICES = [
    ("Low",    CONF_LOW),
    ("Medium", CONF_MED),
    ("High",   CONF_HIGH)
    ]
DEFAULT_CONFIDENCE = CONF_MED
COLOR_CHOICES = [
    ("Black",         "black|False"),
    ("White",         "white|False"),                            
    ("Crimson Red",   "crimson|False"),
    ("Orange",        "orange|False"),
    ("Wheat",         "wheat|False"),
    ("Gold",          "gold|False"),
    ("Forest Green",  "forestgreen|False"),                            
    ("Spring Green",  "mediumspringgreen|False"),
    ("Cyan",          "cyan|False"),                                
    ("Deep Sky Blue", "deepskyblue|False"),
    ("Royal Blue",    "royalblue|False"),    
    ("Blue Violet",   "blueviolet|False"),
    ]
DEFAULT_BOX_COLOR = "orange|False"
APP_THEME_COLOR = "violet"

# Model path
def resource_path(rel_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

MODEL_WEIGHTS = "yolo_10b_50_4_best.pt"
MODEL_PATH = resource_path(MODEL_WEIGHTS)

# App text
TITLE = "DeepCount<br>📷🚶🏽‍♀️🏃‍♀️📷"
AUTHOR = "by [Sergio Sanz](https://www.linkedin.com/in/sergio-sanz-rodriguez/)"
DESCRIPTION = "A cutting-edge AI model to detect and count humans in images."
#WARNING_MESSAGE = "⚠️ Processing times may be longer than usual due to high server demand. If that's the case, consider trying again later for faster response times. ⚠️"
#WARNING_MESSAGE = "⚠️ Large image files may take longer to upload and could slow down processing. For faster performance, consider selecting the 'Downscale Image' checkbox to reduce the image size. ⚠️"  
WARNING_MESSAGE = "⚠️ Large image files may take longer to upload and could slow down processing. ⚠️"  

TECH_COPY_TEXT = """
This app is provided for research and demo purposes and is not intended for commercial use.  
It uses the [YOLOv10 - balanced](https://docs.ultralytics.com/models/yolov10/#key-features) object detection model and was trained on the [Roboflow Universe](https://universe.roboflow.com/leo-ueno/people-detection-o4rdr) dataset, which is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en). No changes were made to the dataset.
"""

# Example images
examples_path = os.path.join(os.path.dirname(__file__), "examples")
EXAMPLES = [
    ["examples/" + e]
    for e in os.listdir(examples_path)
    if e.lower().endswith(('.png', '.jpg', '.jpeg'))
]