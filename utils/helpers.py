import yaml
import os
import json
import kagglehub
import torch
import pynvml
from PIL import Image, ImageOps
import numpy as np

def load_config(config_path):
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None

def get_model_from_Kaggle(kaggle_handle):
    """
    Downloads a model from Kaggle Hub using the provided handle.
    Requires Kaggle API credentials to be set up (~/.kaggle/kaggle.json).

    Args:
        kaggle_handle (str): The Kaggle Hub model handle (e.g., 'google/yolov10/pyTorch/yolov10s').

    Returns:
        str: The local path where the model was downloaded, or None if download failed.
    """
    try:
        # Ensure Kaggle API credentials are set from ~/.kaggle/kaggle.json
        # The kagglehub library usually handles this automatically if the file exists.
        print(f"Attempting to download model from Kaggle Hub: {kaggle_handle}")
        model_path = kagglehub.model_download(handle=kaggle_handle)
        print(f"Model successfully downloaded to: {model_path}")
        return model_path
    except FileNotFoundError:
        print("Error: Kaggle API credentials (~/.kaggle/kaggle.json) not found.")
        print("Please ensure your Kaggle API token is correctly set up.")
        return None
    except Exception as e:
        print(f"An error occurred while downloading the model from Kaggle Hub: {e}")
        return None

class MemoryTracker:
    """
    Robust GPU memory tracker using pynvml.

    Usage:
        mt = MemoryTracker()               # use current cuda device if available
        mt = MemoryTracker(0)              # use cuda:0
        mt = MemoryTracker(torch.device('cuda:0'))
        mt = MemoryTracker(torch.device('cpu'))  # will be disabled safely
    """
    def __init__(self, device=None):
        self.handle = None
        self.initial_used_mem = 0
        self.nvml_initialized = False

        # quick check for CUDA availability
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. MemoryTracker will not function.")
            return

        # normalize device_index
        device_index = None
        try:
            if device is None:
                device_index = torch.cuda.current_device()
            elif isinstance(device, torch.device):
                if device.type != "cuda":
                    print(f"MemoryTracker: provided device {device} is not a CUDA device. Disabling memory tracking.")
                    return
                # Handle index=None by falling back to current_device
                device_index = device.index if device.index is not None else torch.cuda.current_device()
            else:
                # allow passing int-like device index or string convertible to int
                try:
                    device_index = int(device)
                except Exception:
                    print(f"MemoryTracker: could not interpret device={device!r} as an integer index. Disabling memory tracking.")
                    return

            # validate index range
            if device_index < 0 or device_index >= torch.cuda.device_count():
                print(f"MemoryTracker: device index {device_index} out of range (0..{torch.cuda.device_count()-1}). Disabling memory tracking.")
                return

            # init NVML and get handle
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.initial_used_mem = info.used
                print(f"MemoryTracker initialized for device cuda:{device_index}. Initial used memory: {self.initial_used_mem / (1024**2):.2f} MB")
            except pynvml.NVMLError as error:
                print(f"Failed to initialize NVML: {error}. Memory tracking disabled.")
                self.handle = None
                self.initial_used_mem = 0
            except Exception as e:
                print(f"An unexpected error occurred during MemoryTracker initialization: {e}. Memory tracking disabled.")
                self.handle = None
                self.initial_used_mem = 0

        except Exception as e:
            # Catch anything unexpected during normalization
            print(f"MemoryTracker initialization error: {e}. Memory tracking disabled.")
            self.handle = None
            self.initial_used_mem = 0

    def get_used_memory_bytes(self):
        if self.handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.used
            except pynvml.NVMLError as error:
                print(f"Failed to get memory info: {error}")
                return 0
        return 0

    def get_used_memory_mb(self):
        return self.get_used_memory_bytes() / (1024**2)

    def get_memory_usage_since_init_mb(self):
        current_used = self.get_used_memory_bytes()
        return (current_used - self.initial_used_mem) / (1024**2)

    def shutdown(self):
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self.nvml_initialized = False

    def __del__(self):
        # prefer explicit shutdown, but fallback on delete
        self.shutdown()

class ResizeWithPad:
    """
    Resizes an image to a target size while preserving its original aspect ratio 
    by padding the remaining area with a background color (white).
    This prevents geometric distortion of signature strokes.
    
    Attributes:
        target_size (tuple): Desired output size (width, height).
        fill (int): Pixel value for padding (default is 255 for white background).
    """
    def __init__(self, target_size=(220, 150), fill=255):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be resized.

        Returns:
            PIL.Image: Resized and padded image.
        """
        original_w, original_h = img.size
        target_w, target_h = self.target_size
        
        # Calculate scaling factor to fit within target dimensions
        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize the image with high-quality downsampling (BICUBIC)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Calculate padding dimensions to center the image
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # Apply padding
        return ImageOps.expand(img, padding, fill=self.fill)