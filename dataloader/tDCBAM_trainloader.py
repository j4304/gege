import os
import re
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# =============================================================================
# AUGMENTATION PARAMS SAMPLING
# =============================================================================

def sample_augment_params():
    """
    Sample a single set of geometric augmentation parameters.

    For independent augmentation, this should be called individually for the 
    Anchor, Positive, and Negative images. This forces the model to learn 
    spatial invariance and handle natural intra-class geometric variations.

    Bounds are kept tight to realistically simulate human handwriting variability:
        - angle: [-15, 15] degrees (natural slant variation)
        - scale: [0.85, 1.15] (natural size/pressure variation)
        - jitter_frac: [0.0, 1.0] (canvas placement variation)
    """
    return {
        'angle':         random.uniform(-15, 15),
        'scale':         random.uniform(0.85, 1.15),
        'jitter_frac_y': random.random(),
        'jitter_frac_x': random.random(),
    }


# =============================================================================
# PREPROCESSING & AUGMENTATION
# =============================================================================

def preprocess_image(img, img_size=(224, 224), augment=False, augment_params=None):
    """
    Preprocesses a signature image through the full pipeline.
    Background remains white (255) and strokes remain black (0) throughout.
    """
    if img is None:
        return None

    # --- 1. Ensure numpy RGB array ---
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = np.array(img)

    # --- 2. Grayscale conversion ---
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()

    # --- 3. Otsu binarization ---
    # Standard Otsu on a signature gives white background (255) and black strokes (0).
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- 4. Geometric Augmentation ---
    params = None
    if augment_params is not None:
        params = augment_params
    elif augment:
        params = sample_augment_params()

    if params is not None:
        h, w = img_binary.shape

        # HORIZONTAL FLIPPING REMOVED to preserve stroke directionality

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])

        # Apply transformation with borderValue=255 (White Padding)
        img_binary = cv2.warpAffine(
            img_binary, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255  
        )

    # --- 5. Tight crop with margin ---
    # cv2.findNonZero requires the target to be >0. We temporarily pass an inverted 
    # copy of the image JUST to find the coordinates of the black strokes.
    coords = cv2.findNonZero(cv2.bitwise_not(img_binary))
    
    if coords is not None:
        x, y, w_c, h_c = cv2.boundingRect(coords)
        margin = 10
        x_s = max(0, x - margin)
        y_s = max(0, y - margin)
        x_e = min(img_binary.shape[1], x + w_c + margin)
        y_e = min(img_binary.shape[0], y + h_c + margin)
        # Crop the original white-background image
        img_crop = img_binary[y_s:y_e, x_s:x_e]
    else:
        img_crop = img_binary

    # --- 6. Aspect-aware resize ---
    target_size = img_size[0]
    h_c, w_c = img_crop.shape
    scale = target_size / max(h_c, w_c)
    nw = int(w_c * scale)
    nh = int(h_c * scale)

    if nw == 0 or nh == 0:
        img_resized = cv2.resize(img_crop, img_size, interpolation=cv2.INTER_AREA)
        nw, nh = img_size[1], img_size[0]
    else:
        img_resized = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # --- 7. Canvas placement ---
    # Initialize a pure WHITE canvas (255) instead of black
    canvas = np.full(img_size, 255, dtype=np.uint8)
    
    y_slack = max(0, target_size - nh)
    x_slack = max(0, target_size - nw)
    if params is not None:
        y_off = int(params['jitter_frac_y'] * y_slack)
        x_off = int(params['jitter_frac_x'] * x_slack)
    else:
        y_off = y_slack // 2
        x_off = x_slack // 2
        
    canvas[y_off:y_off + nh, x_off:x_off + nw] = img_resized

    # --- 8. Grayscale to RGB + Noise ---
    # No inversion needed here anymore!
    if params is not None:
        # Add subtle Gaussian noise to break pure white/zero-padding artifacts
        noise = np.random.normal(loc=0, scale=5.0, size=canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # --- 9. Float conversion + ImageNet normalization ---
    img_float = img_rgb.astype("float32") / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    norm_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)

    return norm_tensor


# =============================================================================
# TRANSFORM FACTORY  (baseline / val / test — image-level)
# =============================================================================

def get_transforms(mode='train', input_shape=(224, 224), preprocess=True):
    """
    Returns the image-level transform pipeline for a given mode.

    This factory is used by:
        • baseline_cedar.py  — SplitDataset  (train / val / test)
        • proposed_cedar.py  — SplitEpisodicDataset (val / test)
        • proposed_cedar.py  — SplitTripletDataset val/test splits only

    It is NOT used for augmentation in SplitTripletDataset's training split.
    There, augmentation is applied at the triplet level inside __getitem__
    via sample_augment_params() + preprocess_image(augment_params=...).

    Args:
        mode (str): 'train' → augmentation ON (image-level, for baseline).
                    'val' / 'test' → preprocessing only, no augmentation.
        input_shape (tuple): Target image size. Default (224, 224).
        preprocess (bool): If True, runs the full signature binarization
                           pipeline. If False, uses standard resize + ToTensor
                           + Normalize for raw RGB inputs. Default True.

    Returns:
        torchvision.transforms.Compose
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError(f"mode must be 'train', 'val', or 'test'. Got: '{mode}'")

    augment = (mode == 'train')

    if preprocess:
        return transforms.Compose([
            transforms.Lambda(
                lambda img: preprocess_image(img, img_size=input_shape, augment=augment)
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# =============================================================================
# LEGACY: DIRECTORY-BASED DATASET (kept for backward compatibility)
# =============================================================================

class SignaturePretrainDataset(Dataset):
    """
    [LEGACY] Triplet dataset that reads from org_dir / forg_dir directories.

    This class is retained for backward compatibility with directory-based
    pipelines. The active training pipeline uses SplitTripletDataset (in
    proposed_cedar.py) which reads from split JSON user dictionaries instead.

    For new experiments, use SplitTripletDataset + sample_augment_params().
    """

    def __init__(self, org_dir, forg_dir, transform=None, user_list=None):
        self.transform = transform
        self.org_images = []
        self.forg_images = []

        valid_exts = ('.png', '.tif', '.jpg', '.jpeg')

        for root, _, files in os.walk(org_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.org_images.append(os.path.join(root, file))

        for root, _, files in os.walk(forg_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.forg_images.append(os.path.join(root, file))

        if user_list is not None:
            user_list = set(str(u) for u in user_list)
            self.org_images = [
                x for x in self.org_images
                if self._get_user_id(os.path.basename(x)) in user_list
            ]
            self.forg_images = [
                x for x in self.forg_images
                if self._get_user_id(os.path.basename(x)) in user_list
            ]

        self.user_genuine_map = {}
        for path in self.org_images:
            uid = self._get_user_id(os.path.basename(path))
            if uid not in self.user_genuine_map:
                self.user_genuine_map[uid] = []
            self.user_genuine_map[uid].append(path)

        self.users = list(self.user_genuine_map.keys())
        self.triplets = []
        self.on_epoch_end()

    def _get_user_id(self, filename):
        match = re.search(r'\d+', filename)
        if match:
            number = str(int(match.group(0)))
            if 'H-' in filename:
                return f"H-{number}"
            elif 'B-' in filename:
                return f"B-{number}"
            else:
                return number
        return "unknown"

    def on_epoch_end(self):
        self.triplets = []
        all_user_ids = list(self.user_genuine_map.keys())

        for anchor_path in self.org_images:
            anchor_uid = self._get_user_id(os.path.basename(anchor_path))
            positives = self.user_genuine_map.get(anchor_uid, [])

            if len(positives) < 2:
                continue

            possible_pos = [p for p in positives if p != anchor_path]
            if not possible_pos:
                continue
            positive_path = random.choice(possible_pos)

            current_forgeries = [
                f for f in self.forg_images
                if self._get_user_id(os.path.basename(f)) == anchor_uid
            ]
            is_hard_mining = (random.random() < 0.7) and (len(current_forgeries) > 0)

            if is_hard_mining:
                negative_path = random.choice(current_forgeries)
            else:
                other_uid = random.choice([u for u in all_user_ids if u != anchor_uid])
                negatives_from_other = self.user_genuine_map.get(other_uid, [])
                if not negatives_from_other:
                    continue
                negative_path = random.choice(negatives_from_other)

            self.triplets.append((anchor_path, positive_path, negative_path))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor_img)
            pos = self.transform(pos_img)
            neg = self.transform(neg_img)

        return anchor, pos, neg, torch.tensor([1], dtype=torch.float32)