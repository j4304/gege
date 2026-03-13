"""
Script: Generate Train-Val-Test Splits for Multiple Datasets and Split Ratios
==============================================================================
Generates user-level train/val/test splits for:
  - CEDAR (55 users)
  - BHSig-Bengali (100 users)
  - BHSig-Hindi (160 users)

At configurable ratios: 65:18:18, 70:15:15, 60:20:20

Output: JSON files compatible with both baseline and proposed (tDCBAM) pipelines.
Each JSON has structure:
{
    "train": { "user_id": {"genuine": [...], "forged": [...]} },
    "val":   { "user_id": {"genuine": [...], "forged": [...]} },
    "test":  { "user_id": {"genuine": [...], "forged": [...]} }
}

Usage:
    python scripts/prepare_split_ratios.py --data_root ./data --output_dir ./data/ratio_splits
"""

import os
import re
import json
import random
import argparse
from glob import glob
from collections import defaultdict


def parse_cedar(cedar_root):
    """
    Parses CEDAR dataset structure.
    Expected: cedar_root/full_org/original_<uid>_<sid>.png
              cedar_root/full_forg/forgeries_<uid>_<sid>.png
    
    Returns:
        dict: {user_id_str: {"genuine": [abs_paths], "forged": [abs_paths]}}
    """
    users = defaultdict(lambda: {"genuine": [], "forged": []})
    
    org_dir = os.path.join(cedar_root, "full_org")
    forg_dir = os.path.join(cedar_root, "full_forg")
    
    if not os.path.isdir(org_dir):
        print(f"  [CEDAR] WARNING: Genuine dir not found: {org_dir}")
        return {}
    if not os.path.isdir(forg_dir):
        print(f"  [CEDAR] WARNING: Forged dir not found: {forg_dir}")
        return {}
    
    # Parse genuine: original_<uid>_<sid>.png
    for fname in os.listdir(org_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue
        match = re.search(r'_(\d+)_', fname)
        if match:
            uid = str(int(match.group(1)))
            users[uid]["genuine"].append(os.path.abspath(os.path.join(org_dir, fname)))
    
    # Parse forged: forgeries_<uid>_<sid>.png
    for fname in os.listdir(forg_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue
        match = re.search(r'_(\d+)_', fname)
        if match:
            uid = str(int(match.group(1)))
            users[uid]["forged"].append(os.path.abspath(os.path.join(forg_dir, fname)))
    
    # Filter: keep only users with both genuine and forged
    valid = {uid: data for uid, data in users.items() 
             if len(data["genuine"]) > 0 and len(data["forged"]) > 0}
    
    print(f"  [CEDAR] Found {len(valid)} valid users")
    return dict(valid)


def parse_bhsig_subset(subset_root, prefix):
    """
    Parses a BHSig subset (Bengali or Hindi).
    Expected: subset_root/Genuine/<user_folders>/  and subset_root/Forged/<user_folders>/
    OR flat structure with filenames like B-S-001-G-01.tif / H-S-001-F-01.tif
    
    Args:
        subset_root: Path to BHSig100_Bengali or BHSig160_Hindi
        prefix: 'B' for Bengali, 'H' for Hindi
    
    Returns:
        dict: {user_id_str: {"genuine": [abs_paths], "forged": [abs_paths]}}
    """
    users = defaultdict(lambda: {"genuine": [], "forged": []})
    
    if not os.path.isdir(subset_root):
        print(f"  [BHSig-{prefix}] WARNING: Directory not found: {subset_root}")
        return {}
    
    # Pattern: B-S-001-G-01.tif or H-S-150-F-05.png
    pattern = re.compile(rf'({prefix})-S-(\d+)-([GF])-', re.IGNORECASE)
    
    # Also handle directory-based structure: Genuine/001/, Forged/001/
    valid_exts = ('.tif', '.png', '.jpg', '.jpeg', '.bmp')
    
    # Recursively scan
    for root, _, files in os.walk(subset_root):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue
            
            filepath = os.path.abspath(os.path.join(root, fname))
            lower_path = filepath.lower()
            
            # Try filename-based parsing first
            match = pattern.search(fname)
            if match:
                _, uid_str, gf = match.groups()
                uid = f"{prefix}-{int(uid_str)}"
                if gf.upper() == 'G':
                    users[uid]["genuine"].append(filepath)
                else:
                    users[uid]["forged"].append(filepath)
                continue
            
            # Fallback: directory-based parsing
            # Extract user ID from any digits in filename
            uid_match = re.search(r'(\d+)', fname)
            if uid_match:
                raw_uid = int(uid_match.group(1))
                uid = f"{prefix}-{raw_uid}"
                
                if 'genuine' in lower_path or '/g/' in lower_path or '\\g\\' in lower_path:
                    users[uid]["genuine"].append(filepath)
                elif 'forged' in lower_path or '/f/' in lower_path or '\\f\\' in lower_path:
                    users[uid]["forged"].append(filepath)
    
    # Filter valid users
    valid = {uid: data for uid, data in users.items()
             if len(data["genuine"]) > 0 and len(data["forged"]) > 0}
    
    print(f"  [BHSig-{prefix}] Found {len(valid)} valid users")
    return dict(valid)


def create_split(user_dict, train_ratio, val_ratio, seed=42):
    """
    Splits users into train/val/test sets at the specified ratios.
    
    Args:
        user_dict: {uid: {"genuine": [...], "forged": [...]}}
        train_ratio: float (e.g., 0.65 for 65%)
        val_ratio: float (e.g., 0.18 for 18%)
        seed: random seed
    
    Returns:
        dict: {"train": {...}, "val": {...}, "test": {...}}
    """
    user_ids = sorted(list(user_dict.keys()))
    
    random.seed(seed)
    random.shuffle(user_ids)
    
    total_users = len(user_ids)
    n_train = int(total_users * train_ratio)
    n_val = int(total_users * val_ratio)
    
    # Ensure at least 1 user in each split
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    
    # Adjust if splits exceed total
    if n_train + n_val >= total_users:
        n_train = max(1, total_users - 2)
        n_val = 1
    
    train_ids = user_ids[:n_train]
    val_ids = user_ids[n_train:n_train + n_val]
    test_ids = user_ids[n_train + n_val:]
    
    # Ensure test has at least 1 user
    if len(test_ids) == 0 and len(val_ids) > 1:
        test_ids = [val_ids.pop()]
    
    split = {
        "train": {uid: user_dict[uid] for uid in train_ids},
        "val": {uid: user_dict[uid] for uid in val_ids},
        "test": {uid: user_dict[uid] for uid in test_ids}
    }
    
    return split, len(train_ids), len(val_ids), len(test_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate train-val-test splits for signature datasets")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root data directory (contains cedardataset/, bhsig260-hindi-bengali/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save split JSON files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--ratios', type=str, default='65:18:18,70:15:15,60:20:20',
                        help='Comma-separated train:val:test percentages (default: 65:18:18,70:15:15,60:20:20)')
    args = parser.parse_args()
    
    # Parse ratios from train:val:test format
    ratio_triplets = []
    for ratio_str in args.ratios.split(','):
        parts = ratio_str.strip().split(':')
        if len(parts) != 3:
            print(f"ERROR: Invalid ratio format '{ratio_str}'. Expected train:val:test (e.g., 65:18:18)")
            return
        train_pct, val_pct, test_pct = map(int, parts)
        ratio_triplets.append((train_pct / 100.0, val_pct / 100.0, test_pct / 100.0))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING TRAIN-VAL-TEST SPLITS FOR ALL DATASETS")
    print("=" * 60)
    print(f"Data Root: {args.data_root}")
    print(f"Output: {args.output_dir}")
    print(f"Ratios: {[f'{int(r[0]*100)}:{int(r[1]*100)}:{int(r[2]*100)}' for r in ratio_triplets]}")
    print(f"Seed: {args.seed}")
    print()
    
    # ── 1. Parse all datasets ──────────────────────────────────
    datasets = {}
    
    # CEDAR
    cedar_paths = [
        os.path.join(args.data_root, 'cedardataset', 'signatures'),
        os.path.join(args.data_root, 'cedardataset'),
        os.path.join(args.data_root, 'cedar'),
    ]
    for p in cedar_paths:
        if os.path.isdir(p) and (os.path.isdir(os.path.join(p, 'full_org')) or 
                                   os.path.isdir(os.path.join(p, 'full_forg'))):
            datasets['cedar'] = parse_cedar(p)
            break
    
    if 'cedar' not in datasets:
        print("  [CEDAR] Dataset not found, skipping.")
    
    # BHSig-Bengali
    bhsig_root = os.path.join(args.data_root, 'bhsig260-hindi-bengali')
    if not os.path.isdir(bhsig_root):
        bhsig_root = args.data_root  # fallback
    
    bengali_paths = [
        os.path.join(bhsig_root, 'BHSig100_Bengali'),
        os.path.join(bhsig_root, 'Bengali'),
    ]
    for p in bengali_paths:
        if os.path.isdir(p):
            datasets['bhsig_bengali'] = parse_bhsig_subset(p, 'B')
            break
    
    if 'bhsig_bengali' not in datasets:
        print("  [BHSig-Bengali] Dataset not found, skipping.")
    
    # BHSig-Hindi  
    hindi_paths = [
        os.path.join(bhsig_root, 'BHSig160_Hindi'),
        os.path.join(bhsig_root, 'Hindi'),
    ]
    for p in hindi_paths:
        if os.path.isdir(p):
            datasets['bhsig_hindi'] = parse_bhsig_subset(p, 'H')
            break
    
    if 'bhsig_hindi' not in datasets:
        print("  [BHSig-Hindi] Dataset not found, skipping.")
    
    if not datasets:
        print("\nERROR: No datasets found. Check your --data_root path.")
        return
    
    # ── 2. Generate splits for each dataset × ratio ──────────
    print(f"\n{'='*60}")
    print("GENERATING SPLITS")
    print(f"{'='*60}")
    
    summary = []
    
    for ds_name, user_dict in datasets.items():
        for train_ratio, val_ratio, test_ratio in ratio_triplets:
            train_pct = int(train_ratio * 100)
            val_pct = int(val_ratio * 100)
            test_pct = int(test_ratio * 100)
            
            split, n_train, n_val, n_test = create_split(user_dict, train_ratio, val_ratio, seed=args.seed)
            
            # Count total images
            train_gen = sum(len(v["genuine"]) for v in split["train"].values())
            train_forg = sum(len(v["forged"]) for v in split["train"].values())
            val_gen = sum(len(v["genuine"]) for v in split["val"].values())
            val_forg = sum(len(v["forged"]) for v in split["val"].values())
            test_gen = sum(len(v["genuine"]) for v in split["test"].values())
            test_forg = sum(len(v["forged"]) for v in split["test"].values())
            
            fname = f"{ds_name}_split_{train_pct}_{val_pct}_{test_pct}.json"
            fpath = os.path.join(args.output_dir, fname)
            
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(split, f, indent=2)
            
            info = (f"  {ds_name} ({train_pct}:{val_pct}:{test_pct}) | "
                    f"Train: {n_train} users ({train_gen}G + {train_forg}F) | "
                    f"Val: {n_val} users ({val_gen}G + {val_forg}F) | "
                    f"Test: {n_test} users ({test_gen}G + {test_forg}F) | "
                    f"Saved: {fname}")
            print(info)
            summary.append({
                "dataset": ds_name,
                "split": f"{train_pct}:{val_pct}:{test_pct}",
                "train_users": n_train,
                "val_users": n_val,
                "test_users": n_test,
                "train_genuine": train_gen,
                "train_forged": train_forg,
                "val_genuine": val_gen,
                "val_forged": val_forg,
                "test_genuine": test_gen,
                "test_forged": test_forg,
                "file": fname
            })
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Generated {len(summary)} split files.")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
