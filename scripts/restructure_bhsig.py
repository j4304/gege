import os
import json
import random
import re
import argparse
from glob import glob
from sklearn.model_selection import KFold

"""
Script: BHSig Dataset Restructuring and Splitting
=================================================
Description:
    This script parses the raw BHSig260 dataset directory, validates user data integrity,
    and generates disjoint partitions for:
    1. Representation Learning (Pre-training): 150 background users.
    2. Metric Learning (Meta-training/Evaluation): 110 evaluation users (partitioned into 5 folds).

    This ensures strict separation between base classes used for feature extraction training
    and novel classes used for few-shot verification evaluation.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Restructure and Split BHSig Dataset")
    parser.add_argument('--base_dir', type=str, required=True, help='Root directory containing raw BHSig images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON manifests')
    parser.add_argument('--pretrain_users', type=int, default=150, help='Number of users reserved for background set')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[Info] Scanning data structure at: {args.base_dir}")
    
    # ---------------------------------------------------------
    # 1. ROBUST DATA PARSING (REGEX-BASED)
    # ---------------------------------------------------------
    users = {} 
    # Regex to capture User ID (Group 2) regardless of language prefix (Group 1)
    # Matches formats like: B-S-001-G-01.tif or H-S-100-F-05.png
    pattern = re.compile(r'([BH])-S-(\d+)-', re.IGNORECASE)
    
    # Recursively find all image files
    files = glob(os.path.join(args.base_dir, '**', '*'), recursive=True)
    valid_extensions = ('.tif', '.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in files if os.path.isfile(f) and f.lower().endswith(valid_extensions)]
    
    print(f"[Info] Found {len(image_files)} image files. Parsing metadata...")
    
    for filepath in image_files:
        filename = os.path.basename(filepath)
        match = pattern.search(filename)
        
        if match:
            lang, uid = match.groups()
            # Standardize User ID (e.g., 'B-1', 'H-150')
            user_id = f"{lang.upper()}-{int(uid)}"
            
            if user_id not in users:
                users[user_id] = {'genuine': [], 'forged': []}
            
            # Categorize as Genuine or Forged based on directory structure or filename
            lower_path = filepath.lower()
            if 'genuine' in lower_path or '-g-' in filename.lower():
                users[user_id]['genuine'].append(filepath)
            elif 'forged' in lower_path or '-f-' in filename.lower():
                users[user_id]['forged'].append(filepath)

    # ---------------------------------------------------------
    # 2. DATA INTEGRITY CHECK
    # ---------------------------------------------------------
    valid_users = []
    for uid, data in users.items():
        # Ensure user has at least one sample for both classes to be valid
        if len(data['genuine']) > 0 and len(data['forged']) > 0:
            valid_users.append(uid)
        else:
            # Log skipped users for debugging purposes
            # print(f"[Warning] Skipping User {uid}: Insufficient data (G:{len(data['genuine'])}, F:{len(data['forged'])})")
            pass
    
    valid_users.sort()
    total_users = len(valid_users)
    print(f"[Info] Successfully validated {total_users} users with complete data.")
    
    if total_users < args.pretrain_users:
        raise ValueError(f"[Error] Found only {total_users} valid users, but {args.pretrain_users} are required for pre-training.")

    # ---------------------------------------------------------
    # 3. DISJOINT PARTITIONING
    # ---------------------------------------------------------
    # Reproducible shuffling
    random.seed(42)
    random.shuffle(valid_users)
    
    # Split A: Background Set (for Pre-training)
    background_users = valid_users[:args.pretrain_users]
    
    # Split B: Evaluation Set (for Meta-training)
    eval_users = valid_users[args.pretrain_users:]
    
    print(f"[Info] Split Statistics:")
    print(f"   - Background Users (Pre-training): {len(background_users)}")
    print(f"   - Evaluation Users (Meta-learning): {len(eval_users)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save Pre-training Manifest (Compatible with pretraining.ipynb)
    bg_path = os.path.join(args.output_dir, 'bhsig_background_users.json')
    with open(bg_path, 'w') as f:
        # Save as a dictionary for compatibility with existing loaders
        bg_data_dict = {uid: users[uid] for uid in background_users}
        json.dump(bg_data_dict, f, indent=4)
    print(f"[Success] Saved pre-training split to: {bg_path}")
    
    # ---------------------------------------------------------
    # 4. K-FOLD STRATIFICATION FOR META-LEARNING
    # ---------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(eval_users)):
        fold_train_uids = [eval_users[i] for i in train_idx]
        fold_val_uids = [eval_users[i] for i in val_idx]
        
        split_data = {
            'meta-train': {uid: users[uid] for uid in fold_train_uids},
            'meta-test': {uid: users[uid] for uid in fold_val_uids}
        }
        
        out_path = os.path.join(args.output_dir, f'bhsig_meta_split_fold_{fold}.json')
        with open(out_path, 'w') as f:
            json.dump(split_data, f, indent=4)
            
        print(f"   > Generated Fold {fold}: {len(fold_train_uids)} Train / {len(fold_val_uids)} Val users.")

if __name__ == '__main__':
    main()