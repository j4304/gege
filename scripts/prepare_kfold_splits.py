import json
import os
import random
import re
import argparse
from sklearn.model_selection import KFold
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

def get_user_id_from_filename(filename: str) -> Optional[int]:
    """
    Extracts the user ID from CEDAR-style filenames (e.g., 'original_1_1.png').

    Args:
        filename (str): The input filename.

    Returns:
        Optional[int]: The extracted user ID as an integer, or None if not found.
    """
    # Regex to find the user ID pattern '_<digits>_'
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        # print(f"Warning: Could not extract user ID from CEDAR filename: {filename}")
        return None

def create_cedar_kfold_splits(base_data_dir: str, output_dir: str, num_splits: int = 5, seed: int = 42):
    """
    Scans the CEDAR dataset directory, identifies users, and creates JSON files
    defining K-Fold splits for meta-learning.

    Args:
        base_data_dir (str): Path to the root 'signatures' directory of the CEDAR dataset
                             (containing 'full_org' and 'full_forg' subdirectories).
        output_dir (str): Directory where the K-Fold split JSON files will be saved.
        num_splits (int): The number of folds (K) for cross-validation. Defaults to 5.
        seed (int): Random seed for shuffling users to ensure reproducibility. Defaults to 42.
    """
    print("--- Starting CEDAR K-Fold Split Generation ---")
    org_dir = os.path.join(base_data_dir, 'full_org')
    forg_dir = os.path.join(base_data_dir, 'full_forg')

    # --- 1. Validate Input Directories ---
    if not os.path.isdir(org_dir):
        print(f"ERROR: Genuine signatures directory not found: {org_dir}")
        return
    if not os.path.isdir(forg_dir):
        print(f"ERROR: Forged signatures directory not found: {forg_dir}")
        return

    # --- 2. Collect Files per User ---
    user_files: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: {'genuine': [], 'forgery': []})
    all_user_ids: Set[int] = set()
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    print(f"Scanning genuine signatures in: {org_dir}")
    for filename in os.listdir(org_dir):
        if filename.lower().endswith(supported_extensions):
            user_id = get_user_id_from_filename(filename)
            if user_id is not None:
                all_user_ids.add(user_id)
                # Store relative path from base_data_dir for consistency if needed,
                # but absolute paths might be safer depending on dataloader.
                # Let's store relative paths from the base_data_dir/signatures level
                # Assuming base_data_dir is /path/to/signatures
                relative_path = os.path.join('full_org', filename)
                user_files[user_id]['genuine'].append(relative_path)


    print(f"Scanning forged signatures in: {forg_dir}")
    for filename in os.listdir(forg_dir):
         if filename.lower().endswith(supported_extensions):
            user_id = get_user_id_from_filename(filename)
            if user_id is not None:
                 # Ensure we only consider users who also have genuine samples
                 if user_id in all_user_ids:
                      relative_path = os.path.join('full_forg', filename)
                      user_files[user_id]['forgery'].append(relative_path)

    # Convert user IDs set to a sorted list for consistent splitting
    user_id_list = sorted(list(all_user_ids))
    if not user_id_list:
         print("ERROR: No valid user IDs found. Cannot create splits.")
         return

    print(f"Found {len(user_id_list)} unique users with genuine signatures.")

    # --- 3. Perform K-Fold Splitting ---
    print(f"Splitting users into {num_splits} folds...")
    # Use KFold for stratified splitting (though user distribution is likely uniform here)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 4. Generate and Save JSON Files per Fold ---
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(user_id_list)):
        train_user_ids: List[int] = [user_id_list[i] for i in train_indices]
        test_user_ids: List[int] = [user_id_list[i] for i in test_indices]

        # Create the split data structure for this fold
        split_data: Dict[str, Dict[str, Dict[str, List[str]]]] = {
            'meta-train': {
                str(uid): user_files[uid] for uid in train_user_ids if uid in user_files
            },
            'meta-test': {
                str(uid): user_files[uid] for uid in test_user_ids if uid in user_files
            }
        }

        # Define the output file path for this fold
        file_path = os.path.join(output_dir, f'cedar_meta_split_fold_{fold_idx}.json')

        # Save the split data to the JSON file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=4)
            print(f"  Fold {fold_idx + 1}: {len(train_user_ids)} train users, {len(test_user_ids)} test users. Saved to: {file_path}")
        except IOError as e:
            print(f"ERROR: Could not write split file for Fold {fold_idx + 1}: {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while saving Fold {fold_idx + 1}: {e}")


    print("--- CEDAR K-Fold Split Generation Complete ---")


if __name__ == '__main__':
    # Set up argument parser to run from command line
    parser = argparse.ArgumentParser(description="Generate K-Fold splits for CEDAR dataset meta-learning.")
    parser.add_argument('--base_data_dir', type=str, required=True,
                        help="Path to the CEDAR 'signatures' directory (containing 'full_org', 'full_forg').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the generated K-Fold split JSON files.")
    parser.add_argument('--num_splits', type=int, default=5,
                        help="Number of folds (K) for cross-validation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for shuffling users.")

    args = parser.parse_args()

    # Run the split creation function
    create_cedar_kfold_splits(
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        num_splits=args.num_splits,
        seed=args.seed
    )