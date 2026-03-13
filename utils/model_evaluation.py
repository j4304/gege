import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# Set aesthetic style for all plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})


def _zscore_normalize(logits, eps=1e-8):
    """Z-score normalize 1D numpy array of logits and return sigmoid(scores).

    Args:
        logits (np.ndarray): raw model outputs (unbounded).
        eps (float): small value to avoid division by zero.
    Returns:
        norm_scores (np.ndarray): scores after z-score and sigmoid -> in [0,1]
        mu, sigma (float, float): mean and std of raw logits used for reporting
    """
    mu = np.mean(logits)
    sigma = np.std(logits)
    norm_logits = (logits - mu) / (sigma + eps)
    norm_scores = 1.0 / (1.0 + np.exp(-norm_logits))
    return norm_scores, mu, sigma


def compute_metrics(y_true, y_scores, return_curve_data=True):
    """Computes common biometric verification metrics.

    Args:
        y_true (array-like): ground truth labels (1=genuine, 0=forged)
        y_scores (array-like): similarity scores (higher = more similar / genuine)
        return_curve_data (bool): whether to include fpr/tpr/thresholds in output

    Returns:
        dict with keys: eer, auc, accuracy, precision, recall, f1, threshold,
        and optionally fpr,tpr,thresholds,y_true,y_scores,y_pred
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)

    # EER: index where |FNR - FPR| is smallest
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    optimal_threshold = thresholds[eer_idx]

    # Prediction at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    result = {
        'eer': eer,
        'auc': roc_auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': optimal_threshold,
    }

    if return_curve_data:
        result.update({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'y_true': y_true,
            'y_scores': y_scores,
            'y_pred': y_pred,
            'fnr': fnr,
        })

    return result


def evaluate_and_plot(feature_extractor, metric_generator, dataloader, device, save_dir,
                      normalize_logits=False, save_normalized_versions=False):
    """Runs inference, computes metrics and saves a set of consistent plots.

    Args:
        feature_extractor: backbone network that maps images -> feature vector
        metric_generator: network that maps combined features -> raw logit scalar
        dataloader: yields batches with keys 'support_images', 'query_images', 'query_labels'
        device: torch device
        save_dir: directory to save plots
        normalize_logits: if True, apply Z-score normalization on raw logits before sigmoid
        save_normalized_versions: if True and normalize_logits True, also save *_norm images

    Returns:
        results (dict): metrics returned by compute_metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    feature_extractor.eval()
    metric_generator.eval()

    all_labels = []
    all_scores = []        # sigmoid outputs
    all_raw_logits = []    # raw logits (for optional normalization)

    print(" > [Evaluation] Starting inference on Test Set...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Data unpacking and shape handling: keep compatibility with previous implementation
            support_imgs = batch['support_images'].squeeze(1).to(device)  # [B, C, H, W] (K=1)
            query_imgs = batch['query_images'].to(device)                # [B, N_Q, C, H, W]
            labels = batch['query_labels'].to(device)                    # [B, N_Q]

            B, N_Q, C, H, W = query_imgs.shape
            query_imgs_flat = query_imgs.view(B * N_Q, C, H, W)
            labels_flat = labels.view(B * N_Q)
            support_imgs_flat = support_imgs.unsqueeze(1).expand(-1, N_Q, -1, -1, -1).reshape(B * N_Q, C, H, W)

            # Features
            s_feats = feature_extractor(support_imgs_flat)
            q_feats = feature_extractor(query_imgs_flat)

            combined_feats = torch.cat((s_feats, q_feats), dim=1)

            logits = metric_generator(combined_feats)   # [B*N_Q, 1]
            logits = logits.view(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))       # sigmoid on CPU numpy array

            all_raw_logits.extend(logits.flatten().tolist())
            all_scores.extend(probs.flatten().tolist())
            all_labels.extend(labels_flat.cpu().numpy().flatten().tolist())

    # Optionally normalize raw logits with Z-score logic
    if normalize_logits:
        y_scores, mu, sigma = _zscore_normalize(np.array(all_raw_logits))
        print(f" > [Statistics] Raw Logits Mean: {mu:.4f} | Std: {sigma:.4f} (Z-score applied)")
    else:
        y_scores = np.array(all_scores)

    y_true = np.array(all_labels)

    # Compute metrics
    results = compute_metrics(y_true, y_scores)

    # Print report
    print(f"\n{'='*10} FINAL EVALUATION REPORT {'='*10}")
    print(f"EER (Equal Error Rate) : {results['eer']:.2%}")
    print(f"AUC (Area Under Curve) : {results['auc']:.4f}")
    print(f"Optimal Threshold      : {results['threshold']:.4f}")
    print(f"Accuracy (at EER)      : {results['accuracy']:.2%}")
    print(f"Precision              : {results['precision']:.2%}")
    print(f"Recall                 : {results['recall']:.2%}")
    print(f"F1-Score               : {results['f1']:.2%}")
    print("="*40)

    # Plotting
    _plot_roc_curve(results, save_dir, suffix="" if not normalize_logits else "_norm")
    _plot_score_distribution(results, save_dir, suffix="" if not normalize_logits else "_norm")
    _plot_confusion_matrix(results, save_dir, suffix="" if not normalize_logits else "_norm")
    _plot_far_frr(results, save_dir, suffix="" if not normalize_logits else "_norm")
    _plot_det_curve(results, save_dir, suffix="" if not normalize_logits else "_norm")

    # If requested, also save versions using the raw (non-normalized) sigmoid scores
    if normalize_logits and save_normalized_versions is False:
        # nothing to do
        pass

    print(f" > All plots saved to: {save_dir}")

    # Attach some of the raw arrays for downstream inspection if needed
    results['_raw_logits'] = np.array(all_raw_logits)
    results['_sigmoid_scores'] = np.array(all_scores)

    return results


def _plot_roc_curve(results, save_dir, suffix=""):
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr'], results['tpr'], lw=2, label=f"ROC (AUC = {results['auc']:.4f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')

    # Mark EER point
    plt.scatter([results['eer']], [1 - results['eer']], color='red', s=80, zorder=5,
                label=f"EER = {results['eer']:.2%}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TAR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f'roc_curve{suffix}.png')
    plt.savefig(path, bbox_inches='tight')
    print(f" > Saved ROC Plot to: {path}")
    plt.close()


def _plot_score_distribution(results, save_dir, suffix=""):
    plt.figure(figsize=(10, 6))
    y_true = results['y_true']
    scores = results['y_scores']

    gen_scores = scores[y_true == 1]
    forg_scores = scores[y_true == 0]

    sns.histplot(gen_scores, label='Genuine', kde=True, stat='density', element='step', alpha=0.6)
    sns.histplot(forg_scores, label='Forged', kde=True, stat='density', element='step', alpha=0.6)

    plt.axvline(x=results['threshold'], linestyle='--', linewidth=2, label=f"Threshold={results['threshold']:.2f}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title("Similarity Score Distribution")
    plt.legend()

    path = os.path.join(save_dir, f'score_distribution{suffix}.png')
    plt.savefig(path, bbox_inches='tight')
    print(f" > Saved Distribution Plot to: {path}")
    plt.close()


def _plot_confusion_matrix(results, save_dir, suffix=""):
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Forged', 'Pred: Genuine'],
                yticklabels=['True: Forged', 'True: Genuine'])
    plt.title(f"Confusion Matrix (Th={results['threshold']:.2f})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    path = os.path.join(save_dir, f'confusion_matrix{suffix}.png')
    plt.savefig(path, bbox_inches='tight')
    print(f" > Saved Confusion Matrix to: {path}")
    plt.close()


def _plot_far_frr(results, save_dir, suffix=""):
    # Plot FAR (FPR) and FRR (FNR) as functions of thresholds
    fpr = results['fpr']
    fnr = results['fnr']
    thresholds = results['thresholds']

    # roc_curve returns thresholds in decreasing order; sort ascending for nicer plots
    idx = np.argsort(thresholds)
    thr_sorted = thresholds[idx]
    fpr_sorted = fpr[idx]
    fnr_sorted = fnr[idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thr_sorted, fpr_sorted, label='FAR', linestyle='-')
    plt.plot(thr_sorted, fnr_sorted, label='FRR', linestyle='-')
    # Mark EER threshold
    plt.plot(results['threshold'], results['eer'], 'ko', label='EER Point')
    plt.axvline(x=results['threshold'], color='k', linestyle='--', alpha=0.4)
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR & FRR vs Threshold")
    plt.xlim([0.0, 1.0])
    plt.legend()

    path = os.path.join(save_dir, f'far_frr{suffix}.png')
    plt.savefig(path, bbox_inches='tight')
    print(f" > Saved FAR/FRR Plot to: {path}")
    plt.close()


def _plot_det_curve(results, save_dir, suffix=""):
    fpr = np.clip(results['fpr'], 1e-6, 1.0)
    fnr = np.clip(results['fnr'], 1e-6, 1.0)

    # DET-like plot on log-log scale (useful for biometric systems)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, fnr, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("False Reject Rate (FRR)")
    plt.title("DET Curve (log-log)")
    plt.grid(True, which='both', ls='--', alpha=0.2)

    path = os.path.join(save_dir, f'det_curve{suffix}.png')
    plt.savefig(path, bbox_inches='tight')
    print(f" > Saved DET Curve to: {path}")
    plt.close()


def visualize_hard_examples(feature_extractor, metric_generator, dataloader, device, save_dir, top_k=5):
    """
    Finds and saves the most confusing examples (False Positives & False Negatives).
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_extractor.eval()
    metric_generator.eval()
    
    hard_positives = [] # Should be Same (1), but Score is Low (False Negative)
    hard_negatives = [] # Should be Diff (0), but Score is High (False Positive)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mining Hard Examples", leave=False):
            support_imgs = batch['support_images'].squeeze(1).to(device)
            query_imgs = batch['query_images'].to(device)
            labels = batch['query_labels'].to(device)
            user_ids = batch['user_id'] # Assuming dataloader returns user_ids
            
            B, N_Q, C, H, W = query_imgs.shape
            
            # Prepare Logic
            query_flat = query_imgs.view(B * N_Q, C, H, W)
            labels_flat = labels.view(B * N_Q)
            support_flat = support_imgs.unsqueeze(1).expand(-1, N_Q, -1, -1, -1).reshape(B * N_Q, C, H, W)
            
            # Forward
            s_feats = feature_extractor(support_flat)
            q_feats = feature_extractor(query_flat)
            combined = torch.cat((s_feats, q_feats), dim=1)
            scores = torch.sigmoid(metric_generator(combined)).squeeze(1) # [B*N_Q]
            
            # --- Mining Logic ---
            for i in range(len(scores)):
                score = scores[i].item()
                label = labels_flat[i].item()
                
                # Get User ID for logging (Logic to map back from flattened index)
                # flattened_idx i -> batch_idx = i // N_Q
                batch_idx = i // N_Q
                uid = user_ids[batch_idx]
                
                # Image Tensors (CPU for saving)
                s_img = support_flat[i].cpu()
                q_img = query_flat[i].cpu()
                
                info = (score, label, uid, s_img, q_img)
                
                # False Negative: Label=1 (Genuine), Score Low
                if label == 1:
                    hard_positives.append(info)
                    
                # False Positive: Label=0 (Forged), Score High
                if label == 0:
                    hard_negatives.append(info)
    
    # Sort and Save Top-K
    # Hard Positives: Sort by Score Ascending (Lowest score is worst error)
    hard_positives.sort(key=lambda x: x[0])
    _save_example_images(hard_positives[:top_k], "FalseNegative", save_dir)
    
    # Hard Negatives: Sort by Score Descending (Highest score is worst error)
    hard_negatives.sort(key=lambda x: x[0], reverse=True)
    _save_example_images(hard_negatives[:top_k], "FalsePositive", save_dir)

def _save_example_images(example_list, prefix, save_dir):
    # Helper to save tensor images
    inv_normalize = lambda t: t * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    
    for idx, (score, label, uid, s_img, q_img) in enumerate(example_list):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        
        # Un-normalize for display
        s_disp = inv_normalize(s_img).permute(1, 2, 0).numpy()
        q_disp = inv_normalize(q_img).permute(1, 2, 0).numpy()
        s_disp = np.clip(s_disp, 0, 1)
        q_disp = np.clip(q_disp, 0, 1)
        
        ax[0].imshow(s_disp)
        ax[0].set_title(f"Support (Ref)\nUser: {uid}")
        ax[0].axis('off')
        
        ax[1].imshow(q_disp)
        type_str = "Genuine" if label==1 else "Forged"
        ax[1].set_title(f"Query ({type_str})\nModel Score: {score:.4f}")
        ax[1].axis('off')
        
        plt.suptitle(f"Error Type: {prefix} (Confidence: {score:.2%})")
        plt.savefig(os.path.join(save_dir, f"{prefix}_{idx+1}_uid{uid}.png"))
        plt.close()
    print(f" > Saved {len(example_list)} hard examples for {prefix}.")