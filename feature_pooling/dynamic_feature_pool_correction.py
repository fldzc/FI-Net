
import json
import os
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm
from insightface.app import FaceAnalysis
from PIL import Image
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
from collections import defaultdict

# --- Path Setup ---
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.getcwd())
from rg_fiqa.models.rg_fiqa import RGFIQA

# --- Global Model Initialization ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))

# --- Load RG-FIQA model ---
RGFIQA_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'rg_fiqa', 'checkpoints', 'tinyfqnet_best.pth')
rgfiqa_model = RGFIQA().to(DEVICE)
try:
    rgfiqa_model.load_state_dict(torch.load(RGFIQA_CHECKPOINT_PATH, map_location=DEVICE))
    rgfiqa_model.eval()
    print("RG-FIQA model loaded successfully.")
except FileNotFoundError:
    rgfiqa_model = None
tfq_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((112, 112)), transforms.ToTensor()])

# --- Quality Assessment ---
@torch.no_grad()
def rgfiqa_predict(image_bgr):
    if rgfiqa_model is None or image_bgr is None: return 0.0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = tfq_transform(rgb).unsqueeze(0).to(DEVICE)
    return float(rgfiqa_model(x).item())

# --- Gallery Creation ---
def create_gallery(gallery_path):
    print("Creating gallery from reference images...")
    gallery_features = defaultdict(list)
    image_files = glob.glob(os.path.join(gallery_path, '*.jpg'))
    for image_path in tqdm(image_files, desc="Processing Gallery"):
        student_id = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None: continue
        faces = app.get(image)
        if len(faces) == 1:
            embedding = faces[0].embedding
            gallery_features[student_id].append({
                "embedding": embedding / np.linalg.norm(embedding),
                "quality": rgfiqa_predict(image),
                "is_main": True # Mark initial features as main
            })
    print(f"Gallery created with {len(gallery_features)} unique students.")
    return gallery_features

# --- Main Processing Logic with Correction ---
def process_probes_with_correction(probes_path, initial_gallery_features, max_pool_size=10, consistency_threshold=0.3):
    print("Processing probes with Self-Consistency Correction and Learning from Failure...")
    dynamic_gallery = {sid: list(feats) for sid, feats in initial_gallery_features.items()}
    
    genuine_scores, imposter_scores = [], []
    correct_identifications = 0
    total_probes = 0
    
    person_folders = []
    student_ids = set(initial_gallery_features.keys())
    print(f"Scanning for student folders in {probes_path}...")
    for root, dirs, files in os.walk(probes_path):
        for dir_name in dirs:
            if dir_name in student_ids:
                person_path = os.path.join(root, dir_name)
                person_folders.append((person_path, dir_name))

    if not person_folders:
        print("Warning: No matching student folders found in the probes path.")
        return [], [], 0, 0, 0, {}

    for person_path, student_id in tqdm(person_folders, desc="Processing Probes"):
        image_files = sorted(glob.glob(os.path.join(person_path, '*.jpg')))
        if not image_files: continue

        for image_path in image_files:
            total_probes += 1
            image = cv2.imread(image_path)
            if image is None: continue
            faces = app.get(image)
            if len(faces) != 1: continue
            
            probe_feature = faces[0].embedding / np.linalg.norm(faces[0].embedding)
            probe_quality = rgfiqa_predict(image)

            # --- Self-Consistency Check ---
            # Calculate the average feature of the claimed identity's gallery
            student_gallery_feats = dynamic_gallery.get(student_id)
            if not student_gallery_feats:
                # If the student has no gallery, we can't check consistency.
                # We can either skip or trust this sample. Let's trust for now.
                pass 
            else:
                avg_student_feat = np.mean([f['embedding'] for f in student_gallery_feats], axis=0)
                avg_student_feat /= np.linalg.norm(avg_student_feat)
                consistency_score = np.dot(probe_feature, avg_student_feat)

                # If the probe is not consistent with the claimed identity, discard it.
                if consistency_score < consistency_threshold:
                    continue # This is the core of the correction mechanism

            # --- Matching Logic ---
            scores = {}
            for gid, gfeats in dynamic_gallery.items():
                if gfeats:
                    gallery_embeddings = np.array([f['embedding'] for f in gfeats])
                    avg_feat = np.mean(gallery_embeddings, axis=0)
                    scores[gid] = np.dot(probe_feature, avg_feat / np.linalg.norm(avg_feat))
                else: scores[gid] = 0.0

            if not scores: continue
            best_match_id = max(scores, key=scores.get)
            match_score = scores[best_match_id]
            
            # Record scores for metrics
            if student_id in scores: genuine_scores.append(scores[student_id])
            for gid, score in scores.items():
                if gid != student_id: imposter_scores.append(score)

            # --- Pool Update Logic: Learning from Failure (only for consistent probes) ---
            if best_match_id != student_id:
                new_feature = {
                    "embedding": probe_feature, 
                    "quality": probe_quality,
                    "is_main": False
                }
                student_pool = dynamic_gallery[student_id]
                if len(student_pool) < max_pool_size:
                    student_pool.append(new_feature)
                else:
                    non_main_features_with_indices = [
                        (i, f) for i, f in enumerate(student_pool) if not f.get('is_main', False)
                    ]
                    if non_main_features_with_indices:
                        index_to_replace, _ = min(non_main_features_with_indices, key=lambda item: item[1]['quality'])
                        student_pool[index_to_replace] = new_feature
            
            if best_match_id == student_id:
                correct_identifications += 1
    
    return genuine_scores, imposter_scores, correct_identifications, total_probes, dynamic_gallery

# --- Metrics and Reporting ---
def calculate_and_display_metrics(genuine_scores, imposter_scores, correct_ids, total_probes, initial_gallery, final_gallery):
    if total_probes == 0:
        print("No probes processed.")
        return
    accuracy = (correct_ids / total_probes) * 100
    print(f"\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct_ids}/{total_probes})")

    avg_similarity = np.mean(genuine_scores) if genuine_scores else 0
    print(f"Average Genuine Similarity: {avg_similarity:.4f}")

    # --- Calculate and Display Gallery Drift ---
    def _calculate_avg_embedding(gallery_features):
        if not gallery_features: return None
        all_embeddings = [feat['embedding'] for feat in gallery_features]
        avg_embedding = np.mean(all_embeddings, axis=0)
        return avg_embedding / np.linalg.norm(avg_embedding)

    drifts = []
    for sid in initial_gallery.keys():
        initial_avg_feat = _calculate_avg_embedding(initial_gallery[sid])
        final_avg_feat = _calculate_avg_embedding(final_gallery.get(sid, []))

        if initial_avg_feat is not None and final_avg_feat is not None:
            drift = np.linalg.norm(initial_avg_feat - final_avg_feat)
            drifts.append(drift)
    
    if drifts:
        avg_drift = np.mean(drifts)
        print(f"Average Gallery Drift: {avg_drift:.4f}")
    else:
        avg_drift = 0.0


    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_score = np.concatenate([genuine_scores, imposter_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]
    print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")

    print("\nTAR @ FAR:")
    tar_at_far = {}
    for far_val in [1e-3, 1e-2, 1e-1]:
        idx = np.where(fpr <= far_val)[0]
        tar = tpr[idx[-1]] if len(idx) > 0 else 0.0
        tar_at_far[f"{far_val:.0e}"] = tar
        print(f"  TAR @ FAR={far_val:.0e}: {tar:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Misidentification Correction')
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.show()

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": float(roc_auc),
        "acc": float(accuracy / 100.0),
        "avg_genuine_similarity": float(avg_similarity),
        "avg_gallery_drift": float(avg_drift),
        "eer": float(eer),
        "threshold": float(eer_threshold),
        "tar_at_far": tar_at_far
    }

def main(args):
    initial_gallery = create_gallery(args.gallery_path)
    if not initial_gallery:
        print("Gallery is empty. Exiting.")
        return
    
    g, i, c, t, final_gallery = process_probes_with_correction(
        args.probes_path, 
        initial_gallery,
        max_pool_size=args.max_pool_size,
        consistency_threshold=args.consistency_threshold
    )
    
    return calculate_and_display_metrics(g, i, c, t, initial_gallery, final_gallery)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with dynamic pool and self-consistency correction.")
    parser.add_argument('--probes_path', type=str, default='dataset/NB116_person_spatial')
    parser.add_argument('--gallery_path', type=str, default='dataset/images/faces_images')
    parser.add_argument('--max_pool_size', type=int, default=10, help="Maximum size of the feature pool for each person.")
    parser.add_argument('--consistency_threshold', type=float, default=0.3, help="Threshold for self-consistency check.")
    args = parser.parse_args()
    main(args)
