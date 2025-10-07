
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
print(f"Using device: {DEVICE}")

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))

# --- Load RG-FIQA model for quality scoring ---
RGFIQA_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'rg_fiqa', 'checkpoints', 'tinyfqnet_best.pth')
rgfiqa_model = RGFIQA().to(DEVICE)
try:
    rgfiqa_model.load_state_dict(torch.load(RGFIQA_CHECKPOINT_PATH, map_location=DEVICE))
    rgfiqa_model.eval()
    print("RG-FIQA model loaded successfully.")
except FileNotFoundError:
    print(f"Warning: RG-FIQA checkpoint not found at {RGFIQA_CHECKPOINT_PATH}. Quality scores will be 0.")
    rgfiqa_model = None

tfq_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# --- Quality and Pose Functions ---
@torch.no_grad()
def rgfiqa_predict(image_bgr):
    if rgfiqa_model is None or image_bgr is None: return 0.0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = tfq_transform(rgb).unsqueeze(0).to(DEVICE)
    score = rgfiqa_model(x).item()
    return float(score)

def get_pose_category(pose_angles):
    yaw, pitch, roll = pose_angles
    if abs(yaw) < 20: return 'frontal'
    elif abs(yaw) < 50: return 'semi_profile'
    else: return 'profile'

def pose_similarity_score(pose1, pose2):
    """Computes a score based on pose similarity. Higher score for similar poses."""
    yaw1, pitch1, roll1 = pose1
    yaw2, pitch2, roll2 = pose2
    yaw_diff = abs(yaw1 - yaw2)
    pitch_diff = abs(pitch1 - pitch2)
    # Give a bonus if poses are in the same category
    bonus = 1.2 if get_pose_category(pose1) == get_pose_category(pose2) else 1.0
    # Normalize difference to a 0-1 range (approx) and invert
    sim = 1 / (1 + (yaw_diff + pitch_diff) / 180.0) 
    return sim * bonus

# --- Gallery and Probe Processing ---
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
            face = faces[0]
            gallery_features[student_id].append({
                "embedding": face.embedding / np.linalg.norm(face.embedding),
                "quality": rgfiqa_predict(image),
                "pose": face.pose,
                "is_main": True  # Mark initial features as main
            })
    print(f"Gallery created with {len(gallery_features)} unique students.")
    return gallery_features

# --- Main Processing Logic: Pose-Aware with Quality-Based Replacement ---
def process_probes_with_pose_awareness(probes_path, initial_gallery_features, max_pool_size=10):
    print("Processing probes with pose awareness (Learning from Failure)...")
    dynamic_gallery = {sid: list(feats) for sid, feats in initial_gallery_features.items()}
    
    genuine_scores, imposter_scores = [], []
    correct_identifications = 0
    total_probes = 0
    raw_genuine_feat_sims = [] # Add a new list for raw feature similarities
    
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
        return [], [], [], 0, 0, {}

    for person_path, student_id in tqdm(person_folders, desc="Processing Probes"):
        image_files = sorted(glob.glob(os.path.join(person_path, '*.jpg')))
        if not image_files: continue

        for image_path in image_files:
            total_probes += 1
            image = cv2.imread(image_path)
            if image is None: continue

            faces = app.get(image)
            if len(faces) != 1: continue
            
            probe_face = faces[0]
            probe_feature = probe_face.embedding / np.linalg.norm(probe_face.embedding)
            probe_quality = rgfiqa_predict(image)
            probe_pose = probe_face.pose

            scores = {}
            # This dictionary will hold the raw feature similarity for logging/analysis
            raw_feat_sims = {}
            for gid, gfeats in dynamic_gallery.items():
                if not gfeats:
                    scores[gid] = 0.0
                    continue

                best_pose_gfeat = None
                max_pose_sim = -1
                for gfeat in gfeats:
                    pose_sim = pose_similarity_score(probe_pose, gfeat['pose'])
                    if pose_sim > max_pose_sim:
                        max_pose_sim = pose_sim
                        best_pose_gfeat = gfeat
                
                if best_pose_gfeat:
                    feat_sim = np.dot(probe_feature, best_pose_gfeat['embedding'])
                    scores[gid] = feat_sim
                    raw_feat_sims[gid] = feat_sim
                else:
                    scores[gid] = 0.0

            if not scores: continue
            best_match_id = max(scores, key=scores.get)
            match_score = scores[best_match_id]
            
            # Pool Update Logic: Learning from Failure
            if best_match_id != student_id:
                new_feature = {
                    "embedding": probe_feature, 
                    "quality": probe_quality, 
                    "pose": probe_pose,
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
            
            # Record scores for all pairs for ROC analysis
            for gid, score in scores.items():
                if gid == student_id:
                    genuine_scores.append(score)
                else:
                    imposter_scores.append(score)
                if student_id in raw_feat_sims:
                    raw_genuine_feat_sims.append(raw_feat_sims[student_id])
    
    return genuine_scores, imposter_scores, raw_genuine_feat_sims, correct_identifications, total_probes, dynamic_gallery

# --- Metrics and Reporting (Identical to V1) ---
def calculate_and_display_metrics(genuine_scores, imposter_scores, raw_genuine_sims, correct_identifications, total_probes, initial_gallery, final_gallery):
    if total_probes == 0:
        print("No probes were processed.")
        return
    accuracy = (correct_identifications / total_probes) * 100
    print(f"\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct_identifications}/{total_probes})")
    avg_similarity = np.mean(raw_genuine_sims) if raw_genuine_sims else 0
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

    if not genuine_scores or not imposter_scores:
        print("Not enough data for ROC.")
        return
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
    plt.title('ROC Curve - Pose-Aware Dynamic Pool')
    plt.legend(loc="lower right")
    plt.grid(True, which="both")
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
    
    g, i, r, c, t, final_gallery = process_probes_with_pose_awareness(args.probes_path, 
                                                                   initial_gallery,
                                                                   max_pool_size=args.max_pool_size)
    
    return calculate_and_display_metrics(g, i, r, c, t, initial_gallery, final_gallery)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with dynamic pool, pose awareness, and quality-based replacement.")
    parser.add_argument('--probes_path', type=str, default='dataset/NB116_person_spatial')
    parser.add_argument('--gallery_path', type=str, default='dataset/images/faces_images')
    parser.add_argument('--max_pool_size', type=int, default=10, help="Maximum size of the feature pool for each person.")
    args = parser.parse_args()
    main(args)
