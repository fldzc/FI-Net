import json
import os
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm
from insightface.app import FaceAnalysis
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse

# --- Path Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Global Model Initialization ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))


# --- Feature Fusion (Global Average Pooling) ---
def fuse_person_features(person_path):
    """
    Fuses features from all images in a person's folder using Global Average Pooling.
    """
    image_files = glob.glob(os.path.join(person_path, '*.jpg'))
    if not image_files: return None

    features = []
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None: continue
        
        faces = app.get(image)
        if len(faces) == 1:
            # Normalize each feature before adding to the list
            embedding = faces[0].embedding
            norm_embedding = embedding / np.linalg.norm(embedding)
            features.append(norm_embedding)

    if not features: return None
    
    features = np.array(features, dtype=np.float32)

    # Global Average Pooling
    if len(features) > 1:
        fused_feature = np.mean(features, axis=0)
    else:
        fused_feature = features[0]
        
    # Final normalization of the fused feature
    return fused_feature / np.linalg.norm(fused_feature)


# --- Gallery and Probe Processing ---
def create_gallery(gallery_path):
    """Creates a feature gallery from images in the given path."""
    print("Creating gallery from reference images...")
    gallery_features = {}
    image_files = glob.glob(os.path.join(gallery_path, '*.jpg'))
    for image_path in tqdm(image_files, desc="Processing Gallery"):
        student_id = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None: continue

        faces = app.get(image)
        if len(faces) == 1:
            embedding = faces[0].embedding
            gallery_features[student_id] = embedding / np.linalg.norm(embedding)
    print(f"Gallery created with {len(gallery_features)} unique students.")
    return gallery_features

def process_probes(probes_path, gallery_features):
    """Processes probe folders, matches them against the gallery, and collects scores."""
    print("Processing probe folders...")
    genuine_scores, imposter_scores = [], []
    correct_identifications = 0
    total_probes = 0
    
    person_folders = []
    for root, dirs, files in os.walk(probes_path):
        if any(f.endswith('.jpg') for f in files):
            student_id = os.path.basename(root)
            if student_id in gallery_features:
                 person_folders.append((root, student_id))

    for person_path, student_id in tqdm(person_folders, desc="Processing Probes"):
        total_probes += 1
        probe_feature = fuse_person_features(person_path)
        if probe_feature is None:
            print(f"Could not generate feature for probe: {student_id}")
            continue

        scores = {gid: np.dot(probe_feature, gfeat) for gid, gfeat in gallery_features.items()}
        
        best_match_id = max(scores, key=scores.get)
        if best_match_id == student_id:
            correct_identifications += 1
        
        for gid, score in scores.items():
            if gid == student_id:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)
    
    return genuine_scores, imposter_scores, correct_identifications, total_probes

# --- Metrics and Reporting ---
def calculate_and_display_metrics(genuine_scores, imposter_scores, correct_identifications, total_probes):
    """Calculates and prints performance metrics and plots the ROC curve."""
    if total_probes == 0:
        print("No probes were processed. Cannot calculate metrics.")
        return

    accuracy = (correct_identifications / total_probes) * 100
    print(f"\n--- Evaluation Metrics ---")
    print(f"ACC (Accuracy): {accuracy:.2f}% ({correct_identifications}/{total_probes})")

    avg_similarity = np.mean(genuine_scores) if genuine_scores else 0
    print(f"AS (Average Similarity): {avg_similarity:.4f}")

    if not genuine_scores or not imposter_scores:
        print("Not enough data to compute ROC curve.")
        return

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_score = np.concatenate([genuine_scores, imposter_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    print("\nTAR @ FAR:")
    for far_val in [1e-3, 1e-2, 1e-1]:
        tar = tpr[np.where(fpr <= far_val)[0][-1]] if np.any(fpr <= far_val) else 0.0
        print(f"  TAR @ FAR={far_val:.0e}: {tar:.4f}")
        
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (Global Average Pooling Fusion)')
    plt.legend(loc="lower right")
    plt.grid(True, which="both")
    
    # 定义输出目录和文件名
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 保存ROC曲线图
    save_path = os.path.join(output_dir, "roc_curve_gap.png")
    plt.savefig(save_path)
    print(f"\nROC curve saved to {save_path}")
    plt.close() # 关闭图像，避免显示

    # 保存ROC数据到JSON文件
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(roc_auc),
        'accuracy': float(accuracy),
        'average_similarity': float(avg_similarity),
        'correct_identifications': int(correct_identifications),
        'total_probes': int(total_probes)
    }
    json_save_path = os.path.join(output_dir, "gap_eval_results.json")
    with open(json_save_path, 'w') as f:
        json.dump(roc_data, f, indent=4)
    print(f"Evaluation results saved to {json_save_path}")

def main(args):
    """Main function to run the evaluation."""
    gallery_features = create_gallery(args.gallery_path)
    if not gallery_features:
        print("Gallery is empty. Exiting.")
        return
        
    genuine_scores, imposter_scores, correct_ids, total_probes = process_probes(args.probes_path, gallery_features)
    
    calculate_and_display_metrics(genuine_scores, imposter_scores, correct_ids, total_probes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face recognition performance with Global Average Pooling fusion.")
    parser.add_argument('--probes_path', type=str, default='dataset/NB116_person_spatial',
                        help='Path to the root directory of probe images.')
    parser.add_argument('--gallery_path', type=str, default='dataset/images/faces_images',
                        help='Path to the directory of gallery images.')
    args = parser.parse_args()
    
    main(args)
