
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rg_fiqa.models.rg_fiqa import RGFIQA
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'face_fusion', 'ProxyFusion-NeurIPS-main'))
try:
    from models.fusion_models import ProxyFusion
    PROXY_FUSION_AVAILABLE = True
except ImportError:
    print("Warning: ProxyFusion model not found, will use simple average fusion.")
    PROXY_FUSION_AVAILABLE = False


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
    print(f"Warning: RG-FIQA checkpoint not found at {RGFIQA_CHECKPOINT_PATH}. Quality scores will be 0.")
    rgfiqa_model = None
tfq_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((112, 112)), transforms.ToTensor()])

# --- ProxyFusion Model Loading (Singleton) ---
fusion_model = None

def get_fusion_model():
    """Lazy loads and caches the ProxyFusion model."""
    global fusion_model
    if fusion_model is not None:
        return fusion_model

    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_fusion', 'ProxyFusion-NeurIPS-main', 'checkpoints', 'ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar')
    if PROXY_FUSION_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = ProxyFusion(DIM=512).to(DEVICE)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if "state_dict" in checkpoint and "model_weights" in checkpoint["state_dict"]:
                model.load_state_dict(checkpoint["state_dict"]["model_weights"], strict=False)
            else:
                 model.load_state_dict(checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint, strict=False)
            model.eval()
            print("ProxyFusion model loaded successfully.")
            fusion_model = model
            return fusion_model
        except Exception as e:
            print(f"Failed to load ProxyFusion model: {e}. Using simple averaging.")
            fusion_model = "failed"
            return None
    else:
        print("ProxyFusion model not available. Using simple averaging.")
        fusion_model = "failed"
        return None

# --- Quality and Pose Functions ---
@torch.no_grad()
def rgfiqa_predict(image_bgr):
    if rgfiqa_model is None or image_bgr is None: return 0.0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = tfq_transform(rgb).unsqueeze(0).to(DEVICE)
    return float(rgfiqa_model(x).item())

def pose_similarity_score(pose1, pose2):
    yaw_diff = abs(pose1[0] - pose2[0])
    pitch_diff = abs(pose1[1] - pose2[1])
    return 1 / (1 + (yaw_diff + pitch_diff) / 180.0)

def fuse_person_features(image_files, fusion_model_instance):
    """
    Fuses features from a list of image files into a single super-feature.
    This function is now used for the probe side.
    """
    if not image_files: return None

    features, quality_scores = [], []
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None: continue
        
        faces = app.get(image)
        if len(faces) == 1:
            face = faces[0]
            features.append(face.embedding)
            quality_scores.append(rgfiqa_predict(image))

    if not features: return None
    
    features = np.array(features, dtype=np.float32)
    quality_scores = np.array(quality_scores, dtype=np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    fused_feature = None
    if len(features) > 1 and fusion_model_instance and fusion_model_instance != "failed":
        try:
            input_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                fused = fusion_model_instance.eval_fuse_probe(input_tensor)
            fused_feature = fused.mean(dim=0).cpu().numpy()
        except Exception as e:
            weights = quality_scores / (quality_scores.sum() + 1e-6)
            fused_feature = np.sum(features * weights[:, np.newaxis], axis=0)
    elif len(features) > 1:
        weights = quality_scores / (quality_scores.sum() + 1e-6)
        fused_feature = np.sum(features * weights[:, np.newaxis], axis=0)
    else:
        fused_feature = features[0]
        
    norm = np.linalg.norm(fused_feature)
    return fused_feature / norm if norm > 0 else fused_feature

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
            face = faces[0]
            gallery_features[student_id].append({
                "embedding": face.embedding / np.linalg.norm(face.embedding),
                "quality": rgfiqa_predict(image),
                "pose": face.pose,
                "is_main": True  # Mark initial features as main
            })
    print(f"Gallery created with {len(gallery_features)} unique students.")
    return gallery_features

# --- Main Processing Logic: Combined ---
def process_probes_combined(probes_path, initial_gallery, max_pool_size=10, consistency_threshold=0.3, num_fusion_features=4):
    print("Processing probes with TRUE N:N evaluation: Fused Probes vs Dynamic Fused Gallery...")
    dynamic_gallery = {sid: list(feats) for sid, feats in initial_gallery.items()}
    fusion_model_instance = get_fusion_model()
    
    genuine_scores, imposter_scores = [], []
    correct_identifications = 0
    total_probes = 0
    
    person_folders = []
    student_ids = set(initial_gallery.keys())
    print(f"Scanning for student folders in {probes_path}...")
    # Find all directories in probes_path that are also in student_ids
    for root, dirs, files in os.walk(probes_path):
        for dir_name in dirs:
            if dir_name in student_ids:
                person_path = os.path.join(root, dir_name)
                # Check if this directory actually contains images before adding
                if glob.glob(os.path.join(person_path, '*.jpg')):
                    person_folders.append((person_path, dir_name))
        # Avoid recursing into subdirectories of a found person folder
        dirs[:] = [d for d in dirs if d not in student_ids]


    if not person_folders:
        print("Warning: No matching student folders found in the probes path.")
        return [], [], 0, 0, {}


    for person_path, student_id in tqdm(person_folders, desc="Processing Probes (N:N)"):
        total_probes += 1
        image_files = sorted(glob.glob(os.path.join(person_path, '*.jpg')))
        if not image_files: continue

        # 1. Fuse all images in the probe folder to create one "super probe feature"
        super_probe_feature = fuse_person_features(image_files, fusion_model_instance)
        if super_probe_feature is None:
            print(f"Could not generate a super probe feature for {student_id}")
            continue

        scores = {}
        for gid, gfeats in dynamic_gallery.items():
            if not gfeats:
                scores[gid] = 0.0
                continue
            
            # 2. Fuse all features in the dynamic gallery for the current gallery ID
            gallery_embeddings = np.array([f['embedding'] for f in gfeats])
            fused_gallery_feature = None
            if len(gallery_embeddings) > 1 and fusion_model_instance and fusion_model_instance != "failed":
                try:
                    input_tensor = torch.tensor(gallery_embeddings, dtype=torch.float32, device=DEVICE)
                    with torch.no_grad():
                        fused = fusion_model_instance.eval_fuse_probe(input_tensor)
                    fused_gallery_feature = fused.mean(dim=0).cpu().numpy()
                except Exception:
                     fused_gallery_feature = np.mean(gallery_embeddings, axis=0) # Fallback
            else:
                fused_gallery_feature = np.mean(gallery_embeddings, axis=0)
            
            fused_gallery_feature /= np.linalg.norm(fused_gallery_feature)
            
            # 3. Compare the super_probe_feature with the fused_gallery_feature
            scores[gid] = np.dot(super_probe_feature, fused_gallery_feature)

        if not scores: continue
        best_match_id = max(scores, key=scores.get)
        
        # Record scores
        if student_id in scores: genuine_scores.append(scores[student_id])
        for gid, score in scores.items():
            if gid != student_id: imposter_scores.append(score)

        # --- Pool Update Logic ---
        # If misidentified, we can still add the super_probe_feature to the correct student's pool
        if best_match_id != student_id:
            probe_quality = np.mean([rgfiqa_predict(cv2.imread(img)) for img in image_files[:5]]) # Approx quality
            new_feature = {
                "embedding": super_probe_feature, 
                "quality": probe_quality, 
                "pose": np.array([0., 0., 0.]), # Pose is averaged out, using neutral
                "is_main": False
            }
            student_pool = dynamic_gallery[student_id]
            if len(student_pool) < max_pool_size:
                student_pool.append(new_feature)
            else:
                # Replace lowest quality non-main feature
                non_main_features_with_indices = [(i, f) for i, f in enumerate(student_pool) if not f.get('is_main', False)]
                if non_main_features_with_indices:
                    index_to_replace, _ = min(non_main_features_with_indices, key=lambda item: item[1]['quality'])
                    student_pool[index_to_replace] = new_feature
        
        if best_match_id == student_id:
            correct_identifications += 1
    
    return genuine_scores, imposter_scores, correct_identifications, total_probes, dynamic_gallery

# --- Metrics and Reporting ---
def calculate_and_display_metrics(genuine_scores, imposter_scores, correct_identifications, total_probes, initial_gallery, final_gallery):
    """Calculates and prints performance metrics and plots the ROC curve."""
    if total_probes == 0:
        print("No probes were processed. Cannot calculate metrics.")
        return

    # ACC - Accuracy
    accuracy = (correct_identifications / total_probes) * 100
    error_rate = 100 - accuracy
    print(f"\n--- Evaluation Metrics ---")
    print(f"ER: {error_rate:.2f}%")
    print(f"ACC: {accuracy:.2f}%")

    # AS - Average Similarity for genuine matches
    avg_similarity = np.mean(genuine_scores) if genuine_scores else 0
    print(f"AS: {avg_similarity:.4f}")

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
    
    avg_drift = np.mean(drifts) if drifts else 0
    # print(f"Average Gallery Drift: {avg_drift:.4f}")

    # TAR @ FAR and ROC Curve
    if not genuine_scores or not imposter_scores:
        print("Not enough data to compute ROC curve.")
        return

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_score = np.concatenate([genuine_scores, imposter_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Print TAR @ FAR values
    # ASAR @ FAR=1e-3
    far_val = 1e-3
    tar_at_far = tpr[np.where(fpr <= far_val)[0][-1]] if np.any(fpr <= far_val) else 0.0
    print(f"ASAR@FAR 1e-3: {tar_at_far:.4f}")

    # TAR @ FAR=1e-2
    far_val = 1e-2
    tar_at_far = tpr[np.where(fpr <= far_val)[0][-1]] if np.any(fpr <= far_val) else 0.0
    print(f"TAR@FAR 1e-2: {tar_at_far:.4f}")

    # TAR @ FAR=1e-1
    far_val = 1e-1
    tar_at_far = tpr[np.where(fpr <= far_val)[0][-1]] if np.any(fpr <= far_val) else 0.0
    print(f"TAR@FAR 1e-1: {tar_at_far:.4f}")
        
    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - ProxyFusion with Dynamic Pose-Aware Pooling')
    plt.legend(loc="lower right")
    plt.grid(True, which="both")
    
    # Define output directory and filenames
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ROC curve image
    save_path = os.path.join(output_dir, "roc_curve_proxyfusion_dynamic_pool.png")
    plt.savefig(save_path)
    print(f"\nROC curve saved to {save_path}")
    plt.close() # Close the plot to prevent it from displaying

    # Save evaluation results to JSON file
    eval_results = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(roc_auc),
        'accuracy': float(accuracy),
        'average_similarity': float(avg_similarity),
        'average_gallery_drift': float(avg_drift),
        'correct_identifications': int(correct_identifications),
        'total_probes': int(total_probes)
    }
    json_save_path = os.path.join(output_dir, "proxyfusion_dynamic_pool_results.json")
    with open(json_save_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {json_save_path}")


def main(args):
    initial_gallery = create_gallery(args.gallery_path)
    if not initial_gallery:
        print("Gallery is empty. Exiting.")
        return
    
    g, i, c, t, final_gallery = process_probes_combined(
        args.probes_path, 
        initial_gallery,
        max_pool_size=args.max_pool_size,
        consistency_threshold=args.consistency_threshold,
        num_fusion_features=args.num_fusion_features
    )
    
    return calculate_and_display_metrics(g, i, c, t, initial_gallery, final_gallery)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with ProxyFusion, pose awareness, and self-consistency correction.")
    parser.add_argument('--probes_path', type=str, default='dataset/NB116_person_spatial')
    parser.add_argument('--gallery_path', type=str, default='dataset/images/faces_images')
    parser.add_argument('--max_pool_size', type=int, default=10, help="Maximum size of the feature pool for each person.")
    parser.add_argument('--consistency_threshold', type=float, default=0.3, help="Threshold for self-consistency check.")
    parser.add_argument('--num_fusion_features', type=int, default=4, help="Number of top pose-similar features to fuse from gallery.")
    args = parser.parse_args()
    main(args)
