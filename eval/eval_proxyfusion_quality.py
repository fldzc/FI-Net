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

# --- Path Setup ---
# Add parent directories to sys.path to allow for module imports
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
print(f"Using device: {DEVICE}")

# Initialize InsightFace Analysis model
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

# --- ProxyFusion Model Loading (Singleton) ---
fusion_model = None

def get_fusion_model():
    """Lazy loads and caches the ProxyFusion model."""
    global fusion_model
    if fusion_model is not None:
        return fusion_model

    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'face_fusion', 'ProxyFusion-NeurIPS-main', 'checkpoints', 'ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar')
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

# --- Quality Assessment Functions ---
@torch.no_grad()
def rgfiqa_predict(image_bgr):
    """Predicts image quality score using RG-FIQA."""
    if rgfiqa_model is None or image_bgr is None:
        return 0.0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = tfq_transform(rgb).unsqueeze(0).to(DEVICE)
    score = rgfiqa_model(x).item()
    return float(score)


def get_comprehensive_quality_score(image, face):
    """Calculates a comprehensive quality score from image, detection, and pose."""
    if image is None or face is None: return 0
    image_quality = rgfiqa_predict(image)

    return image_quality

# --- Feature Fusion ---
def fuse_person_features(person_path):
    """
    Fuses features from all images in a person's folder using quality-weighted fusion.
    """
    image_files = glob.glob(os.path.join(person_path, '*.jpg'))
    if not image_files: return None

    features, quality_scores = [], []
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None: continue
        
        faces = app.get(image)
        if len(faces) == 1:
            face = faces[0]
            features.append(face.embedding)
            quality_scores.append(get_comprehensive_quality_score(image, face))

    if not features: return None
    
    features = np.array(features, dtype=np.float32)
    quality_scores = np.array(quality_scores, dtype=np.float32)

    # Normalize features before fusion
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    fusion_model_instance = get_fusion_model()
    if fusion_model_instance and fusion_model_instance != "failed" and len(features) > 1:
        try:
            # Quality-based selection for ProxyFusion
            quality_threshold = np.percentile(quality_scores, 20)
            high_quality_indices = np.where(quality_scores >= quality_threshold)[0]
            if len(high_quality_indices) < 2: high_quality_indices = np.arange(len(features))
            
            selected_features = features[high_quality_indices]
            input_tensor = torch.tensor(selected_features, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                fused = fusion_model_instance.eval_fuse_probe(input_tensor)
            fused_feature = fused.mean(dim=0).cpu().numpy()
        except Exception as e:
            print(f"ProxyFusion failed, falling back to weighted average: {e}")
            # Fallback: simple quality-weighted average
            weights = quality_scores / (quality_scores.sum() + 1e-6)
            fused_feature = np.sum(features * weights[:, np.newaxis], axis=0)
    elif len(features) > 1:
        # Simple quality-weighted average
        weights = quality_scores / (quality_scores.sum() + 1e-6)
        fused_feature = np.sum(features * weights[:, np.newaxis], axis=0)
    else:
        fused_feature = features[0]
        
    norm = np.linalg.norm(fused_feature)
    if norm == 0:
        return fused_feature
    return fused_feature / norm


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
        # We assume that any directory containing jpg files is a person folder
        if any(f.endswith('.jpg') for f in files):
             # And the directory name is the student ID
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

    # ACC - Accuracy
    accuracy = (correct_identifications / total_probes) * 100
    error_rate = 100 - accuracy
    print(f"\n--- Evaluation Metrics ---")
    print(f"ER: {error_rate:.2f}%")
    print(f"ACC: {accuracy:.2f}%")

    # AS - Average Similarity for genuine matches
    avg_similarity = np.mean(genuine_scores) if genuine_scores else 0
    print(f"AS: {avg_similarity:.4f}")

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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, which="both")
    
    # Define output directory and filenames
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ROC curve figure
    save_path = os.path.join(output_dir, "roc_curve_quality.png")
    plt.savefig(save_path)
    print(f"\nROC curve saved to {save_path}")
    plt.close() # Close the figure to avoid display

    # Save ROC data to JSON file
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(roc_auc),
        'accuracy': float(accuracy),
        'average_similarity': float(avg_similarity),
        'correct_identifications': int(correct_identifications),
        'total_probes': int(total_probes)
    }
    json_save_path = os.path.join(output_dir, "quality_eval_results.json")
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
    parser = argparse.ArgumentParser(description="Evaluate face recognition performance with feature fusion.")
    parser.add_argument('--probes_path', type=str, default='dataset/NB116_person_spatial',
                        help='Path to the root directory of probe images.')
    parser.add_argument('--gallery_path', type=str, default='dataset/images/faces_images',
                        help='Path to the directory of gallery images.')
    args = parser.parse_args()
    
    main(args)
