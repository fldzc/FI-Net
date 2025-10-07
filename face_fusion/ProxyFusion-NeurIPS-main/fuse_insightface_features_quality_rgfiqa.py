import os
import glob
import cv2
import numpy as np
import torch
import sys
from torchvision import transforms
from PIL import Image

# --- Add project root to sys.path ---
# This allows the script to be run from anywhere and find the models and tinyfqnet modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
# --------------------------------

from models.fusion_models import ProxyFusion
from insightface.app import FaceAnalysis
from rg_fiqa.models.rg_fiqa import RGFIQA


# Configuration
# IMG_DIR = r"C:\Project\Classroom-Reid\dataset\NB116_person\508NB116\Class_4\person_32"  # Path to your image folder
# TEST_IMG_PATH = r"C:\Project\Classroom-Reid\dataset\images\faces_images\3230637027.jpg"  # Path to the test image
IMG_DIR = r"C:\Project\Classroom-Reid\dataset\NB116_person_xh\508NB116\Class_3\3230411002"  # Path to your image folder
TEST_IMG_PATH = r"C:\Project\Classroom-Reid\dataset\images\faces_images\3230411002.jpg"  # Path to the test image
MODEL_PATH = r"C:\Project\Classroom-Reid\face_fusion\ProxyFusion-NeurIPS-main\checkpoints\ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar"  # Relative path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load RG-FIQA model ---
RGFIQA_CHECKPOINT_PATH = os.path.join(project_root, 'rg_fiqa', 'checkpoints', 'rg_fiqa_best.pth')
tfq_device = DEVICE
rgfiqa_model = RGFIQA().to(tfq_device)
try:
    rgfiqa_model.load_state_dict(torch.load(RGFIQA_CHECKPOINT_PATH, map_location=tfq_device))
    rgfiqa_model.eval()
    print("RG-FIQA model loaded successfully!")
except FileNotFoundError:
    print(f"Warning: RG-FIQA model checkpoint not found at {RGFIQA_CHECKPOINT_PATH}")
    rgfiqa_model = None

tfq_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

@torch.no_grad()
def rgfiqa_predict(image_bgr):
    """Predicts image quality score using the RG-FIQA model."""
    if rgfiqa_model is None or image_bgr is None:
        return 0.0
    
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = tfq_transform(rgb).unsqueeze(0).to(tfq_device)
    score = rgfiqa_model(x).item()
    return float(score)
# -------------------------


# ============ Quality Assessment Function (Replaced) ============
def compute_quality(image):
    """
    Computes the face image quality score.
    This version uses the pre-trained RG-FIQA model for prediction.
    """
    return rgfiqa_predict(image)


def get_comprehensive_quality_score(image, face):
    """
    Comprehensive quality assessment.
    This version uses only the RG-FIQA score.
    """
    if image is None or face is None:
        return 0.0
    
    # Use only the image quality score from RG-FIQA
    image_quality = compute_quality(image)
    
    return image_quality


# ============ Helper Functions from Original Script (Simplified) ============
def apply_quality_weights_v2(features_list, quality_scores):
    """
    Weighted average fusion.
    """
    if len(quality_scores) == 0 or len(features_list) == 0:
        return None
    
    quality_weights = np.array(quality_scores)
    quality_weights = quality_weights / (np.sum(quality_weights) + 1e-8)
    
    normalized_features = np.array([feat / np.linalg.norm(feat) for feat in features_list])
    
    weighted_avg = np.average(normalized_features, axis=0, weights=quality_weights)
    
    if np.linalg.norm(weighted_avg) == 0:
        return None
        
    weighted_avg = weighted_avg / np.linalg.norm(weighted_avg)
    
    return weighted_avg

# ============ Feature Extraction Function ============
def extract_features_from_folder_with_quality(img_dir, app):
    """Extracts image features from a folder, including quality assessment."""
    features, quality_scores, valid_images, skipped_images = [], [], [], []
    
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    print(f"Found {len(img_paths)} images")
    
    for img_path in img_paths:
        try:
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                skipped_images.append(os.path.basename(img_path))
                continue
            
            faces = app.get(image_cv)
            
            if len(faces) == 1:
                best_face = faces[0]
                quality_score = get_comprehensive_quality_score(image_cv, best_face)
                
                features.append(best_face.embedding)
                quality_scores.append(quality_score)
                valid_images.append(os.path.basename(img_path))
                
                print(f"✓ {os.path.basename(img_path)}: Total Score {quality_score:.3f}")
            else:
                skipped_images.append(os.path.basename(img_path))
                
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            skipped_images.append(os.path.basename(img_path))

    if len(features) > 0:
        return np.stack(features, axis=0), np.array(quality_scores), valid_images, skipped_images
    else:
        return np.array([]).reshape(0, 512), np.array([]), [], skipped_images


def fuse_features_with_quality(features, quality_scores, model_path, fusion_method='hybrid'):
    """
    Fuses features using quality weighting.
    """
    print(f"\n=== Starting Quality-Weighted Feature Fusion (Method: {fusion_method}) ===")

    if fusion_method == 'quality_only':
        print("Using quality-weighted fusion only.")
        if len(features) > 0:
            return apply_quality_weights_v2(features, quality_scores)
        return None

    try:
        model = ProxyFusion(DIM=512).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        state_dict_key = next((k for k in ["model_weights", "state_dict", "model"] if k in checkpoint or (isinstance(checkpoint.get("state_dict"), dict) and k in checkpoint["state_dict"])), None)
        if state_dict_key:
             actual_state_dict = checkpoint["state_dict"]["model_weights"] if state_dict_key == "model_weights" and "model_weights" in checkpoint["state_dict"] else (checkpoint.get(state_dict_key) or checkpoint)
             model.load_state_dict(actual_state_dict, strict=False)
        else:
             model.load_state_dict(checkpoint, strict=False)

        print("✓ ProxyFusion model loaded successfully!")
    except Exception as e:
        print(f"✗ Model loading failed: {e}. Using quality-weighted average as fallback.")
        return apply_quality_weights_v2(features, quality_scores)
    
    model.eval()
    with torch.no_grad():
        if len(features) > 1:
            if fusion_method == 'hybrid':
                quality_threshold = np.percentile(quality_scores, 20)
                high_quality_indices = quality_scores >= quality_threshold
                
                selected_features = features[high_quality_indices] if np.sum(high_quality_indices) >= 2 else features
                print(f"Selected {len(selected_features)}/{len(features)} features for ProxyFusion")

                normalized_features = np.array([feat / np.linalg.norm(feat) for feat in selected_features])
                
                input_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=DEVICE)
                fused = model.eval_fuse_probe(input_tensor)
                fused_feature = fused.mean(dim=0).cpu().numpy()
            else: # proxyfusion_only
                normalized_features = np.array([feat / np.linalg.norm(feat) for feat in features])
                input_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=DEVICE)
                fused = model.eval_fuse_probe(input_tensor)
                fused_feature = fused.mean(dim=0).cpu().numpy()
        else:
            fused_feature = features[0] if len(features) > 0 else None
        
        if fused_feature is not None:
             return fused_feature / np.linalg.norm(fused_feature)
    return None


def extract_single_image_feature_with_quality(img_path, app):
    """Extracts a single image feature, including quality assessment."""
    try:
        image_cv = cv2.imread(img_path)
        if image_cv is None: return None, 0
        
        faces = app.get(image_cv)
        if len(faces) > 0:
            best_face = max(faces, key=lambda f: f.det_score)
            quality_score = get_comprehensive_quality_score(image_cv, best_face)
            return best_face.embedding, quality_score
        return None, 0
            
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None, 0


def compute_cosine_similarity(feature1, feature2):
    """Computes the cosine similarity between two feature vectors."""
    if feature1 is None or feature2 is None: return 0.0
    feature1_norm = feature1 / (np.linalg.norm(feature1) + 1e-6)
    feature2_norm = feature2 / (np.linalg.norm(feature2) + 1e-6)
    return np.dot(feature1_norm, feature2_norm)


if __name__ == "__main__":
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))

    print("=== ProxyFusion + RG-FIQA Quality-Weighted Feature Fusion Test ===")
    
    folder_features, quality_scores, valid_images, skipped = extract_features_from_folder_with_quality(IMG_DIR, app)
    
    if folder_features.shape[0] == 0:
        print("No features extracted from the folder, exiting.")
        exit()
    
    fused_feature = fuse_features_with_quality(folder_features, quality_scores, MODEL_PATH)
    
    if fused_feature is None:
        print("Feature fusion failed, exiting.")
        exit()
    
    test_feature, test_quality = extract_single_image_feature_with_quality(TEST_IMG_PATH, app)
    
    if test_feature is None:
        print("Test image feature extraction failed, exiting.")
        exit()
    
    similarity = compute_cosine_similarity(fused_feature, test_feature)
    
    print(f"\n=== Final Result ===")
    print(f"Fused features from ({len(valid_images)} images):")
    for img, score in zip(valid_images, quality_scores):
        print(f"  - {img} (Quality: {score:.3f})")
    
    print(f"\nCosine Similarity: {similarity:.4f}")
