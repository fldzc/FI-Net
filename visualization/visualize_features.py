import os
import sys
import numpy as np
import cv2
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import argparse
import torch.nn.functional as F

# --- Path Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Baseline Method Imports (FastReID) ---
from fastreid.config import get_cfg
from fastreid.engine import default_setup, DefaultPredictor

# --- Proposed Method Imports (InsightFace) ---
from insightface.app import FaceAnalysis

def setup_fastreid_predictor(config_file, weights_path, device):
    """Sets up the FastReID predictor."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return DefaultPredictor(cfg), cfg

def extract_baseline_features(image_paths, predictor, cfg):
    """Extracts features using the baseline FastReID model."""
    features = []
    for img_path in tqdm(image_paths, desc="Extracting Baseline Features"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # The model expects RGB inputs
        img = img[:, :, ::-1]
        # Apply pre-processing to image.
        img = cv2.resize(img, tuple(cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # Make shape with a new batch dimension which is adapted for network input
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))[None]

        feat = predictor(img)
        feat = F.normalize(feat)
        features.append(feat.cpu().numpy())
    return np.vstack(features) if features else np.array([])

def setup_insightface_predictor():
    """Sets up the InsightFace predictor."""
    app = FaceAnalysis(name='antelopev2', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))
    return app

def extract_proposed_features(image_paths, predictor):
    """Extracts features using the proposed method (InsightFace)."""
    features = []
    for image_path in tqdm(image_paths, desc="Extracting Proposed Features"):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        faces = predictor.get(image)
        if len(faces) >= 1:
            # Take the first detected face
            embedding = faces[0].embedding
            features.append(embedding / np.linalg.norm(embedding))
        else:
            print(f"Warning: No face detected in {image_path}")
    return np.array(features) if features else np.array([])


def visualize_tsne_comparison(
    baseline_features, baseline_labels, baseline_label_map,
    proposed_features, proposed_labels, proposed_label_map,
    output_path
):
    """Performs t-SNE and plots a side-by-side comparison."""

    # --- t-SNE Calculation ---
    print("Running t-SNE for baseline features...")
    # Adjust perplexity if the number of samples is too small
    perplexity_baseline = min(30, baseline_features.shape[0] - 1)
    tsne_baseline = TSNE(n_components=2, random_state=42, perplexity=perplexity_baseline, max_iter=1000, init='pca', learning_rate='auto')
    baseline_features_2d = tsne_baseline.fit_transform(baseline_features)

    print("Running t-SNE for proposed features...")
    perplexity_proposed = min(30, proposed_features.shape[0] - 1)
    tsne_proposed = TSNE(n_components=2, random_state=42, perplexity=perplexity_proposed, max_iter=1000, init='pca', learning_rate='auto')
    proposed_features_2d = tsne_proposed.fit_transform(proposed_features)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))
    
    # Plot Baseline
    unique_labels_baseline = np.unique(baseline_labels)
    colors_baseline = plt.cm.get_cmap('viridis', len(unique_labels_baseline))
    for i, label in enumerate(unique_labels_baseline):
        idx = (baseline_labels == label)
        ax1.scatter(baseline_features_2d[idx, 0], baseline_features_2d[idx, 1], color=colors_baseline(i), label=f'Person {label + 1}', alpha=0.8, s=50)
    # ax1.set_title('Baseline Method (FastReID)', fontsize=18)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax1.legend(title="Identities")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Proposed
    unique_labels_proposed = np.unique(proposed_labels)
    # Ensure color consistency between plots if the labels are the same
    colors_proposed = plt.cm.get_cmap('viridis', len(unique_labels_proposed))
    for i, label in enumerate(unique_labels_proposed):
        idx = (proposed_labels == label)
        ax2.scatter(proposed_features_2d[idx, 0], proposed_features_2d[idx, 1], color=colors_proposed(i), label=f'Person {label + 1}', alpha=0.8, s=50)
    # ax2.set_title('Proposed Method (Spatial + Quality)', fontsize=18)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax2.legend(title="Identities")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print(f"Saving combined visualization to {output_path}")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def load_data(data_path, num_identities=5, max_images_per_id=20):
    """
    Loads image paths and labels from the dataset.
    It recursively finds all directories containing JPG files and treats them as unique identities.
    """
    image_paths = []
    labels = []
    label_map = {}
    current_label = 0
    identity_folders = []

    # Recursively find all directories that contain .jpg files
    for root, dirs, files in os.walk(data_path):
        if any(f.endswith('.jpg') for f in files):
            identity_folders.append(root)

    if not identity_folders:
        print(f"Error: No folders containing .jpg files found in {data_path}")
        return [], np.array([]), {}

    # Sort to ensure consistent ordering
    identity_folders.sort()

    # Select a subset of identity folders to visualize
    selected_folders = identity_folders[:num_identities]
    print(f"Loading data from {len(selected_folders)} identity folders: {[os.path.relpath(p, data_path) for p in selected_folders]}")

    for person_path in selected_folders:
        # Use the folder name as the identity
        person_id = os.path.basename(person_path)

        if person_id not in label_map:
            label_map[person_id] = current_label
            current_label += 1
        
        # Images are directly inside the person folder
        person_images = sorted(glob.glob(os.path.join(person_path, '*.jpg')))
        
        if not person_images:
            # This check is somewhat redundant now but good for safety
            print(f"Warning: No images found in {person_path}")
            continue

        # Limit the number of images per identity
        person_images = person_images[:max_images_per_id]
        image_paths.extend(person_images)
        labels.extend([label_map[person_id]] * len(person_images))
        
    return image_paths, np.array(labels), {v: k for k, v in label_map.items()}


def main(args):
    """Main function to run the feature visualization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Baseline Method Processing ---
    print("\n--- Processing Baseline Method ---")
    baseline_features, baseline_labels, baseline_label_map = None, None, None
    if not os.path.exists(args.baseline_config):
        print(f"Error: Baseline config file not found at {args.baseline_config}")
        return
    if not os.path.exists(args.baseline_weights):
        print(f"Error: Baseline weights file not found at {args.baseline_weights}")
        return
        
    fastreid_predictor, fastreid_cfg = setup_fastreid_predictor(args.baseline_config, args.baseline_weights, device)
    baseline_images, baseline_labels, baseline_label_map = load_data(args.baseline_data, args.num_ids, args.images_per_id)
    
    if baseline_images:
        baseline_features = extract_baseline_features(baseline_images, fastreid_predictor, fastreid_cfg)
        if baseline_features.shape[0] == 0:
            print("No baseline features were extracted.")
            baseline_features = None # Ensure it's None if empty
    else:
        print("No images found for the baseline method.")

    # --- Proposed Method Processing ---
    print("\n--- Processing Proposed Method ---")
    proposed_features, proposed_labels, proposed_label_map = None, None, None
    insightface_predictor = setup_insightface_predictor()
    proposed_images, proposed_labels, proposed_label_map = load_data(args.proposed_data, args.num_ids, args.images_per_id)

    if proposed_images:
        proposed_features = extract_proposed_features(proposed_images, insightface_predictor)
        if proposed_features.shape[0] == 0:
            print("No proposed features were extracted.")
            proposed_features = None # Ensure it's None if empty
    else:
        print("No images found for the proposed method.")
        
    # --- Combined Visualization ---
    if baseline_features is not None and proposed_features is not None:
        print("\n--- Generating Combined Visualization ---")
        visualize_tsne_comparison(
            baseline_features.squeeze(), baseline_labels, baseline_label_map,
            proposed_features, proposed_labels, proposed_label_map,
            os.path.join(output_dir, 'tsne_comparison.png')
        )
    else:
        print("\nSkipping combined visualization due to missing features for one or both methods.")

    print("\nVisualization script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize and compare feature distributions with t-SNE.")
    parser.add_argument('--baseline_config', type=str, default='reid/config/bagtricks_R50-ibn.yml',
                        help='Path to the FastReID config file for the baseline model.')
    parser.add_argument('--baseline_weights', type=str, default='models/veriwild_bot_R50-ibn.pth',
                        help='Path to the FastReID model weights for the baseline model.')
    parser.add_argument('--baseline_data', type=str, default='dataset/NB116_person',
                        help='Path to the baseline dataset.')
    parser.add_argument('--proposed_data', type=str, default='dataset/NB116_person_spatial',
                        help='Path to the proposed method dataset.')
    parser.add_argument('--output_dir', type=str, default='visualization/results',
                        help='Directory to save the t-SNE plots.')
    parser.add_argument('--num_ids', type=str, default=7,
                        help='Number of identities to visualize.')
    parser.add_argument('--images_per_id', type=str, default=50,
                        help='Maximum number of images per identity.')

    args = parser.parse_args()
    main(args)
