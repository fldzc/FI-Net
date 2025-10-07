import argparse
import json
import os
import sys
import matplotlib.pyplot as plt

# Ensure the feature_pooling directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'feature_pooling'))

# Import the main functions from the evaluation scripts
from dynamic_feature_pool import main as dfp_main
from dynamic_feature_pool_correction import main as dfpc_main
from dynamic_feature_pool_pose_aware import main as dfppa_main
from dynamic_feature_pool_pose_aware_correction import main as dfppac_main

def run_all(args):
    """
    Runs all four evaluation scripts, collects their ROC data,
    saves it to a JSON file, and plots a combined ROC curve.
    """
    all_roc_data = {}

    # Define the configurations for each script
    evaluations = {
        "DFP": dfp_main,
        "DFP_Correction": dfpc_main,
        "DFP_Pose_Aware": dfppa_main,
        "DFP_Pose_Aware_Correction": dfppac_main,
    }

    # Run each evaluation
    for name, main_func in evaluations.items():
        print(f"\n--- Running Evaluation: {name} ---")
        try:
            roc_data = main_func(args)
            if roc_data:
                all_roc_data[name] = roc_data
                print(f"--- Completed {name} ---")
            else:
                print(f"--- {name} did not return ROC data ---")
        except Exception as e:
            print(f"--- Error running {name}: {e} ---")

    # Save all ROC data to a single JSON file
    output_json_path = os.path.join(args.output_dir, 'all_roc_data.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(all_roc_data, f, indent=4)
    print(f"\nAll ROC data saved to {output_json_path}")

    # Plot all ROC curves on a single figure
    plt.figure(figsize=(10, 8))
    for name, data in all_roc_data.items():
        if "fpr" in data and "tpr" in data and "roc_auc" in data:
            plt.plot(data['fpr'], data['tpr'], lw=2, label=f'{name} (AUC = {data["roc_auc"]:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Combined ROC Curves for All Evaluation Strategies')
    plt.legend(loc="lower right")
    plt.grid(True, which="both")

    output_image_path = os.path.join(args.output_dir, 'roc_curve_combined.png')
    plt.savefig(output_image_path)
    print(f"Combined ROC curve saved to {output_image_path}")
    plt.show()

    # Save summary of results to a text file
    output_txt_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(output_txt_path, 'w') as f:
        f.write("Evaluation Summary:\n")
        f.write("===================\n\n")
        for name, data in all_roc_data.items():
            f.write(f"Strategy: {name}\n")
            if "roc_auc" in data:
                f.write(f"  AUC: {data['roc_auc']:.4f}\n")
            if "acc" in data:
                f.write(f"  Accuracy (ACC): {data['acc']:.4f}\n")
            if "avg_genuine_similarity" in data:
                f.write(f"  Average Genuine Similarity: {data['avg_genuine_similarity']:.4f}\n")
            if "avg_gallery_drift" in data:
                f.write(f"  Average Gallery Drift: {data['avg_gallery_drift']:.4f}\n")
            if "eer" in data:
                f.write(f"  EER: {data['eer']:.4f}\n")
            if "threshold" in data:
                f.write(f"  Threshold at EER: {data['threshold']:.4f}\n")
            if "tar_at_far" in data:
                f.write(f"  TAR @ FARs:\n")
                for far, tar in data['tar_at_far'].items():
                    f.write(f"    {far}: {tar:.4f}\n")
            f.write("\n")
    print(f"Evaluation summary saved to {output_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all feature pooling evaluations and combine results.")
    # Use paths from one of the scripts as defaults
    parser.add_argument('--probes_path', type=str, default='dataset/NB116_person_spatial',
                        help='Path to the root directory of probe images.')
    parser.add_argument('--gallery_path', type=str, default='dataset/images/faces_images',
                        help='Path to the directory of initial gallery images.')
    parser.add_argument('--output_dir', type=str, default='feature_pooling/results',
                        help='Directory to save the combined results (JSON data and plot).')
    # Add other relevant arguments that are shared or need to be passed
    parser.add_argument('--max_pool_size', type=int, default=10, help="Maximum size of the feature pool for each person.")
    parser.add_argument('--consistency_threshold', type=float, default=0.3, help="Threshold for self-consistency check.")
    
    args = parser.parse_args()
    run_all(args)
