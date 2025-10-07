# encoding: utf-8
"""
Run four evaluation scripts under reid/val and summarize key metrics into a single txt/csv.

Usage (from project root):
    python reid/val/run_all_evals.py \
        --config-file reid/config/bagtricks_R50-ibn.yml \
        --dataset-path dataset/NB116_person_spatial \
        --output-root reid/val/combined_results

Note: This script creates separate output directories per eval script to avoid JSON name conflicts:
    - Base:    results_base/evaluation_results.json
    - Spatial: results_spatial/evaluation_results_spatial.json
    - IOU:     results_spatial_iou/evaluation_results_spatial.json
    - MD:      results_spatial_md/evaluation_results_spatial.json

Final summary outputs:
    - reid/val/combined_results/summary.txt
    - reid/val/combined_results/summary.csv
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def build_parser():
    parser = argparse.ArgumentParser(description="Run 4 eval scripts and summarize metrics")
    parser.add_argument(
        "--config-file",
        default="reid/config/bagtricks_R50-ibn.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset-path",
        default="dataset/NB116_person_spatial",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-root",
        default="reid/val/combined_results",
        help="Unified output root for all four evaluations (subdirs created)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold (passed to eval scripts)",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=None,
        help="Position weight (only for Spatial/IOU/MD; default if not provided)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multiprocessing during feature extraction (pass-through)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if output JSON exists",
    )
    return parser


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_eval(script_path: str, output_dir: str, args: argparse.Namespace, is_spatial: bool) -> str:
    """运行单个评估脚本，返回其结果 JSON 文件路径。"""
    python_exec = sys.executable
    ensure_dir(output_dir)

    # Result filename: base vs spatial
    result_filename = "evaluation_results_spatial.json" if is_spatial else "evaluation_results.json"
    result_path = os.path.join(output_dir, result_filename)

    if (not args.force) and os.path.isfile(result_path):
        print(f"[Skip] Result exists: {result_path}")
        return result_path

    cmd = [
        python_exec,
        script_path,
        "--config-file", args.config_file,
        "--dataset-path", args.dataset_path,
        "--output-path", output_dir,
        "--threshold", str(args.threshold),
    ]

    if args.parallel:
        cmd.append("--parallel")

    if is_spatial and args.position_weight is not None:
        cmd.extend(["--position-weight", str(args.position_weight)])

    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return result_path


def load_summary(json_path: str) -> dict:
    if not os.path.isfile(json_path):
        print(f"[Warning] Result JSON not found: {json_path}")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary", {})
    except Exception as e:
        print(f"[Error] Failed to parse JSON: {json_path} -> {e}")
        return {}


def fmt(v, digits=4):
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "-"


def write_summary_txt_csv(output_root: str, rows: list):
    """将汇总写入 txt 与 csv。"""
    ensure_dir(output_root)
    txt_path = os.path.join(output_root, "summary.txt")
    csv_path = os.path.join(output_root, "summary.csv")

    headers = [
        "Method",
        "Mean_Rank1", "Std_Rank1",
        "Mean_Rank5", "Std_Rank5",
        "Mean_mAP", "Std_mAP",
        "Mean_AUC", "Std_AUC",
        "Mean_ACC", "Std_ACC",
        "Mean_NMI", "Std_NMI",
        "Mean_RE",  "Std_RE",
        "Position_Weight",
        "Total_Courses",
    ]

    # TXT (aligned columns)
    col_widths = [max(len(h), 12) for h in headers]
    def pad(s, i):
        return str(s).ljust(col_widths[i])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Summary generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(" ".join(pad(h, i) for i, h in enumerate(headers)) + "\n")
        f.write("-" * (sum(col_widths) + len(col_widths) - 1) + "\n")
        for r in rows:
            f.write(" ".join(pad(r.get(h, "-"), i) for i, h in enumerate(headers)) + "\n")

    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")

    print(f"[Done] Summary written to:\n  {txt_path}\n  {csv_path}")


def main():
    args = build_parser().parse_args()

    # 统一输出根目录
    output_root = args.output_root
    ensure_dir(output_root)

    # 定义四个评估任务
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    scripts = [
        {
            "method": "Base",
            "script": os.path.join(project_root, "reid", "val", "eval_base.py"),
            "output": os.path.join(output_root, "results_base"),
            "is_spatial": False,
        },
        {
            "method": "Spatial",
            "script": os.path.join(project_root, "reid", "val", "eval_Spatial.py"),
            "output": os.path.join(output_root, "results_spatial"),
            "is_spatial": True,
        },
        {
            "method": "Spatial_IOU",
            "script": os.path.join(project_root, "reid", "val", "eval_Spatial_Iou.py"),
            "output": os.path.join(output_root, "results_spatial_iou"),
            "is_spatial": True,
        },
        {
            "method": "Spatial_MD",
            "script": os.path.join(project_root, "reid", "val", "eval_Spatial_MD.py"),
            "output": os.path.join(output_root, "results_spatial_md"),
            "is_spatial": True,
        },
    ]

    rows = []
    for item in scripts:
        try:
            result_json = run_eval(item["script"], item["output"], args, item["is_spatial"])
            summary = load_summary(result_json)
            if not summary:
                print(f"[Warning] No summary for {item['method']}, filling with '-'.")
                pos_w = summary.get("position_weight") if summary else "-"
                rows.append({
                    "Method": item["method"],
                    "Mean_Rank1": "-", "Std_Rank1": "-",
                    "Mean_Rank5": "-", "Std_Rank5": "-",
                    "Mean_mAP": "-",  "Std_mAP": "-",
                    "Mean_AUC": "-",  "Std_AUC": "-",
                    "Mean_ACC": "-",  "Std_ACC": "-",
                    "Mean_NMI": "-",  "Std_NMI": "-",
                    "Mean_RE":  "-",  "Std_RE":  "-",
                    "Position_Weight": pos_w,
                    "Total_Courses": "-",
                })
                continue

            rows.append({
                "Method": item["method"],
                "Mean_Rank1": fmt(summary.get("mean_rank1")),
                "Std_Rank1": fmt(summary.get("std_rank1")),
                "Mean_Rank5": fmt(summary.get("mean_rank5")),
                "Std_Rank5": fmt(summary.get("std_rank5")),
                "Mean_mAP": fmt(summary.get("mean_mAP")),
                "Std_mAP": fmt(summary.get("std_mAP")),
                "Mean_AUC": fmt(summary.get("mean_AUC")),
                "Std_AUC": fmt(summary.get("std_AUC")),
                "Mean_ACC": fmt(summary.get("mean_ACC")),
                "Std_ACC": fmt(summary.get("std_ACC")),
                "Mean_NMI": fmt(summary.get("mean_NMI")),
                "Std_NMI": fmt(summary.get("std_NMI")),
                "Mean_RE": fmt(summary.get("mean_RE")),
                "Std_RE": fmt(summary.get("std_RE")),
                "Position_Weight": summary.get("position_weight", "-"),
                "Total_Courses": summary.get("total_courses", "-"),
            })
        except subprocess.CalledProcessError as e:
            print(f"[Error] Running {item['method']} failed, return code {e.returncode}")
            rows.append({
                "Method": item["method"],
                "Mean_Rank1": "-", "Std_Rank1": "-",
                "Mean_Rank5": "-", "Std_Rank5": "-",
                "Mean_mAP": "-",  "Std_mAP": "-",
                "Mean_AUC": "-",  "Std_AUC": "-",
                "Mean_ACC": "-",  "Std_ACC": "-",
                "Mean_NMI": "-",  "Std_NMI": "-",
                "Mean_RE":  "-",  "Std_RE":  "-",
                "Position_Weight": "-",
                "Total_Courses": "-",
            })

    write_summary_txt_csv(output_root, rows)


if __name__ == "__main__":
    main()


