import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def run_evaluation_script(script_name):
    """
    运行指定的评估脚本并等待其完成。
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n--- Running evaluation: {script_name} ---")
    try:
        # 使用 sys.executable 确保使用与当前环境相同的 Python 解释器
        process = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8', # 指定编码
            errors='replace'  # 增加错误处理，防止解码中断
        )
        print(process.stdout)
        if process.stderr:
            print("--- Stderr ---")
            print(process.stderr)
        print(f"--- Finished: {script_name} ---\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}")
        return False

def load_results(json_path):
    """
    从JSON文件中加载评估结果。
    """
    if not os.path.exists(json_path):
        print(f"Error: Results file not found at {json_path}")
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_and_save_comparison(results_with_quality, results_no_quality, output_dir):
    """
    绘制并保存ROC曲线对比图。
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制带质量分数的ROC曲线
    if results_with_quality:
        fpr_q = results_with_quality['fpr']
        tpr_q = results_with_quality['tpr']
        auc_q = results_with_quality['auc']
        plt.plot(fpr_q, tpr_q, lw=2, label=f'With Quality (AUC = {auc_q:.4f})')

    # 绘制不带质量分数的ROC曲线
    if results_no_quality:
        fpr_nq = results_no_quality['fpr']
        tpr_nq = results_no_quality['tpr']
        auc_nq = results_no_quality['auc']
        plt.plot(fpr_nq, tpr_nq, lw=2, label=f'No Quality (AUC = {auc_nq:.4f})')

    # 绘制随机线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random Guess')

    # 设置图表样式
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    # plt.title('ROC Curve Comparison: Quality vs. No Quality')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    # 保存图像
    save_path = os.path.join(output_dir, "comparison_roc_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison ROC curve saved to {save_path}")
    plt.show()
    plt.close()

def main():
    """
    主函数：运行评估，加载结果，生成对比图并保存数据。
    """
    # 脚本文件名
    script_quality = 'eval_proxyfusion_quality.py'
    script_no_quality = 'eval_proxyfusion_no_quality.py'
    
    # 运行评估脚本
    success_quality = run_evaluation_script(script_quality)
    success_no_quality = run_evaluation_script(script_no_quality)
    
    # 结果文件路径
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    results_quality_path = os.path.join(output_dir, "quality_eval_results.json")
    results_no_quality_path = os.path.join(output_dir, "no_quality_eval_results.json")

    # 加载结果
    results_with_quality = load_results(results_quality_path) if success_quality else None
    results_no_quality = load_results(results_no_quality_path) if success_no_quality else None

    if not results_with_quality and not results_no_quality:
        print("Both evaluation scripts failed or produced no results. Cannot generate comparison.")
        return

    # 绘制对比图
    plot_and_save_comparison(results_with_quality, results_no_quality, output_dir)

    # 准备并保存用于后期可视化的聚合数据
    comparison_data = {
        "with_quality": results_with_quality,
        "no_quality": results_no_quality
    }
    
    comparison_json_path = os.path.join(output_dir, "comparison_roc_data.json")
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_data, f, indent=4)
    print(f"Aggregated data for visualization saved to {comparison_json_path}")


if __name__ == "__main__":
    import sys
    # 将父目录添加到sys.path以允许导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
