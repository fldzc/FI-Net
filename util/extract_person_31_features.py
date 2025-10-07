#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取person_31文件夹下所有人脸图片特征并保存为npy格式
"""

import os
import cv2
import numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import argparse

def extract_face_features(image_folder, output_file, save_format='dict'):
    """
    提取指定文件夹下所有人脸图片的特征
    
    Args:
        image_folder (str): 包含人脸图片的文件夹路径
        output_file (str): 输出的npy文件路径
        save_format (str): 保存格式 'dict'(默认) 或 'array_only'
    """
    
    # 初始化InsightFace模型
    print("正在初始化InsightFace模型...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    # 获取所有jpg图片文件
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")) + list(image_folder.glob("*.png"))
    image_files.sort()  # 按文件名排序
    
    print(f"找到 {len(image_files)} 张图片")
    
    if len(image_files) == 0:
        print("错误：未找到任何图片文件")
        return
    
    features_list = []
    image_names = []
    
    for i, image_path in enumerate(image_files):
        print(f"处理图片 {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"警告：无法读取图片 {image_path.name}")
                continue
                
            # 检测人脸并提取特征
            faces = app.get(img)
            
            if len(faces) == 0:
                print(f"警告：在图片 {image_path.name} 中未检测到人脸")
                continue
            elif len(faces) > 1:
                print(f"警告：在图片 {image_path.name} 中检测到多个人脸，跳过")
                continue
            # 获取第一个人脸的特征向量
            face = faces[0]
            feature = face.embedding  # 512维特征向量
            
            # 确保特征为float32类型
            feature = feature.astype(np.float32)
            
            # 归一化特征向量
            feature = feature / np.linalg.norm(feature)
            
            features_list.append(feature)
            image_names.append(image_path.name)
            
        except Exception as e:
            print(f"错误：处理图片 {image_path.name} 时出错: {str(e)}")
            continue
    
    if len(features_list) == 0:
        print("错误：未成功提取任何特征")
        return
    
    # 将特征列表转换为numpy数组，确保为float32类型
    features_array = np.array(features_list, dtype=np.float32)
    
    print(f"成功提取 {len(features_list)} 个特征向量")
    print(f"特征数组形状: {features_array.shape}")
    
    # 根据保存格式选择保存方式
    if save_format == 'array_only':
        # 直接保存特征数组，可以直接使用
        np.save(output_file, features_array)
        print(f"特征数组已保存到: {output_file}")
        print(f"可以直接使用: features = np.load('{output_file}')")
        
        # 同时保存图片名称到单独文件
        names_file = output_file.replace('.npy', '_names.npy')
        np.save(names_file, image_names)
        print(f"图片名称已保存到: {names_file}")
    else:
        # 保存为字典格式（原有方式）
        output_data = {
            'features': features_array,
            'image_names': image_names,
            'person_id': 'person_31',
            'total_images': len(image_files),
            'successful_extractions': len(features_list)
        }
        np.save(output_file, output_data)
        print(f"特征字典已保存到: {output_file}")
        print(f"使用方式: data = np.load('{output_file}', allow_pickle=True).item(); features = data['features']")
    
    # 打印统计信息
    print("\n=== 提取统计 ===")
    print(f"总图片数: {len(image_files)}")
    print(f"成功提取特征数: {len(features_list)}")
    print(f"成功率: {len(features_list)/len(image_files)*100:.1f}%")
    print(f"特征维度: {features_array.shape[1]}")
    
    return features_array, image_names

def load_and_verify_features(npy_file):
    """
    加载并验证保存的特征文件
    
    Args:
        npy_file (str): npy文件路径
    """
    print(f"\n=== 验证特征文件 ===")
    
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        
        features = data['features']
        image_names = data['image_names']
        person_id = data['person_id']
        
        print(f"人员ID: {person_id}")
        print(f"特征数组形状: {features.shape}")
        print(f"图片数量: {len(image_names)}")
        print(f"总图片数: {data['total_images']}")
        print(f"成功提取数: {data['successful_extractions']}")
        
        # 显示前5个图片名称
        print(f"\n前5个图片名称:")
        for i, name in enumerate(image_names[:5]):
            print(f"  {i+1}. {name}")
        
        # 验证特征向量的范围
        print(f"\n特征统计:")
        print(f"  最小值: {features.min():.6f}")
        print(f"  最大值: {features.max():.6f}")
        print(f"  均值: {features.mean():.6f}")
        print(f"  标准差: {features.std():.6f}")
        
        # 检查是否已归一化
        norms = np.linalg.norm(features, axis=1)
        print(f"  L2范数 - 最小: {norms.min():.6f}, 最大: {norms.max():.6f}, 均值: {norms.mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"错误：加载特征文件失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='提取person_31文件夹下所有人脸图片特征')
    parser.add_argument('--input_folder', type=str, 
                       default=r'c:\Project\Classroom-Reid\dataset\NB116_person\508NB116\Class_4\person_31',
                       help='输入图片文件夹路径')
    parser.add_argument('--output_file', type=str, 
                       default=r'c:\Project\Classroom-Reid\person_31_features.npy',
                       help='输出npy文件路径')
    parser.add_argument('--verify', action='store_true', 
                       help='提取完成后验证结果')
    parser.add_argument('--save_format', type=str, choices=['dict', 'array_only'], 
                       default='dict', help='保存格式: dict(字典格式) 或 array_only(直接数组格式)')
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input_folder):
        print(f"错误：输入文件夹不存在: {args.input_folder}")
        return
    
    print(f"输入文件夹: {args.input_folder}")
    print(f"输出文件: {args.output_file}")
    print("="*50)
    
    # 提取特征
    try:
        features, image_names = extract_face_features(args.input_folder, args.output_file, args.save_format)
        
        if features is not None:
            print("\n特征提取完成！")
            
            # 如果指定了验证选项，则验证结果
            if args.verify:
                load_and_verify_features(args.output_file)
        else:
            print("特征提取失败！")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()