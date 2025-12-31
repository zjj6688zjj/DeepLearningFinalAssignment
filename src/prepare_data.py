"""
数据准备脚本
下载和准备 FlickrLogos-32 数据集（公开可用的logo数据集）
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET


# FlickrLogos-32 数据集信息
# 官方下载需要申请，这里提供数据组织结构

FLICKRLOGOS32_CLASSES = [
    'adidas', 'aldi', 'apple', 'becks', 'bmw', 'carlsberg', 'chimay',
    'cocacola', 'corona', 'dhl', 'erdinger', 'esso', 'fedex', 'ferrari',
    'ford', 'fosters', 'google', 'guiness', 'heineken', 'hp', 'milka',
    'nvidia', 'paulaner', 'pepsi', 'rittersport', 'shell', 'singha',
    'starbucks', 'stellaartois', 'texaco', 'tsingtao', 'ups'
]


def create_data_structure(base_dir):
    """创建数据目录结构"""
    print("创建数据目录结构...")
    
    dirs = [
        # 检测器数据
        'detection/train/images',
        'detection/train/annotations',
        'detection/val/images',
        'detection/val/annotations',
        
        # 识别器数据 (按logo类别分文件夹)
        'recognition/train',
        'recognition/val',
        'recognition/reference',
        
        # 原始数据存放
        'raw/flickrlogos32',
    ]
    
    for d in dirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"  创建: {path}")
    
    # 为识别器创建类别文件夹
    for split in ['train', 'val', 'reference']:
        for cls in FLICKRLOGOS32_CLASSES:
            path = os.path.join(base_dir, 'recognition', split, cls)
            os.makedirs(path, exist_ok=True)
    
    print(f"  创建 {len(FLICKRLOGOS32_CLASSES)} 个logo类别文件夹")


def create_voc_annotation(img_path, boxes, output_path):
    """
    创建VOC格式的XML标注文件
    
    boxes: list of [xmin, ymin, xmax, ymax, class_name]
    """
    from PIL import Image
    
    img = Image.open(img_path)
    width, height = img.size
    
    root = ET.Element('annotation')
    
    ET.SubElement(root, 'filename').text = os.path.basename(img_path)
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    
    for box in boxes:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = 'logo'  # 通用logo类别
        ET.SubElement(obj, 'difficult').text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(box[0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(box[1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(box[2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(box[3]))
    
    tree = ET.ElementTree(root)
    tree.write(output_path)


def load_bboxes_from_txt(mask_file):
    """从 .bboxes.txt 文件中读取边界框，返回 [xmin, ymin, xmax, ymax] 列表"""
    boxes = []
    if not os.path.exists(mask_file):
        return boxes
    
    with open(mask_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('x'):
                # 跳过表头或空行
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes


def build_detection_split(raw_root, relpaths_file, images_dir, ann_dir, split_name):
    """根据 relpaths 划分构建检测器数据集"""
    print(f"  构建检测器 {split_name} 集: {os.path.basename(relpaths_file)}")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    
    with open(relpaths_file, 'r', encoding='utf-8') as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            # 例如: classes/jpg/adidas/144503924.jpg
            parts = rel.split('/')
            if len(parts) < 4:
                continue
            brand = parts[2]
            filename = parts[-1]
            
            img_path = os.path.join(raw_root, *parts)
            if not os.path.exists(img_path):
                continue
            
            boxes = []
            if brand != 'no-logo':
                mask_path = os.path.join(
                    raw_root, 'classes', 'masks', brand, filename + '.bboxes.txt'
                )
                if os.path.exists(mask_path):
                    boxes = load_bboxes_from_txt(mask_path)
            
            dst_img_path = os.path.join(images_dir, filename)
            if not os.path.exists(dst_img_path):
                shutil.copy2(img_path, dst_img_path)
            
            xml_path = os.path.join(ann_dir, os.path.splitext(filename)[0] + '.xml')
            create_voc_annotation(img_path, boxes, xml_path)


def build_recognition_split(raw_root, relpaths_file, output_root, split_name):
    """根据 relpaths 构建识别器数据集（裁剪 logo 区域）"""
    from PIL import Image
    
    print(f"  构建识别器 {split_name} 集: {os.path.basename(relpaths_file)}")
    os.makedirs(output_root, exist_ok=True)
    
    with open(relpaths_file, 'r', encoding='utf-8') as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            parts = rel.split('/')
            if len(parts) < 4:
                continue
            brand = parts[2]
            if brand == 'no-logo':
                # 识别器只关心有 logo 的类别
                continue
            filename = parts[-1]
            
            img_path = os.path.join(raw_root, *parts)
            if not os.path.exists(img_path):
                continue
            
            mask_path = os.path.join(
                raw_root, 'classes', 'masks', brand, filename + '.bboxes.txt'
            )
            if not os.path.exists(mask_path):
                continue
            
            boxes = load_bboxes_from_txt(mask_path)
            if not boxes:
                continue
            
            img = Image.open(img_path).convert('RGB')
            
            for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                xmin_i = int(round(xmin))
                ymin_i = int(round(ymin))
                xmax_i = int(round(xmax))
                ymax_i = int(round(ymax))
                
                crop = img.crop((xmin_i, ymin_i, xmax_i, ymax_i))
                cls_dir = os.path.join(output_root, brand)
                os.makedirs(cls_dir, exist_ok=True)
                crop_name = f"{os.path.splitext(filename)[0]}_{idx}.jpg"
                crop_path = os.path.join(cls_dir, crop_name)
                crop.save(crop_path)


def build_reference_from_train(rec_train_root, rec_ref_root, max_per_class=5):
    """从训练集中为每个类别拷贝少量样本作为 reference"""
    print("  构建 reference 集 (每类最多拷贝几张样本)...")
    os.makedirs(rec_ref_root, exist_ok=True)
    
    for cls in os.listdir(rec_train_root):
        cls_train_dir = os.path.join(rec_train_root, cls)
        if not os.path.isdir(cls_train_dir):
            continue
        images = sorted([
            f for f in os.listdir(cls_train_dir)
            if os.path.isfile(os.path.join(cls_train_dir, f))
        ])
        if not images:
            continue
        
        cls_ref_dir = os.path.join(rec_ref_root, cls)
        os.makedirs(cls_ref_dir, exist_ok=True)
        
        for name in images[:max_per_class]:
            src = os.path.join(cls_train_dir, name)
            dst = os.path.join(cls_ref_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)


def process_flickrlogos_dataset(raw_root, base_dir):
    """从原始 FlickrLogos-v2 生成检测器和识别器所需数据"""
    print("\n开始从 FlickrLogos-v2 构建数据集...")
    
    # 检测器数据目录
    det_train_images = os.path.join(base_dir, 'detection', 'train', 'images')
    det_train_anns = os.path.join(base_dir, 'detection', 'train', 'annotations')
    det_val_images = os.path.join(base_dir, 'detection', 'val', 'images')
    det_val_anns = os.path.join(base_dir, 'detection', 'val', 'annotations')
    
    # 识别器数据目录
    rec_train_root = os.path.join(base_dir, 'recognition', 'train')
    rec_val_root = os.path.join(base_dir, 'recognition', 'val')
    rec_ref_root = os.path.join(base_dir, 'recognition', 'reference')
    
    # 构建检测器数据：使用 trainvalset 作为训练，valset 作为验证
    trainval_relpaths = os.path.join(raw_root, 'trainvalset.relpaths.txt')
    val_relpaths = os.path.join(raw_root, 'valset.relpaths.txt')
    if os.path.exists(trainval_relpaths):
        build_detection_split(raw_root, trainval_relpaths, det_train_images, det_train_anns, 'train')
    if os.path.exists(val_relpaths):
        build_detection_split(raw_root, val_relpaths, det_val_images, det_val_anns, 'val')
    
    # 构建识别器数据：只用带 logo 的样本
    train_relpaths = os.path.join(raw_root, 'trainset.relpaths.txt')
    val_logos_relpaths = os.path.join(raw_root, 'valset-logosonly.relpaths.txt')
    if os.path.exists(train_relpaths):
        build_recognition_split(raw_root, train_relpaths, rec_train_root, 'train')
    if os.path.exists(val_logos_relpaths):
        build_recognition_split(raw_root, val_logos_relpaths, rec_val_root, 'val')
    
    # 构建 reference 集
    if os.path.isdir(rec_train_root):
        build_reference_from_train(rec_train_root, rec_ref_root)
    
    print("FlickrLogos-32 数据转换完成。")


def print_data_instructions():
    """打印数据准备说明"""
    
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           数据准备指南                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  推荐数据集: FlickrLogos-32                                                   ║
║  官方网站: http://www.multimedia-computing.de/flickrlogos/                   ║
║                                                                              ║
║  【下载步骤】                                                                  ║
║                                                                              ║
║  1. 访问官网填写申请表获取下载链接                                              ║
║     或使用镜像: https://www.kaggle.com/datasets/faisalzaman23/flickrlogos32  ║
║                                                                              ║
║  2. 下载后解压到:                                                              ║
║     logo_recognition/data/raw/flickrlogos32/                                 ║
║                                                                              ║
║  【数据结构】                                                                  ║
║                                                                              ║
║  解压后应该包含:                                                               ║
║  - flickrlogos32_images.zip (图片)                                           ║
║  - flickrlogos32_annotations.zip (标注)                                      ║
║                                                                              ║
║  【替代方案 - 使用自己的数据】                                                  ║
║                                                                              ║
║  检测器数据格式 (VOC):                                                        ║
║  data/detection/train/                                                       ║
║  ├── images/                                                                 ║
║  │   ├── img001.jpg                                                          ║
║  │   └── img002.jpg                                                          ║
║  └── annotations/                                                            ║
║      ├── img001.xml  (VOC格式)                                               ║
║      └── img002.xml                                                          ║
║                                                                              ║
║  识别器数据格式 (分类文件夹):                                                   ║
║  data/recognition/train/                                                     ║
║  ├── adidas/                                                                 ║
║  │   ├── logo1.jpg                                                           ║
║  │   └── logo2.jpg                                                           ║
║  ├── nike/                                                                   ║
║  │   └── logo1.jpg                                                           ║
║  └── ...                                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    print("=" * 60)
    print("Logo Recognition 数据准备工具")
    print("=" * 60)
    
    # 创建目录结构
    create_data_structure(base_dir)
    
    # 如果检测到原始 FlickrLogos-32 数据，自动转换
    raw_root = os.path.join(base_dir, 'raw', 'flickrlogs32', 'FlickrLogos-v2')
    if os.path.isdir(raw_root):
        print("\n检测到 FlickrLogos-v2 原始数据，开始自动转换...")
        process_flickrlogos_dataset(raw_root, base_dir)
    else:
        print("\n未检测到 FlickrLogos-v2 原始数据，仅创建目录结构。")
    
    # 打印说明
    print_data_instructions()
    
    print("\n数据目录已创建完成!")
    print(f"位置: {os.path.abspath(base_dir)}")


if __name__ == '__main__':
    main()
