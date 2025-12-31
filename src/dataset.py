"""
Logo Recognition Dataset Module
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)
"""

import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xml.etree.ElementTree as ET


class LogoDetectionDataset(Dataset):
    """
    Dataset for Universal Logo Detector training.
    Supports PASCAL VOC format annotations.
    """
    
    def __init__(self, root_dir, annotation_dir=None, transform=None, 
                 img_size=512, mode='train'):
        """
        Args:
            root_dir: Directory containing images
            annotation_dir: Directory containing XML annotations (VOC format)
            transform: Optional transform to be applied
            img_size: Input image size (default 512x512 as per paper)
            mode: 'train' or 'val'
        """
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir or os.path.join(root_dir, 'annotations')
        self.img_size = img_size
        self.mode = mode
        
        # Get image list
        self.images = []
        self.annotations = []
        self._load_dataset()
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_dataset(self):
        """Load image paths and annotations"""
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for filename in os.listdir(self.root_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in img_extensions:
                img_path = os.path.join(self.root_dir, filename)
                ann_path = os.path.join(
                    self.annotation_dir,
                    os.path.splitext(filename)[0] + '.xml'
                )
                
                if os.path.exists(ann_path):
                    # 检查标注是否有效（至少有一个框）
                    boxes, _ = self._parse_voc_annotation(ann_path)
                    if len(boxes) > 0:
                        self.images.append(img_path)
                        self.annotations.append(ann_path)
    
    def _parse_voc_annotation(self, ann_path):
        """Parse VOC format XML annotation"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # For universal detector, all logos are class 1 (binary: logo vs background)
            labels.append(1)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Parse annotation
        boxes, labels = self._parse_voc_annotation(self.annotations[idx])
        
        # Scale boxes to new image size
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        
        scaled_boxes = []
        for box in boxes:
            scaled_boxes.append([
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y
            ])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        target = {
            'boxes': torch.tensor(scaled_boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        
        return image, target


class LogoRecognitionDataset(Dataset):
    """
    Dataset for Few-shot Logo Recognizer training.
    Uses cropped logo regions for metric learning.
    """
    
    def __init__(self, root_dir, class_file=None, transform=None,
                 img_size=160, mode='train'):
        """
        Args:
            root_dir: Directory with class subdirectories containing cropped logos
            class_file: Optional JSON file mapping class names to indices
            transform: Optional transform
            img_size: Input size (160x160 as per paper)
            mode: 'train' or 'val'
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        
        # Load class information
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all samples
        self.samples = []
        self._load_samples()
        
        # Default transforms (as per paper)
        if transform is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((img_size + 20, img_size + 20)),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                         saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def _load_samples(self):
        """Load all image samples with their class labels"""
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in img_extensions:
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        return len(self.classes)


class TripletLogoDataset(Dataset):
    """
    Triplet Dataset for training with triplet loss.
    Generates (anchor, positive, negative) triplets.
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: LogoRecognitionDataset instance
        """
        self.base_dataset = base_dataset
        self.transform = base_dataset.transform
        
        # Group samples by class
        self.class_to_samples = {}
        for idx, (img_path, label) in enumerate(base_dataset.samples):
            if label not in self.class_to_samples:
                self.class_to_samples[label] = []
            self.class_to_samples[label].append(idx)
        
        self.all_classes = list(self.class_to_samples.keys())
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get anchor
        anchor_path, anchor_label = self.base_dataset.samples[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        # Get positive (same class, different image)
        pos_candidates = [i for i in self.class_to_samples[anchor_label] if i != idx]
        if pos_candidates:
            pos_idx = random.choice(pos_candidates)
        else:
            pos_idx = idx  # Fallback if only one sample
        
        pos_path, _ = self.base_dataset.samples[pos_idx]
        pos_img = Image.open(pos_path).convert('RGB')
        
        # Get negative (different class)
        neg_classes = [c for c in self.all_classes if c != anchor_label]
        neg_class = random.choice(neg_classes)
        neg_idx = random.choice(self.class_to_samples[neg_class])
        neg_path, neg_label = self.base_dataset.samples[neg_idx]
        neg_img = Image.open(neg_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        return anchor_img, pos_img, neg_img, anchor_label, neg_label


def collate_fn_detection(batch):
    """Custom collate function for detection dataset"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets


def get_detection_dataloader(root_dir, annotation_dir=None, batch_size=8,
                            img_size=512, mode='train', num_workers=4):
    """Get DataLoader for logo detection"""
    dataset = LogoDetectionDataset(
        root_dir=root_dir,
        annotation_dir=annotation_dir,
        img_size=img_size,
        mode=mode
    )
    
    shuffle = (mode == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True
    )


def get_recognition_dataloader(root_dir, batch_size=32, img_size=160,
                               mode='train', num_workers=4, use_triplet=False):
    """Get DataLoader for logo recognition"""
    base_dataset = LogoRecognitionDataset(
        root_dir=root_dir,
        img_size=img_size,
        mode=mode
    )
    
    if use_triplet:
        dataset = TripletLogoDataset(base_dataset)
    else:
        dataset = base_dataset
    
    shuffle = (mode == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
