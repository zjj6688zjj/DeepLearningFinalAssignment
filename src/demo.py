"""
Quick Demo for Logo Recognition System
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)

This script demonstrates:
1. Model architecture overview
2. Simple forward pass test
3. How to prepare data and train
"""

import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import UniversalLogoDetector
from models.recognizer import (
    FewShotLogoRecognizer,
    SEResNet50,
    ProxyTripletLoss
)


def test_detector():
    """Test Universal Logo Detector"""
    print("=" * 60)
    print("Testing Universal Logo Detector (Faster R-CNN)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    detector = UniversalLogoDetector(
        num_classes=2,
        pretrained_backbone=True,
        min_size=512,
        max_size=512
    ).to(device)
    
    print(f"Model created successfully")
    
    # Test forward pass with random input
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # Inference mode
    detector.eval()
    with torch.no_grad():
        predictions = detector([img for img in dummy_images])
    
    print(f"\nInference output for {batch_size} images:")
    for i, pred in enumerate(predictions):
        print(f"  Image {i+1}: {len(pred['boxes'])} detections")
        if len(pred['boxes']) > 0:
            print(f"    Top score: {pred['scores'][0].item():.4f}")
    
    # Training mode
    detector.train()
    dummy_targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], 
                                  dtype=torch.float32).to(device),
            'labels': torch.tensor([1, 1], dtype=torch.int64).to(device)
        }
        for _ in range(batch_size)
    ]
    
    loss_dict = detector([img for img in dummy_images], dummy_targets)
    total_loss = sum(loss for loss in loss_dict.values())
    
    print(f"\nTraining losses:")
    for name, loss in loss_dict.items():
        print(f"  {name}: {loss.item():.4f}")
    print(f"  Total: {total_loss.item():.4f}")
    
    return True


def test_recognizer():
    """Test Few-shot Logo Recognizer"""
    print("\n" + "=" * 60)
    print("Testing Few-shot Logo Recognizer (SE-ResNet50 + STN)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 100  # Example: 100 logo classes
    
    # Create model
    recognizer = FewShotLogoRecognizer(
        num_classes=num_classes,
        embedding_dim=128,
        use_stn=True,
        margin=0.2,
        loss_type='proxy_triplet'
    ).to(device)
    
    print(f"Model created with {num_classes} classes")
    
    # Test forward pass
    batch_size = 32
    dummy_images = torch.randn(batch_size, 3, 160, 160).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Get embeddings
    embeddings = recognizer(dummy_images)
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding norm (should be ~1.0): {embeddings.norm(dim=1).mean():.4f}")
    
    # Compute loss
    loss = recognizer.compute_loss(embeddings, dummy_labels)
    print(f"Proxy-Triplet Loss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful")
    
    return True


def test_components():
    """Test individual components"""
    print("\n" + "=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test SE-ResNet50 backbone
    print("\n1. SE-ResNet50 Backbone:")
    backbone = SEResNet50(embedding_dim=128, use_stn=False).to(device)
    x = torch.randn(4, 3, 160, 160).to(device)
    out = backbone(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test Proxy-Triplet Loss
    print("\n2. Proxy-Triplet Loss:")
    loss_fn = ProxyTripletLoss(num_classes=50, embedding_dim=128, margin=0.2).to(device)
    embeddings = torch.randn(32, 128).to(device)
    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, 50, (32,)).to(device)
    loss = loss_fn(embeddings, labels)
    print(f"   Loss value: {loss.item():.4f}")
    
    return True


def print_usage():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    
    print("""
1. PREPARE DATA:
   
   For Detector (VOC format):
   data/
   ├── train/
   │   ├── images/          # JPG/PNG images
   │   └── annotations/     # XML files (VOC format)
   └── val/
       ├── images/
       └── annotations/
   
   For Recognizer:
   data/
   └── logos/
       ├── adidas/          # Class subdirectory
       │   ├── img1.jpg
       │   └── img2.jpg
       ├── nike/
       └── ...

2. TRAIN DETECTOR:
   
   python train.py --mode detector \\
       --train_dir ./data/train/images \\
       --train_ann_dir ./data/train/annotations \\
       --val_dir ./data/val/images \\
       --val_ann_dir ./data/val/annotations \\
       --epochs 20 \\
       --batch_size 8

3. TRAIN RECOGNIZER:
   
   python train.py --mode recognizer \\
       --train_dir ./data/logos/train \\
       --val_dir ./data/logos/val \\
       --epochs 100 \\
       --batch_size 32 \\
       --use_stn

4. INFERENCE:
   
   python inference.py \\
       --detector ./checkpoints/detector_best.pth \\
       --recognizer ./checkpoints/recognizer_best.pth \\
       --reference_dir ./data/logos/reference \\
       --input ./test_image.jpg \\
       --visualize

5. PAPER RESULTS (FlickrLogos-32):
   
   - Universal Detector: 79.87% Recall, 0.42 AP
   - Few-shot Recognizer: 97.16% Top-1 Recall
   - End-to-end: 56.55% mAP@5 (state-of-the-art)
""")


def main():
    print("=" * 60)
    print("Logo Recognition System Demo")
    print("Based on: 'Scalable Logo Recognition using Proxies'")
    print("IEEE WACV 2019")
    print("=" * 60)
    
    try:
        # Test detector
        detector_ok = test_detector()
        
        # Test recognizer
        recognizer_ok = test_recognizer()
        
        # Test components
        components_ok = test_components()
        
        if detector_ok and recognizer_ok and components_ok:
            print("\n" + "=" * 60)
            print("All tests passed!")
            print("=" * 60)
        
        # Print usage
        print_usage()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
