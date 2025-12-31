"""
Training Scripts for Logo Recognition System
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)

Training consists of two stages:
1. Train Universal Logo Detector (Faster R-CNN)
2. Train Few-shot Logo Recognizer (SE-ResNet50 + Proxy-Triplet Loss)
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (
    LogoDetectionDataset, LogoRecognitionDataset,
    get_detection_dataloader, get_recognition_dataloader,
    collate_fn_detection
)
from models.detector import (
    UniversalLogoDetector, compute_detection_loss, compute_detection_metrics
)
from models.recognizer import (
    FewShotLogoRecognizer, create_optimizer, create_scheduler
)


# ============================================================================
# Training for Universal Logo Detector
# ============================================================================

def train_detector(args):
    """
    Train Universal Logo Detector
    
    Paper specifications:
    - Input size: 512x512
    - Epochs: 20
    - Backbone: ResNet50 pretrained on ImageNet
    """
    print("=" * 60)
    print("Training Universal Logo Detector")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UniversalLogoDetector(
        num_classes=2,  # background + logo
        pretrained_backbone=True,
        min_size=args.img_size,
        max_size=args.img_size
    ).to(device)
    
    # Create datasets
    train_dataset = LogoDetectionDataset(
        root_dir=args.train_dir,
        annotation_dir=args.train_ann_dir,
        img_size=args.img_size,
        mode='train'
    )
    
    val_dataset = LogoDetectionDataset(
        root_dir=args.val_dir,
        annotation_dir=args.val_ann_dir,
        img_size=args.img_size,
        mode='val'
    ) if args.val_dir else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True
    ) if val_dataset else None
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, 'detector'))
    
    # Training loop
    best_recall = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(list(images), targets)
            total_loss = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader and (epoch + 1) % args.val_freq == 0:
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Validating"):
                    images = images.to(device)
                    # Move targets to device for metric computation
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    predictions = model(list(images))
                    
                    all_predictions.extend(predictions)
                    all_targets.extend(targets)
            
            metrics = compute_detection_metrics(all_predictions, all_targets)
            print(f"Validation - Recall: {metrics['recall']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}")
            
            writer.add_scalar('Metrics/recall', metrics['recall'], epoch)
            writer.add_scalar('Metrics/precision', metrics['precision'], epoch)
            
            # Save best model
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall': best_recall,
                }, os.path.join(args.save_dir, 'detector_best.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'detector_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.save_dir, 'detector_final.pth'))
    
    writer.close()
    print("Detector training complete!")


# ============================================================================
# Training for Few-shot Logo Recognizer
# ============================================================================

def train_recognizer(args):
    """
    Train Few-shot Logo Recognizer
    
    Paper specifications:
    - Input size: 160x160
    - Embedding dimension: 128
    - Batch size: 32
    - Optimizer: Adam (lr=1e-4, momentum=0.9, weight_decay=5e-4)
    - LR schedule: decay by 0.8 every 20 epochs
    - Loss: Proxy-Triplet Loss
    """
    print("=" * 60)
    print("Training Few-shot Logo Recognizer")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = LogoRecognitionDataset(
        root_dir=args.train_dir,
        img_size=args.img_size,
        mode='train'
    )
    
    val_dataset = LogoRecognitionDataset(
        root_dir=args.val_dir,
        img_size=args.img_size,
        mode='val'
    ) if args.val_dir else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if val_dataset else None
    
    num_classes = train_dataset.get_num_classes()
    print(f"Number of logo classes: {num_classes}")
    
    # Create model
    model = FewShotLogoRecognizer(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        use_stn=args.use_stn,
        margin=args.margin,
        loss_type=args.loss_type
    ).to(device)
    
    # Optimizer (as per paper)
    optimizer = create_optimizer(
        model,
        lr=args.lr,
        weight_decay=5e-4,
        momentum=0.9
    )
    
    # LR scheduler (decay by 0.8 every 20 epochs)
    scheduler = create_scheduler(optimizer, step_size=20, gamma=0.8)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, 'recognizer'))
    
    # Training loop
    best_recall = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings = model(images)
            loss = model.compute_loss(embeddings, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Validation
        if val_loader and (epoch + 1) % args.val_freq == 0:
            recall = evaluate_recognizer(model, val_loader, train_loader, device)
            print(f"Validation - Top1 Recall: {recall:.4f}")
            
            writer.add_scalar('Metrics/top1_recall', recall, epoch)
            
            # Save best model
            if recall > best_recall:
                best_recall = recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'num_classes': num_classes,
                    'embedding_dim': args.embedding_dim,
                    'recall': best_recall,
                }, os.path.join(args.save_dir, 'recognizer_best.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'embedding_dim': args.embedding_dim,
            }, os.path.join(args.save_dir, f'recognizer_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'embedding_dim': args.embedding_dim,
    }, os.path.join(args.save_dir, 'recognizer_final.pth'))
    
    writer.close()
    print("Recognizer training complete!")


def evaluate_recognizer(model, val_loader, train_loader, device):
    """
    Evaluate recognizer using K-NN retrieval
    
    Returns:
        Top-1 recall
    """
    model.eval()
    
    # Build reference set from training data
    ref_embeddings = []
    ref_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building references"):
            images = images.to(device)
            embeddings = model(images)
            ref_embeddings.append(embeddings.cpu())
            ref_labels.append(labels)
    
    ref_embeddings = torch.cat(ref_embeddings, dim=0)
    ref_labels = torch.cat(ref_labels, dim=0)
    
    # Evaluate on validation set
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            query_embeddings = model(images).cpu()
            
            # K-NN search
            dist_matrix = 2 - 2 * torch.mm(query_embeddings, ref_embeddings.t())
            _, indices = dist_matrix.min(dim=1)
            predictions = ref_labels[indices]
            
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    return correct / total


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Logo Recognition Training')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['detector', 'recognizer'],
                       help='Training mode: detector or recognizer')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Training data directory')
    parser.add_argument('--train_ann_dir', type=str, default=None,
                       help='Training annotations directory (for detector)')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Validation data directory')
    parser.add_argument('--val_ann_dir', type=str, default=None,
                       help='Validation annotations directory (for detector)')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=512,
                       help='Input image size (512 for detector, 160 for recognizer)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension (recognizer only)')
    parser.add_argument('--use_stn', action='store_true', default=True,
                       help='Use Spatial Transformer Network')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Triplet loss margin')
    parser.add_argument('--loss_type', type=str, default='proxy_triplet',
                       choices=['proxy_triplet', 'proxy_nca'],
                       help='Loss function type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='TensorBoard log directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Model checkpoint directory')
    parser.add_argument('--val_freq', type=int, default=5,
                       help='Validation frequency (epochs)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint save frequency (epochs)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Adjust default parameters based on mode
    if args.mode == 'detector':
        if args.img_size == 160:
            args.img_size = 512  # Default for detector
        if args.epochs == 100:
            args.epochs = 20  # Paper uses 20 epochs for detector
        if args.batch_size == 32:
            args.batch_size = 8  # Detector needs smaller batch
    else:
        if args.img_size == 512:
            args.img_size = 160  # Default for recognizer
    
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    
    # Train
    if args.mode == 'detector':
        train_detector(args)
    else:
        train_recognizer(args)


if __name__ == '__main__':
    main()
