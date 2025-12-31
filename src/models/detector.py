"""
Universal Logo Detector Module
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)

Uses Faster R-CNN with ResNet50 backbone as the universal logo detector.
The detector is class-agnostic: it only predicts logo vs background.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50, ResNet50_Weights


class UniversalLogoDetector(nn.Module):
    """
    Universal Logo Detector using Faster R-CNN.
    
    Class-agnostic detector that learns to identify "logoness" -
    the characteristics of what makes a logo, regardless of brand.
    
    Architecture as per paper:
    - Backbone: ResNet50 (pretrained on ImageNet)
    - Detection: Faster R-CNN
    - Output: Binary classification (logo vs background)
    - Input size: 512x512
    """
    
    def __init__(self, num_classes=2, pretrained_backbone=True,
                 min_size=512, max_size=512, trainable_backbone_layers=3):
        """
        Args:
            num_classes: 2 for binary (background + logo)
            pretrained_backbone: Use ImageNet pretrained weights
            min_size: Minimum input size
            max_size: Maximum input size
            trainable_backbone_layers: Number of trainable layers (0-5)
        """
        super().__init__()
        
        # Load pretrained Faster R-CNN with ResNet50 backbone
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT' if pretrained_backbone else None,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Replace the classifier head for binary classification
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Custom anchor generator for logo detection
        # Logos can vary greatly in size, so we use multiple scales
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        self.model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of tensors or batched tensor [B, C, H, W]
            targets: List of dicts with 'boxes' and 'labels' (training only)
        
        Returns:
            During training: dict of losses
            During inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        if isinstance(images, torch.Tensor):
            images = [img for img in images]
        
        return self.model(images, targets)
    
    def predict(self, images, score_threshold=0.5, nms_threshold=0.5):
        """
        Inference with post-processing
        
        Args:
            images: Input images
            score_threshold: Minimum confidence score
            nms_threshold: NMS IoU threshold
        
        Returns:
            List of dicts with filtered predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(images)
        
        filtered_predictions = []
        for pred in predictions:
            # Filter by score
            keep = pred['scores'] >= score_threshold
            
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions


class SSDLogoDetector(nn.Module):
    """
    Alternative detector using SSD (Single Shot MultiBox Detector).
    Faster but slightly lower accuracy on cross-domain data.
    """
    
    def __init__(self, num_classes=2, pretrained_backbone=True, size=512):
        super().__init__()
        
        self.model = torchvision.models.detection.ssd300_vgg16(
            weights='DEFAULT' if pretrained_backbone else None,
            num_classes=num_classes
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)


class YOLOv3LogoDetector(nn.Module):
    """
    YOLOv3-style detector placeholder.
    For full YOLOv3, consider using ultralytics or other implementations.
    
    Note: Paper found YOLOv3 had highest AP on same-domain (PL2K)
    but lower generalization to different domains.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Placeholder - recommend using ultralytics YOLOv3
        raise NotImplementedError(
            "For YOLOv3, please use: pip install ultralytics\n"
            "Then use: from ultralytics import YOLO"
        )


def create_detector(detector_type='faster_rcnn', **kwargs):
    """
    Factory function to create logo detector
    
    Args:
        detector_type: 'faster_rcnn', 'ssd', or 'yolov3'
        **kwargs: Arguments passed to detector constructor
    
    Returns:
        Logo detector model
    """
    detectors = {
        'faster_rcnn': UniversalLogoDetector,
        'ssd': SSDLogoDetector,
        'yolov3': YOLOv3LogoDetector
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    return detectors[detector_type](**kwargs)


def compute_detection_loss(model, images, targets):
    """
    Compute detection loss for training
    
    Args:
        model: Logo detector model
        images: Batch of images
        targets: List of target dicts
    
    Returns:
        Total loss scalar
    """
    model.train()
    loss_dict = model(images, targets)
    
    # Sum all losses
    total_loss = sum(loss for loss in loss_dict.values())
    
    return total_loss, loss_dict


def compute_detection_metrics(predictions, targets, iou_threshold=0.5):
    """
    Compute detection metrics (recall, precision, AP)
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dict of metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        gt_boxes = target['boxes']
        
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        # Compute IoU matrix
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth
        matched_gt = set()
        for i in range(len(pred_boxes)):
            max_iou, max_idx = iou_matrix[i].max(dim=0)
            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                total_tp += 1
                matched_gt.add(max_idx.item())
            else:
                total_fp += 1
        
        total_fn += len(gt_boxes) - len(matched_gt)
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] tensor
        boxes2: [M, 4] tensor
    
    Returns:
        [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - intersection
    
    return intersection / (union + 1e-6)
