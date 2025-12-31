"""
Logo Recognition System
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)

A two-stage logo recognition pipeline:
1. Universal Logo Detector - Class-agnostic Faster R-CNN
2. Few-shot Logo Recognizer - SE-ResNet50 with Proxy-Triplet Loss

Paper: https://arxiv.org/abs/1811.08009
"""

__version__ = '1.0.0'
__author__ = 'Based on Fehérvári & Appalaraju (Amazon)'

from .models import (
    UniversalLogoDetector,
    FewShotLogoRecognizer,
    create_detector
)

from .dataset import (
    LogoDetectionDataset,
    LogoRecognitionDataset,
    TripletLogoDataset,
    get_detection_dataloader,
    get_recognition_dataloader
)

from .inference import LogoRecognitionPipeline
