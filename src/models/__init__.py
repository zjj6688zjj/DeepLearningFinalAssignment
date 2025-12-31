"""
Logo Recognition Models Package
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)
"""

from .detector import (
    UniversalLogoDetector,
    SSDLogoDetector,
    create_detector,
    compute_detection_loss,
    compute_detection_metrics
)

from .recognizer import (
    SEResNet50,
    SpatialTransformerNetwork,
    SEBlock,
    ProxyTripletLoss,
    ProxyNCALoss,
    FewShotLogoRecognizer,
    create_optimizer,
    create_scheduler
)

__all__ = [
    # Detector
    'UniversalLogoDetector',
    'SSDLogoDetector',
    'create_detector',
    'compute_detection_loss',
    'compute_detection_metrics',
    # Recognizer
    'SEResNet50',
    'SpatialTransformerNetwork',
    'SEBlock',
    'ProxyTripletLoss',
    'ProxyNCALoss',
    'FewShotLogoRecognizer',
    'create_optimizer',
    'create_scheduler',
]
