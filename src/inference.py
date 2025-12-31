"""
Inference Pipeline for Logo Recognition System
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)

End-to-end pipeline:
1. Universal Logo Detector finds logo regions
2. Few-shot Logo Recognizer classifies each region
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import UniversalLogoDetector
from models.recognizer import FewShotLogoRecognizer


class LogoRecognitionPipeline:
    """
    End-to-end Logo Recognition Pipeline
    
    Combines:
    1. Universal Logo Detector (Faster R-CNN)
    2. Few-shot Logo Recognizer (SE-ResNet50 + STN + Proxy-Triplet)
    """
    
    def __init__(self, detector_path, recognizer_path, reference_dir,
                 device='cuda', score_threshold=0.5, k_neighbors=5):
        """
        Args:
            detector_path: Path to trained detector checkpoint
            recognizer_path: Path to trained recognizer checkpoint
            reference_dir: Directory with reference logo images (class subdirs)
            device: 'cuda' or 'cpu'
            score_threshold: Detection confidence threshold
            k_neighbors: K for K-NN classification
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.score_threshold = score_threshold
        self.k_neighbors = k_neighbors
        
        print(f"Loading models on {self.device}...")
        
        # Define transforms first (needed for reference embedding)
        self.detector_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.recognizer_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load detector
        self.detector = self._load_detector(detector_path)
        
        # Load recognizer
        self.recognizer, self.class_names = self._load_recognizer(
            recognizer_path, reference_dir
        )
        
        print("Pipeline ready!")
    
    def _load_detector(self, checkpoint_path):
        """Load trained detector"""
        detector = UniversalLogoDetector(num_classes=2).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            detector.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded detector from {checkpoint_path}")
        else:
            print("Warning: Using untrained detector")
        
        detector.eval()
        return detector
    
    def _load_recognizer(self, checkpoint_path, reference_dir):
        """Load trained recognizer and build reference embeddings"""
        # Get class names from reference directory
        class_names = sorted([d for d in os.listdir(reference_dir) 
                            if os.path.isdir(os.path.join(reference_dir, d))])
        num_classes = len(class_names)
        
        print(f"Found {num_classes} logo classes")
        
        # Load model
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            embedding_dim = checkpoint.get('embedding_dim', 128)
        else:
            embedding_dim = 128
        
        recognizer = FewShotLogoRecognizer(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            use_stn=True
        ).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            recognizer.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded recognizer from {checkpoint_path}")
        else:
            print("Warning: Using untrained recognizer")
        
        recognizer.eval()
        
        # Build reference embeddings
        self._build_reference_embeddings(recognizer, reference_dir, class_names)
        
        return recognizer, class_names
    
    def _build_reference_embeddings(self, recognizer, reference_dir, class_names):
        """Build reference embeddings from reference images"""
        print("Building reference embeddings...")
        
        all_embeddings = []
        all_labels = []
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for class_idx, class_name in enumerate(tqdm(class_names)):
            class_dir = os.path.join(reference_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            count = 0
            for img_name in os.listdir(class_dir):
                if count >= 10:  # Max 10 refs per class
                    break
                
                # 只处理图片文件
                ext = os.path.splitext(img_name)[1].lower()
                if ext not in img_extensions:
                    continue
                
                img_path = os.path.join(class_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.recognizer_transform(img).unsqueeze(0)
                    img_tensor = img_tensor.to(self.device)
                    
                    with torch.no_grad():
                        embedding = recognizer(img_tensor)
                    
                    all_embeddings.append(embedding.cpu())
                    all_labels.append(class_idx)
                    count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
                    continue
        
        if all_embeddings:
            ref_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
            ref_labels = torch.tensor(all_labels).to(self.device)
            
            recognizer.reference_embeddings = ref_embeddings
            recognizer.reference_labels = ref_labels
            
            print(f"Built {len(ref_labels)} reference embeddings")
        else:
            print("Warning: No reference embeddings built")
    
    def detect(self, image):
        """
        Detect logo regions in image
        
        Args:
            image: PIL Image or path
        
        Returns:
            List of (box, score) tuples
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        orig_w, orig_h = image.size
        
        # Transform
        img_tensor = self.detector_transform(image)
        img_tensor = img_tensor.to(self.device)
        
        # Detect
        with torch.no_grad():
            predictions = self.detector.predict(
                [img_tensor],
                score_threshold=self.score_threshold
            )[0]
        
        # Scale boxes back to original size
        scale_x = orig_w / 512
        scale_y = orig_h / 512
        
        results = []
        for i in range(len(predictions['boxes'])):
            box = predictions['boxes'][i].cpu().numpy()
            box = [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y
            ]
            score = predictions['scores'][i].item()
            results.append((box, score))
        
        return results
    
    def recognize(self, image, box):
        """
        Recognize logo in cropped region
        
        Args:
            image: PIL Image
            box: [x1, y1, x2, y2] bounding box
        
        Returns:
            (class_name, confidence)
        """
        # Crop region
        x1, y1, x2, y2 = [int(x) for x in box]
        crop = image.crop((x1, y1, x2, y2))
        
        # Transform
        crop_tensor = self.recognizer_transform(crop).unsqueeze(0)
        crop_tensor = crop_tensor.to(self.device)
        
        # Get embedding
        with torch.no_grad():
            query_embedding = self.recognizer(crop_tensor)
        
        # K-NN search
        ref_embeddings = self.recognizer.reference_embeddings
        ref_labels = self.recognizer.reference_labels
        
        # Compute distances
        dist = 2 - 2 * torch.mm(query_embedding, ref_embeddings.t())
        
        # Get k nearest neighbors
        distances, indices = dist.topk(self.k_neighbors, dim=1, largest=False)
        neighbor_labels = ref_labels[indices[0]]
        
        # Majority voting
        pred_label = torch.mode(neighbor_labels).values.item()
        
        # Confidence as 1 - normalized distance
        avg_dist = distances[0, neighbor_labels == pred_label].mean()
        confidence = max(0, 1 - avg_dist.item() / 2)
        
        class_name = self.class_names[pred_label]
        
        return class_name, confidence
    
    def predict(self, image):
        """
        Full pipeline: detect and recognize logos
        
        Args:
            image: PIL Image or path
        
        Returns:
            List of dict with 'box', 'class', 'detection_score', 'recognition_score'
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Step 1: Detect
        detections = self.detect(image)
        
        # Step 2: Recognize each detection
        results = []
        for box, det_score in detections:
            class_name, rec_score = self.recognize(image, box)
            
            results.append({
                'box': box,
                'class': class_name,
                'detection_score': det_score,
                'recognition_score': rec_score,
                'combined_score': det_score * rec_score
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def visualize(self, image, results, output_path=None):
        """
        Visualize detection and recognition results
        
        Args:
            image: PIL Image or path
            results: Output from predict()
            output_path: Save path (optional)
        
        Returns:
            Annotated PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        
        for i, result in enumerate(results):
            box = result['box']
            color = colors[i % len(colors)]
            
            # Draw box
            draw.rectangle(box, outline=color, width=3)
            
            # Draw label
            label = f"{result['class']} ({result['combined_score']:.2f})"
            text_bbox = draw.textbbox((box[0], box[1] - 20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((box[0], box[1] - 20), label, fill='white', font=font)
        
        if output_path:
            image.save(output_path)
            print(f"Saved result to {output_path}")
        
        return image


def main():
    parser = argparse.ArgumentParser(description='Logo Recognition Inference')
    
    # Model paths
    parser.add_argument('--detector', type=str, required=True,
                       help='Path to detector checkpoint')
    parser.add_argument('--recognizer', type=str, required=True,
                       help='Path to recognizer checkpoint')
    parser.add_argument('--reference_dir', type=str, required=True,
                       help='Directory with reference logo images')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory')
    
    # Parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--k', type=int, default=5,
                       help='K for K-NN classification')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize pipeline
    pipeline = LogoRecognitionPipeline(
        detector_path=args.detector,
        recognizer_path=args.recognizer,
        reference_dir=args.reference_dir,
        device=args.device,
        score_threshold=args.threshold,
        k_neighbors=args.k
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        images = [args.input]
    else:
        # Directory
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if os.path.splitext(f)[1].lower() in extensions]
    
    print(f"Processing {len(images)} images...")
    
    all_results = {}
    
    for img_path in tqdm(images):
        # Run prediction
        results = pipeline.predict(img_path)
        
        img_name = os.path.basename(img_path)
        all_results[img_name] = results
        
        # Print results
        print(f"\n{img_name}:")
        for r in results:
            print(f"  {r['class']}: det={r['detection_score']:.3f}, "
                  f"rec={r['recognition_score']:.3f}")
        
        # Visualize
        if args.visualize:
            output_path = os.path.join(args.output, f"vis_{img_name}")
            pipeline.visualize(img_path, results, output_path)
    
    # Save results as JSON (convert numpy types to native Python types)
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_path = os.path.join(args.output, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
