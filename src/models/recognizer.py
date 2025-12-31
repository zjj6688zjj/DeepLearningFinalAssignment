"""
Few-shot Logo Recognizer Module
Based on: "Scalable Logo Recognition using Proxies" (IEEE WACV 2019)

Core components:
1. SE-ResNet50 backbone (Squeeze-and-Excitation networks)
2. Spatial Transformer Network (STN) layer
3. Proxy-based Triplet Loss

Architecture (from paper):
- Input: 160x160 pixels
- Embedding dimension: 128
- Batch size: 32
- Optimizer: Adam (momentum=0.9, weight_decay=0.0005, lr=1e-4)
- LR schedule: decay by 0.8 every 20 epochs
- Initialization: Xavier with magnitude 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Squeeze-and-Excitation Block
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Reference: "Squeeze-and-Excitation Networks" (Hu et al., 2017)
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    """
    SE-ResNet Bottleneck block with Squeeze-and-Excitation.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, 
                 downsample=None, reduction=16):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # SE attention
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


# ============================================================================
# Spatial Transformer Network (STN)
# ============================================================================

class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network for learning logo orientation.
    Reference: "Spatial Transformer Networks" (Jaderberg et al., 2015)
    
    Helps the model handle rotated, scaled, or skewed logos.
    """
    
    def __init__(self, in_channels=3, img_size=160):
        super().__init__()
        
        self.img_size = img_size
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Regressor for affine transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 6 parameters for 2D affine transformation
        )
        
        # Initialize to identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x):
        # Get transformation parameters
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Apply affine transformation
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        
        return x


# ============================================================================
# SE-ResNet50 with STN
# ============================================================================

class SEResNet50(nn.Module):
    """
    SE-ResNet50 backbone for logo embedding.
    
    Architecture as per paper:
    - SE-ResNet50 with modifications from ArcFace paper
    - Output embedding dimension: 128
    """
    
    def __init__(self, embedding_dim=128, use_stn=True, pretrained=False):
        super().__init__()
        
        self.use_stn = use_stn
        self.embedding_dim = embedding_dim
        
        # Optional STN layer
        if use_stn:
            self.stn = SpatialTransformerNetwork(in_channels=3, img_size=160)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # SE-ResNet layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Global average pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, embedding_dim)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)
        
        # Initialize weights (Xavier with magnitude 2)
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * SEBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * SEBottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * SEBottleneck.expansion),
            )
        
        layers = []
        layers.append(SEBottleneck(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(SEBottleneck(
                out_channels * SEBottleneck.expansion, 
                out_channels
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Xavier initialization with magnitude 2"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        # Apply STN
        if self.use_stn:
            x = self.stn(x)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and embedding
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_embed(x)
        
        # L2 normalize embedding
        x = F.normalize(x, p=2, dim=1)
        
        return x


# ============================================================================
# Proxy-based Triplet Loss
# ============================================================================

class ProxyTripletLoss(nn.Module):
    """
    Proxy-based Triplet Loss for few-shot logo recognition.
    
    Instead of sampling triplets from training images (expensive),
    we learn proxies (one per class) and compute triplet loss using proxies.
    
    Reference: "No Fuss Distance Metric Learning Using Proxies" 
               (Movshovitz-Attias et al., 2017)
    
    Loss function (Equation 4 from paper):
        L_triplet(x, y, Z) = [d(x, p(y)) + M - d(x, p(Z))]+
    
    where:
        - x: anchor embedding
        - p(y): proxy for positive class
        - p(Z): proxy for negative class
        - M: margin (typically 0.1 - 0.5)
    """
    
    def __init__(self, num_classes, embedding_dim=128, margin=0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Learnable proxies - one per class
        # Initialized with same norm as embeddings
        self.proxies = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.proxies)
        
        # Normalize proxies
        with torch.no_grad():
            self.proxies.data = F.normalize(self.proxies.data, p=2, dim=1)
    
    def forward(self, embeddings, labels):
        """
        Compute proxy-based triplet loss
        
        Args:
            embeddings: [B, embedding_dim] normalized embeddings
            labels: [B] class labels
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize proxies
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        batch_size = embeddings.size(0)
        
        # Compute distances to all proxies
        # Using squared Euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # For normalized vectors: ||a-b||^2 = 2 - 2*a.b
        dist_matrix = 2 - 2 * torch.mm(embeddings, proxies.t())  # [B, num_classes]
        
        # Get positive distances (distance to proxy of same class)
        pos_distances = dist_matrix[torch.arange(batch_size), labels]  # [B]
        
        # For each sample, compute triplet loss with hard negative
        losses = []
        
        for i in range(batch_size):
            pos_dist = pos_distances[i]
            
            # Get distances to all negative proxies
            neg_mask = torch.ones(self.num_classes, dtype=torch.bool, 
                                  device=embeddings.device)
            neg_mask[labels[i]] = False
            neg_distances = dist_matrix[i, neg_mask]
            
            # Hard negative mining: use closest negative proxy
            min_neg_dist = neg_distances.min()
            
            # Triplet loss with margin
            loss = F.relu(pos_dist + self.margin - min_neg_dist)
            losses.append(loss)
        
        return torch.stack(losses).mean()


class ProxyNCALoss(nn.Module):
    """
    Proxy-NCA Loss (alternative, tends to overfit earlier).
    
    Loss function (Equation 3 from paper):
        L_NCA(x, y, Z) = -log(exp(-d(x,p(y))) / sum_z exp(-d(x,p(z))))
    """
    
    def __init__(self, num_classes, embedding_dim=128, scale=3.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        
        self.proxies = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.proxies)
    
    def forward(self, embeddings, labels):
        proxies = F.normalize(self.proxies, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarities
        similarities = self.scale * torch.mm(embeddings, proxies.t())
        
        # Cross-entropy loss with proxy logits
        loss = F.cross_entropy(similarities, labels)
        
        return loss


# ============================================================================
# Complete Few-shot Logo Recognizer
# ============================================================================

class FewShotLogoRecognizer(nn.Module):
    """
    Complete Few-shot Logo Recognizer combining:
    1. SE-ResNet50 backbone
    2. Optional STN layer
    3. Proxy-based training
    
    At inference, use K-nearest neighbor search on embeddings.
    """
    
    def __init__(self, num_classes, embedding_dim=128, use_stn=True, 
                 margin=0.2, loss_type='proxy_triplet'):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Backbone
        self.backbone = SEResNet50(
            embedding_dim=embedding_dim,
            use_stn=use_stn,
            pretrained=False
        )
        
        # Loss function with proxies
        if loss_type == 'proxy_triplet':
            self.loss_fn = ProxyTripletLoss(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                margin=margin
            )
        elif loss_type == 'proxy_nca':
            self.loss_fn = ProxyNCALoss(
                num_classes=num_classes,
                embedding_dim=embedding_dim
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Reference embeddings for inference (populated during training/registration)
        self.register_buffer('reference_embeddings', None)
        self.register_buffer('reference_labels', None)
    
    def forward(self, x):
        """Extract embeddings"""
        return self.backbone(x)
    
    def compute_loss(self, embeddings, labels):
        """Compute training loss"""
        return self.loss_fn(embeddings, labels)
    
    def register_references(self, images, labels):
        """
        Register reference embeddings for K-NN inference
        
        Args:
            images: Reference images [N, C, H, W]
            labels: Reference labels [N]
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(images)
        
        self.reference_embeddings = embeddings
        self.reference_labels = labels
    
    def predict(self, images, k=1):
        """
        Predict logo class using K-NN search
        
        Args:
            images: Query images [B, C, H, W]
            k: Number of nearest neighbors
        
        Returns:
            predictions: Predicted labels [B]
            distances: Distances to nearest neighbors [B, k]
        """
        if self.reference_embeddings is None:
            raise RuntimeError("No reference embeddings registered. "
                             "Call register_references() first.")
        
        self.eval()
        with torch.no_grad():
            query_embeddings = self.forward(images)
        
        # Compute distances to all references
        # For normalized vectors: d^2 = 2 - 2*cos_sim
        dist_matrix = 2 - 2 * torch.mm(query_embeddings, 
                                        self.reference_embeddings.t())
        
        # Get k nearest neighbors
        distances, indices = dist_matrix.topk(k, dim=1, largest=False)
        
        # Get labels of nearest neighbors
        neighbor_labels = self.reference_labels[indices]
        
        # Vote for final prediction
        if k == 1:
            predictions = neighbor_labels.squeeze(1)
        else:
            # Majority voting
            predictions = []
            for i in range(len(images)):
                labels_k = neighbor_labels[i]
                pred = torch.mode(labels_k).values
                predictions.append(pred)
            predictions = torch.stack(predictions)
        
        return predictions, distances


def create_optimizer(model, lr=1e-4, weight_decay=5e-4, momentum=0.9):
    """
    Create optimizer as per paper specifications
    
    Args:
        model: FewShotLogoRecognizer model
        lr: Learning rate (1e-4)
        weight_decay: Weight decay (5e-4)
        momentum: Momentum (0.9)
    
    Returns:
        optimizer: Adam optimizer
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(momentum, 0.999)
    )


def create_scheduler(optimizer, step_size=20, gamma=0.8):
    """
    Create LR scheduler as per paper
    
    Reduce LR by factor of 0.8 every 20 epochs
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
