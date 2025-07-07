import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import psycopg
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Constants
BASE_DIR = "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE"
MODEL_DIR = os.path.join(BASE_DIR, "models")
PPE_ITEMS = [
    "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest",
    "Gloves", "NO-Gloves", "Goggles", "NO-Goggles",
    "Mask", "NO-Mask", "Fall-Detected", "Person"
]
PPE_COLORS = {
    "Hardhat": (0, 0, 255), "NO-Hardhat": (0, 128, 255),
    "Safety Vest": (0, 255, 0), "NO-Safety Vest": (0, 128, 128),
    "Gloves": (255, 0, 0), "NO-Gloves": (128, 0, 128),
    "Goggles": (255, 255, 0), "NO-Goggles": (128, 128, 0),
    "Mask": (0, 255, 255), "NO-Mask": (0, 128, 128),
    "Fall-Detected": (255, 0, 255), "Person": (128, 0, 255)
}
VALID_DEPTS = ['Admin', 'Maintenance']
MAX_FRAMES_TO_PROCESS = 100
DETECTION_CONFIDENCE_THRESHOLD = 0.5
DB_PARAMS = {
    "dbname": "ppe_detection",
    "user": "postgres",
    "password": "Calcite*1234",
    "host": "localhost",
    "port": "5432"
}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CUDATransform:
    def __init__(self, standard_transform, use_edge=True):
        self.standard_transform = standard_transform
        self.use_edge = use_edge

    def __call__(self, img):
        if self.use_edge:
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges = edges / 255.0
            img = Image.fromarray(img_np)
            tensor = self.standard_transform(img)
            edge_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0)
            edge_tensor = F.interpolate(edge_tensor.unsqueeze(0), size=tensor.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
            tensor = torch.cat([tensor, edge_tensor], dim=0)
        else:
            tensor = self.standard_transform(img)
        return tensor

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * spatial_out
        return x

# STN Module
class STN(nn.Module):
    def __init__(self, in_channels):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 56 * 56, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 56 * 56)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

# Model classes
class EnhancedCustomCNN(nn.Module):
    def __init__(self, num_classes=len(PPE_ITEMS)):
        super(EnhancedCustomCNN, self).__init__()
        self.stn = STN(in_channels=4)
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            CBAM(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            CBAM(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3)
        )
        self.feature_size = 256 * (224 // 8) * (224 // 8)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits, x

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=len(PPE_ITEMS)):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_size = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.attention = CBAM(self.feature_size)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.feature_size, num_classes))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.attention(x)
        x = self.resnet.avgpool(x)
        features = x.view(x.size(0), -1)
        logits = self.classifier(features)
        return logits, features

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=len(PPE_ITEMS)):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.feature_size = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        self.attention = CBAM(self.feature_size)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(self.feature_size, num_classes))

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.attention(x)
        x = self.efficientnet.avgpool(x)
        features = x.view(x.size(0), -1)
        logits = self.classifier(features)
        return logits, features

# Load model
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_best.pt")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise Exception(f"Model file not found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        num_classes = len(PPE_ITEMS)

        if model_name == "CustomCNN":
            model = EnhancedCustomCNN(num_classes=num_classes).to(device)
            new_state_dict = {}
            for key, value in state_dict.items():
                target_shape = model.state_dict().get(key, None)
                if target_shape is None:
                    logger.warning(f"Skipping key {key}: not found in model")
                    continue
                try:
                    if key == "features.0.weight" and value.size(1) == 3:
                        # Adapt 3-channel input to 4 channels
                        logger.info(f"Adapting {key}: {value.shape} to {target_shape}")
                        edge_channel = value.mean(dim=1, keepdim=True)
                        new_weight = torch.cat([value, edge_channel], dim=1)
                        new_state_dict[key] = new_weight
                    elif key in [
                        "features.0.weight", "features.3.weight", "features.9.weight",
                        "features.12.weight", "features.19.weight"
                    ]:
                        # Pad or truncate convolutional weights
                        logger.info(f"Adapting {key}: {value.shape} to {target_shape}")
                        new_weight = torch.zeros_like(target_shape)
                        min_filters = min(value.shape[0], target_shape[0])
                        min_in_channels = min(value.shape[1], target_shape[1])
                        min_height = min(value.shape[2], target_shape[2])
                        min_width = min(value.shape[3], target_shape[3])
                        new_weight[:min_filters, :min_in_channels, :min_height, :min_width] = value[:min_filters, :min_in_channels, :min_height, :min_width]
                        new_state_dict[key] = new_weight
                    elif key in [
                        "features.0.bias", "features.1.weight", "features.1.bias",
                        "features.1.running_mean", "features.1.running_var",
                        "features.3.bias", "features.4.weight", "features.4.bias",
                        "features.4.running_mean", "features.4.running_var",
                        "features.9.bias", "features.12.bias", "features.19.bias"
                    ]:
                        # Pad or truncate 1D parameters
                        logger.info(f"Adapting {key}: {value.shape} to {target_shape}")
                        new_param = torch.zeros_like(target_shape)
                        min_size = min(value.shape[0], target_shape[0])
                        new_param[:min_size] = value[:min_size]
                        new_state_dict[key] = new_param
                    elif key == "classifier.0.weight":
                        # Adapt classifier input size
                        logger.info(f"Adapting {key}: {value.shape} to {target_shape}")
                        new_weight = torch.zeros_like(target_shape)
                        min_out = min(value.shape[0], target_shape[0])
                        min_in = min(value.shape[1], target_shape[1])
                        new_weight[:min_out, :min_in] = value[:min_out, :min_in]
                        new_state_dict[key] = new_weight
                    elif key == "classifier.3.weight":
                        # Adapt classifier output to num_classes
                        logger.info(f"Adapting {key}: {value.shape} to {target_shape}")
                        if value.shape[0] != target_shape[0]:  # If output classes don't match
                            logger.warning(f"Output classes mismatch: checkpoint has {value.shape[0]}, model expects {target_shape[0]}. Initializing new classifier.")
                            new_state_dict[key] = model.state_dict()[key]  # Use model's initialized weights
                        else:
                            new_weight = torch.zeros_like(target_shape)
                            min_out = min(value.shape[0], target_shape[0])
                            min_in = min(value.shape[1], target_shape[1])
                            new_weight[:min_out, :min_in] = value[:min_out, :min_in]
                            new_state_dict[key] = new_weight
                    elif key == "classifier.3.bias":
                        # Adapt classifier bias
                        logger.info(f"Adapting {key}: {value.shape} to {target_shape}")
                        if value.shape[0] != target_shape[0]:  # If output classes don't match
                            logger.warning(f"Bias classes mismatch: checkpoint has {value.shape[0]}, model expects {target_shape[0]}. Initializing new bias.")
                            new_state_dict[key] = model.state_dict()[key]  # Use model's initialized bias
                        else:
                            new_param = torch.zeros_like(target_shape)
                            min_size = min(value.shape[0], target_shape[0])
                            new_param[:min_size] = value[:min_size]
                            new_state_dict[key] = new_param
                    else:
                        new_state_dict[key] = value
                except Exception as e:
                    logger.warning(f"Failed to adapt {key}: {e}")
                    new_state_dict[key] = value
            state_dict = new_state_dict
        elif model_name == "ResNet18":
            model = ModifiedResNet(num_classes=num_classes).to(device)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key == "resnet.conv1.weight" and value.size(1) == 3:
                    edge_channel = value.mean(dim=1, keepdim=True)
                    new_weight = torch.cat([value, edge_channel], dim=1)
                    new_state_dict[key] = new_weight
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        elif model_name == "EfficientNet":
            model = EfficientNetModel(num_classes=num_classes).to(device)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key == "efficientnet.features.0.0.weight" and value.size(1) == 3:
                    edge_channel = value.mean(dim=1, keepdim=True)
                    new_weight = torch.cat([value, edge_channel], dim=1)
                    new_state_dict[key] = new_weight
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info(f"{model_name} model loaded successfully from {model_path} with {num_classes} classes")
        return model, num_classes
    except Exception as e:
        logger.error(f"Error loading {model_name} from {model_path}: {e}")
        raise Exception(f"Failed to load model: {str(e)}")

# Database functions
def connect_db():
    try:
        conn = psycopg.connect(**DB_PARAMS)
        logger.info("Database connected")
        return conn
    except psycopg.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise Exception("Database connection failed")

def init_database(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id VARCHAR(50) PRIMARY KEY,
                    department VARCHAR(50) NOT NULL
                )
            """)
            logger.info("Employees table created or already exists")
            conn.commit()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id SERIAL PRIMARY KEY,
                    employee_id VARCHAR(50) REFERENCES employees(employee_id),
                    duration_seconds FLOAT NOT NULL,
                    status VARCHAR(20),
                    target_class VARCHAR(50) NOT NULL,
                    session_date DATE NOT NULL,
                    timestamp TIME NOT NULL
                )
            """)
            logger.info("Sessions table created or already exists")
            conn.commit()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS ppe_details (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES sessions(id),
                    ppe_item VARCHAR(20) NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    percentage FLOAT NOT NULL DEFAULT 0.0,
                    flagged BOOLEAN NOT NULL DEFAULT FALSE
                )
            """)
            logger.info("ppe_details table created or already exists")
            conn.commit()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS ppe_violations (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES sessions(id),
                    ppe_item VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("ppe_violations table created or already exists")
            conn.commit()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(50),
                    epoch INT,
                    train_loss FLOAT,
                    train_accuracy FLOAT,
                    train_f1 FLOAT,
                    val_loss FLOAT,
                    val_accuracy FLOAT,
                    val_f1 FLOAT,
                    class_accuracies JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("training_logs table created or already exists")
            conn.commit()

        logger.info("Database tables initialized successfully")
    except psycopg.Error as e:
        logger.error(f"Database initialization error: {e}")
        conn.rollback()
        raise Exception(f"Database initialization failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        conn.rollback()
        raise Exception(f"Unexpected error: {str(e)}")

# PPE History class
class PPEHistory:
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.ppe_history = []

    def add_ppe(self, detected_items, confidence):
        self.ppe_history.append((detected_items, confidence))
        if len(self.ppe_history) > self.history_size:
            self.ppe_history.pop(0)

    def get_status(self, target_class, num_classes=1):
        if not self.ppe_history:
            return None, 0.0, []
        accepted_count = 0
        flagged_items = []
        for detected_items, confidence in self.ppe_history:
            if target_class in detected_items:
                accepted_count += 1
            else:
                flagged_items.append(target_class)
        total = len(self.ppe_history)
        acceptance_rate = (accepted_count / total) * 100 if total > 0 else 0.0
        return "Accepted" if acceptance_rate >= 50 else "Flagged", acceptance_rate, list(set(flagged_items))

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_rois = []
    face_coords = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_rois.append(face_img)
        face_coords.append((x, y, w, h))
    return face_rois, face_coords