import sys
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

# ======================
# Config
# ======================
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sustainability Info
SUSTAINABILITY = {
    "cardboard": {"Recyclability": 90, "Biodegradability": 95, "Eco-Friendliness": 85,
                  "Components": "Corrugated sheets, paper fibers"},
    "glass": {"Recyclability": 99, "Biodegradability": 0, "Eco-Friendliness": 80,
              "Components": "Silica, soda ash, limestone"},
    "metal": {"Recyclability": 95, "Biodegradability": 0, "Eco-Friendliness": 75,
              "Components": "Aluminum, steel, alloys"},
    "paper": {"Recyclability": 85, "Biodegradability": 90, "Eco-Friendliness": 80,
              "Components": "Wood pulp, fibers, cellulose"},
    "plastic": {"Recyclability": 30, "Biodegradability": 10, "Eco-Friendliness": 25,
                "Components": "PET, HDPE, polymers"},
    "trash": {"Recyclability": 10, "Biodegradability": 20, "Eco-Friendliness": 15,
              "Components": "Mixed waste, food scraps, non-recyclables"}
}

# ======================
# Model
# ======================
class BetterCNN(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super(BetterCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model(path="material_model.pth"):
    model = BetterCNN(num_classes=len(CLASS_NAMES)).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully")
        return model
    else:
        print("⚠️ Model file not found.")
        return None

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ======================
# GUI
# ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("♻️ Material Detector")
        self.setGeometry(200, 200, 600, 600)

        self.layout = QVBoxLayout()

        self.label = QLabel("Upload an image or start live detection", self)
        self.layout.addWidget(self.label)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 300)
        self.layout.addWidget(self.image_label)

        # Buttons
        self.upload_button = QPushButton("📂 Detect Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        self.live_button = QPushButton("🎥 Start Live Camera", self)
        self.live_button.clicked.connect(self.toggle_camera)
        self.layout.addWidget(self.live_button)

        # Sustainability bars
        self.recycle_bar = QProgressBar(self); self.layout.addWidget(self.recycle_bar)
        self.bio_bar = QProgressBar(self); self.layout.addWidget(self.bio_bar)
        self.eco_bar = QProgressBar(self); self.layout.addWidget(self.eco_bar)

        # Component info
        self.component_label = QLabel("Components: -", self)
        self.layout.addWidget(self.component_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Timer for live camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height()
            ))
            self.predict_image(file_name)

    def predict_image(self, image_path):
        if model is None:
            QMessageBox.warning(self, "Error", "Model not loaded.")
            return

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence = confidence.item() * 100

        self.label.setText(f"Prediction: {predicted_class} ({confidence:.1f}%)")
        self.update_sustainability(predicted_class)

    def update_sustainability(self, material):
        info = SUSTAINABILITY.get(material, {})
        self.recycle_bar.setValue(info.get("Recyclability", 0))
        self.recycle_bar.setFormat(f"Recyclability: {info.get('Recyclability', 0)}%")

        self.bio_bar.setValue(info.get("Biodegradability", 0))
        self.bio_bar.setFormat(f"Biodegradability: {info.get('Biodegradability', 0)}%")

        self.eco_bar.setValue(info.get("Eco-Friendliness", 0))
        self.eco_bar.setFormat(f"Eco-Friendliness: {info.get('Eco-Friendliness', 0)}%")

        self.component_label.setText(f"Components: {info.get('Components', '-')}")


    # ======================
    # Live Camera
    # ======================
    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.live_button.setText("⏹ Stop Live Camera")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.live_button.setText("🎥 Start Live Camera")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.width(), self.image_label.height()
        ))

        # Prediction
        image = Image.fromarray(rgb_image).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence = confidence.item() * 100

        self.label.setText(f"Prediction: {predicted_class} ({confidence:.1f}%)")
        self.update_sustainability(predicted_class)


# ======================
# Run App
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
