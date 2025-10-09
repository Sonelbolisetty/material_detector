import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QProgressBar, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
import cv2

# ======================
# Config
# ======================
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Material info database
MATERIAL_INFO = {
    "cardboard": {"components": "Paper fibers, adhesives, recycled pulp, wax coating", "recyclability": 90, "biodegradability": 85, "sustainability": 80},
    "glass": {"components": "Silica, soda, lime, alumina, metal oxides", "recyclability": 100, "biodegradability": 0, "sustainability": 95},
    "metal": {"components": "Iron, aluminum, copper, alloys, coatings", "recyclability": 98, "biodegradability": 0, "sustainability": 85},
    "paper": {"components": "Wood pulp, fillers, cellulose, ink, binders", "recyclability": 85, "biodegradability": 95, "sustainability": 90},
    "plastic": {"components": "Polyethylene, polypropylene, PVC, PET, pigments", "recyclability": 30, "biodegradability": 5, "sustainability": 40},
    "trash": {"components": "Mixed organic and inorganic waste", "recyclability": 10, "biodegradability": 20, "sustainability": 15},
}

# ======================
# CNN Model Definition
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

# ======================
# Load model
# ======================
def load_model(path):
    model = BetterCNN(num_classes=len(CLASS_NAMES)).to(device)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            print("✅ Model loaded successfully from", path)
            return model
        except Exception as e:
            print("⚠️ Error loading model:", e)
            return None
    else:
        print("⚠️ Model file not found at", path)
        return None

model = load_model("material_model.pth")

# ======================
# Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ======================
# GUI Class
# ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("♻️ Smart Material Detector — Eco Dashboard")
        self.setGeometry(200, 150, 1000, 650)

        # Main vertical layout
        app_layout = QVBoxLayout()

        # ===== Top Bar with Tech Mahindra Logo and ECO DETECTOR Title =====
        top_bar = QHBoxLayout()

        # Tech Mahindra Logo (Top Left)
        self.company_logo = QLabel()
        logo_path = "techmahindra.png"  # ensure this image exists in your folder
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaledToHeight(60, Qt.SmoothTransformation)
            self.company_logo.setPixmap(pixmap)
        self.company_logo.setAlignment(Qt.AlignLeft)
        self.company_logo.setStyleSheet("padding: 8px;")

        # ECO DETECTOR Title (Top Center)
        self.title_label = QLabel("♻️ MATERIAL DETECTOR")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #00e676; padding: 10px;")

        # Spacer for right side
        spacer = QLabel()
        spacer.setFixedWidth(80)

        top_bar.addWidget(self.company_logo)
        top_bar.addWidget(self.title_label, 1)
        top_bar.addWidget(spacer)
        app_layout.addLayout(top_bar)

        # ===== Main Content Layout =====
        main_layout = QHBoxLayout()

        # Left Panel (Image/Camera View)
        self.image_label = QLabel("Camera / Image Preview")
        self.image_label.setFixedSize(550, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa; border: 2px solid #444;")

        self.upload_button = QPushButton("📁 Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.camera_button = QPushButton("📷 Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.camera_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addLayout(button_layout)

        # Right Panel (Info)
        right_layout = QVBoxLayout()
        title = QLabel("🌿 Material Insights")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00e676; margin-bottom: 15px;")
        right_layout.addWidget(title)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setStyleSheet("background-color: #1e1e1e; color: #eeeeee; border-radius: 10px; padding: 8px;")
        self.info_box.setFont(QFont("Consolas", 11))

        # Progress Bars
        self.recyc_bar = self.create_progress_bar("♻️ Recyclability")
        self.bio_bar = self.create_progress_bar("🌱 Biodegradability")
        self.sust_bar = self.create_progress_bar("🌎 Sustainability")

        right_layout.addWidget(self.info_box)
        right_layout.addWidget(self.recyc_bar["label"])
        right_layout.addWidget(self.recyc_bar["bar"])
        right_layout.addWidget(self.bio_bar["label"])
        right_layout.addWidget(self.bio_bar["bar"])
        right_layout.addWidget(self.sust_bar["label"])
        right_layout.addWidget(self.sust_bar["bar"])

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)

        app_layout.addLayout(main_layout)

        # ===== Footer (Bottom Right Text) =====
        footer_label = QLabel("IOT LAB INITIATIVE")
        footer_label.setAlignment(Qt.AlignRight)
        footer_label.setStyleSheet("color: #999; font-size: 12px; padding: 5px 10px;")
        app_layout.addWidget(footer_label)

        # Set central widget
        container = QWidget()
        container.setLayout(app_layout)
        self.setCentralWidget(container)

        # Camera setup
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.camera_running = False

    # ====================
    # Helper: Create Progress Bar
    # ====================
    def create_progress_bar(self, label_text):
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 12))
        label.setStyleSheet("color: #00bfa5; margin-top: 10px;")

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setStyleSheet("""
            QProgressBar {
                background-color: #333;
                border-radius: 10px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00e676, stop:1 #1de9b6);
                border-radius: 10px;
            }
        """)
        return {"label": label, "bar": bar}

    # ====================
    # Upload Image
    # ====================
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
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

        self.update_info(predicted_class, confidence)

    # ====================
    # Camera Controls
    # ====================
    def toggle_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Unable to access camera.")
                return
            self.timer.start(30)
            self.camera_running = True
            self.camera_button.setText("⏹ Stop Camera")
        else:
            self.timer.stop()
            self.cap.release()
            self.camera_running = False
            self.camera_button.setText("📷 Start Camera")
            self.image_label.clear()
            self.info_box.clear()
            self.reset_bars()
            self.image_label.setText("Camera / Image Preview")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_val = confidence.item() * 100

        cv2.putText(frame, f"{predicted_class} ({confidence_val:.1f}%)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_display.shape
        qimg = QImage(rgb_display.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

        self.update_info(predicted_class, confidence_val)

    # ====================
    # Update Info
    # ====================
    def update_info(self, predicted_class, confidence):
        info = MATERIAL_INFO.get(predicted_class, {})
        comp = info.get("components", "Unknown")
        rec = info.get("recyclability", 0)
        bio = info.get("biodegradability", 0)
        sus = info.get("sustainability", 0)

        self.info_box.setHtml(f"""
            <b>Prediction:</b> {predicted_class.capitalize()} <br>
            <b>Confidence:</b> {confidence:.2f}%<br><br>
            <b>🧩 Components:</b> {comp}
        """)

        self.recyc_bar["bar"].setValue(rec)
        self.bio_bar["bar"].setValue(bio)
        self.sust_bar["bar"].setValue(sus)

    def reset_bars(self):
        self.recyc_bar["bar"].setValue(0)
        self.bio_bar["bar"].setValue(0)
        self.sust_bar["bar"].setValue(0)

# ======================
# Run App
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
