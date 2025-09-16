"""
Material Detector GUI App
Detects material type: cardboard, metal, glass, paper, plastic, trash
Features:
- Detect image from file
- Live camera prediction
- Sustainability bars (Recyclability, Biodegradability, Eco-Friendliness)
"""

from PyQt5.QtGui import QPixmap
import sys
import os
import time
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QProgressBar, QSizePolicy, QFrame
)

# Classes
CLASS_NAMES = ['cardboard', 'metal', 'glass', 'paper', 'plastic', 'trash']


def heuristic_predict_pil(pil_img):
    """
    Simple heuristic classifier (fallback when no ML model is available).
    """
    img = np.array(pil_img.convert('RGB'))
    small = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    mean_rgb = small.mean(axis=(0, 1))
    mean_hsv = hsv.mean(axis=(0, 1))
    saturation = mean_hsv[1]
    value = mean_hsv[2]

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.std(lap)

    # Heuristic rules
    if value > 160 and texture < 10 and saturation < 70:
        return 'glass', 0.72
    if saturation < 50 and value > 120 and texture > 8:
        return 'metal', 0.75
    r, g, b = mean_rgb
    if (r > g > b) and (texture > 6 and texture < 30) and (saturation > 40):
        return 'cardboard', 0.7
    if value > 180 and texture < 6 and saturation < 60:
        return 'paper', 0.8
    if saturation > 90 and texture < 40:
        return 'plastic', 0.68
    return 'trash', 0.55


def sustainability_scores(material_label):
    """
    Map material to three scores (0–100).
    """
    mapping = {
        'cardboard': (90, 85, 80),
        'metal':    (95, 10, 75),
        'glass':    (92, 40, 78),
        'paper':    (88, 80, 82),
        'plastic':  (30, 5, 25),
        'trash':    (10, 2, 8)
    }
    return mapping.get(material_label, (20, 10, 15))


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, str, float)

    def __init__(self, predictor_func, cam_index=0):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(cam_index)
        self.predictor = predictor_func

    def run(self):
        while self._run_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            label, conf = self.predictor(pil)
            self.frame_ready.emit(rgb, label, conf)
            time.sleep(0.03)
        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait(2000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Material Detector")
        self.setGeometry(200, 100, 1100, 650)
        self._init_ui()
        self.video_thread = None

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # LEFT: Image/Video display
        self.display_label = QLabel("No image loaded")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setFixedSize(720, 540)
        self.display_label.setStyleSheet("background-color: #111; color: #ddd; border-radius: 6px;")

        # RIGHT: Controls
        self.pred_label = QLabel("Prediction: —")
        self.pred_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.conf_label = QLabel("Confidence: —")

        btn_load = QPushButton("Detect Image")
        btn_load.clicked.connect(self.load_image)

        self.btn_camera = QPushButton("Start Live Camera")
        self.btn_camera.setCheckable(True)
        self.btn_camera.clicked.connect(self.toggle_camera)

        # Sustainability bars
        self.bar_recycle = QProgressBar()
        self.bar_recycle.setMaximum(100)
        self.bar_recycle.setFormat("Recyclability: %p%")

        self.bar_biodeg = QProgressBar()
        self.bar_biodeg.setMaximum(100)
        self.bar_biodeg.setFormat("Biodegradability: %p%")

        self.bar_eco = QProgressBar()
        self.bar_eco.setMaximum(100)
        self.bar_eco.setFormat("Eco-Friendliness: %p%")

        # RIGHT layout
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.pred_label)
        right_layout.addWidget(self.conf_label)
        right_layout.addSpacing(8)
        right_layout.addWidget(btn_load)
        right_layout.addWidget(self.btn_camera)
        right_layout.addSpacing(16)
        right_layout.addWidget(QLabel("<b>Sustainability</b>"))
        right_layout.addWidget(self.bar_recycle)
        right_layout.addWidget(self.bar_biodeg)
        right_layout.addWidget(self.bar_eco)
        right_layout.addStretch(1)

        frame = QFrame()
        frame.setLayout(right_layout)
        frame.setFixedWidth(320)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.display_label)
        main_layout.addWidget(frame)
        central.setLayout(main_layout)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open image', os.getcwd(),
                                               "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not fname:
            return
        pil = Image.open(fname).convert('RGB')
        self.show_image(pil)
        label, conf = heuristic_predict_pil(pil)
        self.update_prediction(label, conf)

    def show_image(self, pil_img):
        qimg = pil_to_qimage(pil_img)
        pix = QPixmap.fromImage(qimg).scaled(
            self.display_label.width(), self.display_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(pix)

    def update_prediction(self, label, conf):
        self.pred_label.setText(f"Prediction: {label}")
        self.conf_label.setText(f"Confidence: {conf*100:.1f}%")
        r, b, e = sustainability_scores(label)
        self.bar_recycle.setValue(int(r))
        self.bar_biodeg.setValue(int(b))
        self.bar_eco.setValue(int(e))

    def toggle_camera(self, checked):
        if checked:
            self.btn_camera.setText("Stop Live Camera")
            self.video_thread = VideoThread(predictor_func=heuristic_predict_pil, cam_index=0)
            self.video_thread.frame_ready.connect(self.on_frame)
            self.video_thread.start()
        else:
            self.btn_camera.setText("Start Live Camera")
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None

    def on_frame(self, rgb_frame, label, conf):
        h, w, _ = rgb_frame.shape
        qimg = QImage(rgb_frame.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.display_label.width(), self.display_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(pix)
        self.update_prediction(label, conf)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()


def pil_to_qimage(pil_img):
    rgb = pil_img.convert('RGB')
    arr = np.array(rgb)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
