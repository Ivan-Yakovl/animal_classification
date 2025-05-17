# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QFileDialog, 
                            QMessageBox, QLineEdit, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import tensorflow as tf
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class AnimalClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Classifier (Cats, Dogs, Horses)")
        self.setFixedSize(900, 800)  # Увеличенный размер окна
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Image Display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px solid gray;
            background-color: #f0f0f0;
            min-height: 500px;
            max-height: 500px;
        """)
        layout.addWidget(self.image_label, stretch=1)
        
        # Path Input
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter image path or click Browse...")
        self.path_input.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.path_input)
        
        # Browse Button
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setStyleSheet("font-size: 14px; padding: 5px;")
        self.browse_btn.clicked.connect(self.browse_image)
        layout.addWidget(self.browse_btn)
        
        # Predict Button
        self.predict_btn = QPushButton("Classify Animal")
        self.predict_btn.setStyleSheet("""
            font-size: 16px;
            padding: 8px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        """)
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)
        
        # Results
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            font-size: 14px;
            min-height: 150px;
            border: 1px solid #ccc;
            padding: 10px;
        """)
        layout.addWidget(self.result_text)
        
        # Load Model
        model_path = 'models/best_model.keras'
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Model file not found at: {model_path}")
            sys.exit(1)
            
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['cat', 'dog', 'horse']
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.path_input.setText(file_path)
            self.display_image(file_path)
            self.predict_btn.setEnabled(True)
    
    def display_image(self, file_path):
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Invalid image file")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Увеличиваем максимальный размер отображаемого изображения
            max_display_size = 700  # Максимальный размер по большей стороне
            scale = max_display_size / max(h, w)
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            img = cv2.resize(img, (new_width, new_height))
            
            qimg = QImage(
                img.data, 
                img.shape[1], 
                img.shape[0], 
                img.shape[1] * 3,  # bytesPerLine
                QImage.Format_RGB888
            )
            self.image_label.setPixmap(QPixmap.fromImage(qimg))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load image: {e}")
            self.image_label.setText("Could not display image")
    
    def predict(self):
        file_path = self.path_input.text()
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "File does not exist!")
            return
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Invalid image file")
                
            img = cv2.resize(img, (150, 150))  # Размер для модели
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            pred = self.model.predict(img, verbose=0)[0]
            class_id = np.argmax(pred)
            confidence = pred[class_id]
            
            result = f"<h3>Classification Results:</h3>"
            result += f"<p><b>Predicted:</b> {self.classes[class_id].upper()}</p>"
            result += f"<p><b>Confidence:</b> {confidence:.2%}</p>"
            result += "<h4>All probabilities:</h4>"
            
            for i, prob in enumerate(pred):
                bar_width = int(prob * 200)
                result += (
                    f"<p>{self.classes[i].ljust(6)}: "
                    f"<span style='color: {'#4CAF50' if i == class_id else '#333'};'>"
                    f"{prob:.2%}</span> "
                    f"<span style='background-color: #ddd; display: inline-block; "
                    f"width: {bar_width}px; height: 15px;'></span></p>"
                )
            
            self.result_text.setHtml(result)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    # Настройка шрифта
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    
    window = AnimalClassifier()
    window.show()
    sys.exit(app.exec_())