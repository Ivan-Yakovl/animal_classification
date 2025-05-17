# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QFileDialog, 
                            QMessageBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import tensorflow as tf
import cv2
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class SimpleImageClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Image Classifier")
        self.setFixedSize(600, 500)
        
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel("Result: Not determined")
        self.confidence_label = QLabel("Confidence: 0%")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        
        
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.predict)
        self.btn_predict.setEnabled(False)
        
     
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.progress_bar)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_predict)
        
        layout.addLayout(button_layout)
        central_widget.setLayout(layout)
        
      
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.model = tf.keras.models.load_model('models/best_model.keras')
            self.progress_bar.setValue(100)
            self.progress_bar.hide()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            sys.exit(1)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                
                h, w = image.shape[:2]
                scale = min(500/w, 400/h)
                display_img = cv2.resize(image, (int(w*scale), int(h*scale)))
                
                
                qimage = QImage(display_img.data, display_img.shape[1], 
                                display_img.shape[0], QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qimage))
                
                self.btn_predict.setEnabled(True)
                self.result_label.setText("Result: Not determined")
                self.confidence_label.setText("Confidence: 0%")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")
    
    def predict(self):
        if not hasattr(self, 'current_image_path'):
            return
            
        try:
            self.progress_bar.show()
            self.progress_bar.setValue(20)
            
           
            img = cv2.imread(self.current_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            self.progress_bar.setValue(50)
            
            
            pred = self.model.predict(img, verbose=0)[0]
            class_id = np.argmax(pred)
            confidence = float(pred[class_id])
            
            self.progress_bar.setValue(80)
            
            
            class_names = getattr(self.model, 'class_names', ['Class 0', 'Class 1', 'Class 2'])
            
            
            self.result_label.setText(f"Result: {class_names[class_id]}")
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")
            
            self.progress_bar.setValue(100)
            self.progress_bar.hide()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Classification error: {str(e)}")
            self.progress_bar.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    classifier = SimpleImageClassifier()
    classifier.show()
    sys.exit(app.exec_())