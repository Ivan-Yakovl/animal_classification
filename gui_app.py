import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import tensorflow as tf

class AnimalClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("������������� ��������")
        self.setGeometry(100, 100, 400, 400)
        
        # �������� ������
        self.model = tf.keras.models.load_model('models/animal_model.h5')
        self.classes = ['cat', 'dog', 'horse']  # �������� �� ���� ������
        
        # �������
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.result_label = QLabel("���������: ", self)
        self.load_button = QPushButton("��������� �����������", self)
        self.load_button.clicked.connect(self.load_image)
        
        # ��������
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.load_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def load_image(self):
        # �������� ������� ������ �����
        file_path, _ = QFileDialog.getOpenFileName(self, "�������� �����������", "", "Images (*.png *.jpg *.jpeg)")
        
        if file_path:
            # ����������� �����������
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300))
            
            # �������������
            predicted_class = self.predict_image(file_path)
            self.result_label.setText(f"���������: {predicted_class}")
    
    def predict_image(self, image_path):
        # ������������� �����������
        img = cv2.imread(image_path)
        if img is None:
            return "������ ��������"
        
        img = cv2.resize(img, (150, 150))  # ������ ������ ��������� � ���������
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # ������������
        pred = self.model.predict(img)
        return self.classes[np.argmax(pred)]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalClassifierApp()
    window.show()
    sys.exit(app.exec_())