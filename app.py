from flask import Flask, request, jsonify, render_template
import os
import cv2 as cv
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/animal_model.h5')
classes = ['cat', 'dog', 'horse']  # Замените на свои классы

# Создаем папку для загрузок, если её нет
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})
    
    # Сохраняем файл
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Обработка изображения
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({'error': 'Invalid image'})
    
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Предсказание
    pred = model.predict(img)
    result = classes[np.argmax(pred)]
    
    # Удаляем файл после обработки
    os.remove(file_path)
    
    return jsonify({'class': result})

if __name__ == '__main__':
    app.run(debug=True)