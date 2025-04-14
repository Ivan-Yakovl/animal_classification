import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

CLASSES = ['cat', 'dog', 'horse']

def load_model():
    try:
        model = tf.keras.models.load_model('models/animal_model.h5')
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def predict(image_path, model):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Failed to read image", 0
        
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        pred = model.predict(img)
        class_id = np.argmax(pred)
        confidence = float(np.max(pred))
        
        return CLASSES[class_id], confidence
    except Exception as e:
        return f"Error: {str(e)}", 0

def main():
    model = load_model()
    
    while True:
        print("\n1. Test image")
        print("2. Exit")
        choice = input("Select option: ")
        
        if choice == '1':
            path = input("Image path: ").strip('"')
            result, conf = predict(path, model)
            print(f"Result: {result} ({conf:.0%} confidence)")
        elif choice == '2':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()