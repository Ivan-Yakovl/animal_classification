import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

IMG_SIZE = (150, 150)

def load_model():
    try:
        model = tf.keras.models.load_model('models/best_model.keras')
        
       
        assert len(model.layers) > 0, "Model has no layers!"
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def predict(image_path, model):
    try:
       
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image path or corrupted image")
        
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        
        img = img.astype('float32') / 255.0
        
      
        img = np.expand_dims(img, axis=0)
        
        
        pred = model.predict(img, verbose=0)[0]
        class_id = np.argmax(pred)
        confidence = float(pred[class_id])
        
       
        if hasattr(model, 'class_names'):
            class_names = model.class_names
        else:
            class_names = ['cat', 'dog', 'horse']
            
        return class_names[class_id], confidence
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  
        return str(e), 0

def main():
    model = load_model()
    
    
    print("\nPerforming sanity check...")
    test_img = np.random.randint(0, 256, (*IMG_SIZE, 3), dtype=np.uint8)
    cv2.imwrite('test.jpg', test_img)
    result, conf = predict('test.jpg', model)
    print(f"Sanity check result: {result} ({conf:.2%})")
    os.remove('test.jpg')
    
    while True:
        print("\n1. Test image")
        print("2. Exit")
        choice = input("Select option: ").strip()
        
        if choice == '1':
            path = input("Image path: ").strip('"')
            if not os.path.exists(path):
                print("Error: File not found")
                continue
                
            result, conf = predict(path, model)
            print(f"\nResult: {result} (confidence: {conf:.2%})")
            
        elif choice == '2':
            break
            
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()