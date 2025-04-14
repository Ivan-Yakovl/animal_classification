import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


IMG_SIZE = (224, 224) 
BATCH_SIZE = 32        
EPOCHS = 15
LEARNING_RATE = 0.0001 
def create_datasets():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/train',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    ), tf.keras.utils.image_dataset_from_directory(
        'data/train',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Получаем имена классов из генератора
    class_names = tf.keras.utils.image_dataset_from_directory(
        'data/train'
    ).class_names
    
    # Аугментация и нормализация
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])
    
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True)/255., y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(
        lambda x, y: (x/255., y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names

def build_model(num_classes):
    
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )
    
   
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
def main():
    train_ds, val_ds, class_names = create_datasets()
    print(f"Classes: {class_names}")

    model = build_model(len(class_names))
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

   
    callback_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            'models/animal_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callback_list,  
        verbose=1
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/animal_model.h5')
    print("\nModel saved successfully!")
if __name__ == "__main__":
    main()