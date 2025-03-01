import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Setup ImageDataGenerators for loading images
image_size = (224, 224)  # Resize images to (224, 224) pixels
batch_size = 32

# Load images from the "winner" and "loser" folders
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Specify the validation split (20% validation, 80% training)
)









train_generator = train_datagen.flow_from_directory(
    'Image_folder/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Binary classification (winner vs loser)
    shuffle=True,
    subset='training'  # Use the training subset
)

# Create a validation generator
validation_generator = train_datagen.flow_from_directory(
    'Image_folder/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    shuffle=True,
    subset='validation'  # Use the validation subset
)

# Check the number of images in each set
print(f"Training data: {len(train_generator.filenames)} images")
print(f"Validation data: {len(validation_generator.filenames)} images")

model = tf.keras.Sequential([
    # Convolutional layers
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    
    # Fully connected layers
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

model.save('my_model.keras')