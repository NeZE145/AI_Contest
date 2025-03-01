import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('F:\Contest\my_model.keras')

# Image size should match the input shape of the model
image_size = (224, 224)

# Setup ImageDataGenerator for loading images from the test folder
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from the "test" folder, which contains 'winner' and 'loser' subfolders
test_generator = test_datagen.flow_from_directory(
    'Test_folder/',  # Path to the main test folder
    target_size=image_size,
    batch_size=32,
    class_mode='binary',  # Binary classification (winner vs loser)
    shuffle=False  # No shuffling for evaluation (preserves order)
)

# Evaluate the model on the test data
#test_loss, test_acc = model.evaluate(test_generator)
#print(f"Test accuracy: {test_acc}")
#print(f"Test loss: {test_loss}")

# Predict which image is more delicious (for an individual image)
def predict_image(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(img_array)
    print(prediction[0])
    return "Winner (More Delicious)" if prediction[0] > 0.5 else "Loser (Less Delicious)"

# Example usage
# Predict a specific image from the 'test' folder
print(predict_image(r'F:\Contest\Test_folder\winners_test\b1_1.jpg'))  # Replace with the actual image path
print(predict_image(r'F:\Contest\Test_folder\loser_test\b11_2.jpg'))  # Replace with the actual image path
