import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def load_model(model_path='models/cifar10_model.h5'):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    """Preprocess uploaded image for prediction."""
    image = image.resize((32, 32))  # Resize to CIFAR-10 dimensions
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 32 * 32 * 3)  # Flatten
    return image

def main():
    st.title("CIFAR-10 Image Classifier")
    st.write("Upload an image to classify it into one of 10 categories.")
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction, axis=1)[0]
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        confidence = prediction[0][class_idx]
        
        st.write(f"Prediction: **{class_names[class_idx]}**")
        st.write(f"Confidence: **{confidence:.4f}**")

if __name__ == '__main__':
    main()