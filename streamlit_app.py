import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🫁 Pneumonia Detection System</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Chest X-Ray Analysis using DenseNet121")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pneumonia_multiclass_densenet_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block16_concat', pred_index=None):
    """Generate Grad-CAM heatmap"""
    try:
        # Find the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Ensure predictions is a tensor
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the predicted class channel
            class_channel = predictions[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        # Return a blank heatmap as fallback
        return np.zeros((7, 7))

def create_gradcam_overlay(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image"""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Convert image to BGR if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_colored

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize
    img_resized = cv2.resize(image, (224, 224))
    
    # Convert to RGB if grayscale
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    return img_array, img_resized

# Main app
def main():
    # Sidebar
    st.sidebar.title("📋 About")
    st.sidebar.info(
        """
        **Model Details:**
        - Architecture: DenseNet121
        - Dataset: Chest X-Ray Images
        - Classes: Normal, Pneumonia
        - Accuracy: 89.42%
        - Sensitivity: 98.21%
        - Specificity: 74.79%
        
        **How to use:**
        1. Upload a chest X-ray image
        2. View AI prediction
        3. Examine Grad-CAM visualization
        """
    )
    
    st.sidebar.title("⚙️ Settings")
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Model not found! Please ensure 'pneumonia_multiclass_densenet_model.keras' is in the same directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image for pneumonia detection"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with col1:
            st.markdown("### 📷 Original X-Ray")
            st.image(image_rgb, use_container_width=True)
        
        # Preprocess and predict
        img_array, img_resized = preprocess_image(image)
        
        with st.spinner("🔍 Analyzing X-Ray..."):
            predictions = model.predict(img_array, verbose=0)
            pred_class = np.argmax(predictions[0])
            pred_prob = predictions[0][pred_class]
            
            class_names = ['NORMAL', 'PNEUMONIA']
            predicted_label = class_names[pred_class]
            
            # Generate Grad-CAM
            if show_gradcam:
                heatmap = make_gradcam_heatmap(img_array, model)
                overlay, heatmap_colored = create_gradcam_overlay(img_resized, heatmap)
        
        # Display results
        with col2:
            if show_gradcam:
                st.markdown("### 🔥 Grad-CAM Heatmap")
                st.image(heatmap_colored, channels="BGR", use_container_width=True)
        
        with col3:
            if show_gradcam:
                st.markdown("### 🎯 Overlay")
                st.image(overlay, channels="BGR", use_container_width=True)
        
        # Prediction results
        st.markdown("---")
        st.markdown("## 🔬 Diagnosis Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prediction", predicted_label)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{pred_prob*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if pred_prob >= confidence_threshold:
                st.metric("Status", "High Confidence" if predicted_label == "PNEUMONIA" else "Normal")
            else:
                st.metric("Status", "Low Confidence - Review Needed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed probabilities
        st.markdown("### 📊 Class Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("NORMAL", f"{predictions[0][0]*100:.2f}%")
        with prob_col2:
            st.metric("PNEUMONIA", f"{predictions[0][1]*100:.2f}%")
        
        # Warning message
        if predicted_label == "PNEUMONIA" and pred_prob >= confidence_threshold:
            st.error("⚠️ **Pneumonia Detected** - Please consult a healthcare professional for proper diagnosis and treatment.")
        elif predicted_label == "NORMAL" and pred_prob >= confidence_threshold:
            st.success("✅ **Normal X-Ray** - No signs of pneumonia detected. However, always consult with a healthcare professional.")
        else:
            st.warning("⚠️ **Low Confidence Prediction** - Results are inconclusive. Please consult a healthcare professional.")
        
        # Disclaimer
        st.markdown("---")
        st.caption("""
        **Medical Disclaimer:** This tool is for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of qualified health providers with any questions regarding medical conditions.
        """)

if __name__ == "__main__":
    main()