import streamlit as st
import torch
import torchvision.models as models
#from efficientnet_pytorch import EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import io
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image
import base64
import gdown
import torch
from torchvision import models
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === Set Streamlit Page Config ===
st.set_page_config(page_title="TB Detection from Chest X-rays", layout="wide")
st.title("🩺 TB Detection from Chest X-rays using Deep Learning")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom theme styles
st.markdown("""
<style>
/* Entire background + font */
.stApp {
        background-image:;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(255, 255, 255, 0.85);  /* Light white overlay */
        z-index: -1;
    }
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
    color: #1e3d59;
}

/* Fix dropdown text visibility and make it colorful */
div[data-baseweb="select"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 500;
    color: #1e1e1e;
}

div[data-baseweb="select"] > div {
    width: 320px !important;
    background-color: #e6f0ff !important;
    border: 1px solid #80bfff !important;
    border-radius: 10px !important;
    color: #1e1e1e !important;
}

div[data-baseweb="select"] input {
    color: #1e1e1e !important;
    font-weight: 600;
    background-color: transparent;
}

div[data-baseweb="select"] [role="option"] {
    color: #000 !important;
    background-color: #fff !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: #edf6ff;
    padding: 15px;
    border: 2px dashed #b3d7ff;
    border-radius: 12px;
}

/* Footer */
footer {
    font-size: 13px;
    color: #999;
}
</style>
""", unsafe_allow_html=True)

# Model parameters
NUM_CLASSES = 2
IMG_SIZE = 224
CLASS_NAMES = ['Normal Chest X-rays', 'TB Chest X-rays']  # Replace with your classes

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# Model definitions
def load_vgg16_model(model_path):
    model = models.vgg16(pretrained=False)
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load the entire model
    # try:
    #     model = torch.load(model_path, map_location=device)
    #     print("VGG16 entire model loaded successfully from:", model_path)
    # except RuntimeError as e:
    #     print(f"Error loading model: {e}")
    #     raise

    return model.to(device).eval()

def load_resnet50_model(model_path):
    model = models.resnet50(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

import json
import os
import zipfile
import sys
from io import StringIO
def load_efficientnet_b1_model(model_path):
    
    # Check if the path is a zip file
    if model_path.endswith('.zip'):
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall()  # Extracts to current directory
        model_path = os.path.splitext(model_path)[0]  # Use the directory name (e.g., 'my_model1 (1).keras')

    # Load model architecture from config.json
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Rebuild the model architecture
    model = tf.keras.models.model_from_json(json.dumps(config))

    # Load weights from model.weights.h5
    weights_path = os.path.join(model_path, 'model.weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
    
    model.load_weights(weights_path)
    #st.write(model.summary())
    #summary_str = str(model.summary())
    #st.write("Model Summary:")
    #old_stdout = sys.stdout
    #sys.stdout = mystdout = StringIO()
    #model.summary()
    #sys.stdout = old_stdout
    #summary_str = mystdout.getvalue()
    #st.write("Model Summary:")
    #st.text(summary_str)
    return model
    #model = tf.keras.models.load_model(model_path)
    

def load_densenet121_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Model loading function with dropdown selection
@st.cache_resource
def load_selected_model(model_name, model_paths):
    model_funcs = {
        'VGG16': load_vgg16_model,
        'ResNet50': load_resnet50_model,
        'EfficientNet-B1': load_efficientnet_b1_model,
        'DenseNet121': load_densenet121_model
    }
    if not os.path.exists(model_paths[model_name]):
        raise FileNotFoundError(f"Model file not found at: {model_paths[model_name]}")
    return model_funcs[model_name](model_paths[model_name])

# Predict function
def predict_image(model, image,model_name):
 if model_name == 'DenseNet121':  # Keras
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.image.resize(img_array, [IMG_SIZE, IMG_SIZE])
        img_array = np.expand_dims(img_array / 255.0, axis=0)
        predictions = model.predict(img_array)
        #st.write(f"Predictions shape: {predictions.shape}, Values: {predictions}")
        tb_prob = float(predictions[0][0])
        #st.write(f"TB prob: {tb_prob}")
        normal_prob = 1 - tb_prob 
        #st.write(f"Non-TB prob: {normal_prob}")
        probs = [normal_prob, tb_prob]

        # Decide predicted class
        pred_index = 1 if tb_prob > 0.5 else 0
        pred_class = "TB Chest X-rays" if pred_index == 1 else "Normal Chest X-rays"
        
        heatmap = make_gradcam_keras(img_array, model, layer_name="conv5_block16_2_conv")
        overlay_img = overlay_heatmap(heatmap, image)
        
        return pred_class, probs,overlay_img
        
 #else:  # PyTorch        
  #  image_tensor = transform(image).unsqueeze(0).to(device)
   # with torch.no_grad():
   #     output = model(image_tensor)
   #     probs = nn.Softmax(dim=1)(output)
    #    _, pred = torch.max(output, 1)
   # return CLASS_NAMES[pred.item()], probs.cpu().numpy()[0]
   
 elif model_name == "VGG16":
    # ✅ VGG16 with Grad-CAM
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs_tensor = nn.Softmax(dim=1)(output)
        pred_index = torch.argmax(probs_tensor).item()
        probs = probs_tensor.cpu().numpy()[0]
        pred_class = CLASS_NAMES[pred_index]

    # ✅ Get last Conv2d layer
    def get_last_conv_layer(model):
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("❌ No Conv2d layer found in the model!")
        print(f"✅ Using last Conv2d layer: {last_conv}")
        return last_conv

    target_layer = get_last_conv_layer(model)

    # ✅ Grad-CAM and overlay
    cam = make_gradcam_pytorch(model, image_tensor, target_layer,model_type="vgg",image_size=(224, 224))
    overlay_img = overlay_heatmap(cam, image)

    return pred_class, probs, overlay_img
    
 elif model_name == 'ResNet50':
    # ✅ ResNet50 with LIME
    #pred_class = None
    #probs = None
    overlay_img = image
    image_tensor = transform(image).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probs_tensor = nn.Softmax(dim=1)(output)
            pred_index = torch.argmax(probs_tensor).item()
           # st.write("Testong explain_with_lime...")
            probs = probs_tensor.cpu().numpy()[0]
            #st.write("Calling explain_with_lime...")
            pred_class = CLASS_NAMES[pred_index]
            lime_image = explain_with_lime(model, image, transform, CLASS_NAMES)  # Returns overlaid image
            # Use the wrapper to display LIME representation
            display_lime_representation(lime_image, image, pred_class, probs, CLASS_NAMES)
            overlay_img = lime_image
            #st.write("explain_with_lime completed")
    except Exception as e:
        st.error(f"ResNet50 prediction or LIME failed: {str(e)}")
        #overlay_img = image  # fallback to original image
    return pred_class, probs, overlay_img
    
 elif model_name == 'EfficientNet-B1':
        pred_class = None
        probs = None
        overlay_img = image
        try:
            # Resize image to 240x240 to match EfficientNet-B1 expected input
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.image.resize(img_array, [240, 240])  # Change to 240x240
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)  # Normalize and add batch dimension
            predictions = model.predict(img_array)
            #st.write(f"Predictions shape: {predictions.shape}, Values: {predictions}")
            pred_index = int(np.argmax(predictions[0]))
            pred_class = "TB Chest X-rays" if pred_index == 1 else "Normal Chest X-rays"
            probs = predictions[0].tolist()
            #tb_prob = predictions[0][1]  # Assuming index 1 is TB
            #st.write(f"TB prob: {probs}")
            #normal_prob = predictions[0][0]
            #st.write(f"Non-TB prob: {normal_prob}")
            #probs = [normal_prob, tb_prob]
           # pred_index = 1 if tb_prob > 0.5 else 0
            #pred_class = "TB Chest X-rays" if pred_index == 1 else "Normal Chest X-rays"
            #pred_class = CLASS_NAMES[pred_index]
            
            #if predictions.shape[1] == 1:  # Sigmoid output
             #   tb_prob = float(predictions[0][0])
                #normal_prob = 1 - tb_prob
            #elif predictions.shape[1] == 2:  # Softmax output
            #    normal_prob = float(predictions[0][0])
            #tb_prob = float(predictions[0][1])
            #non_tb_prob = float(predictions[0][0])
            #probs=[non_tb_prob,tb_prob]
            #pred_index = 1 if tb_prob > 0.5 else 0
            #pred_class = "Normal Chest X-rays" if pred_index == 0 else "TB Chest X-rays" 
            #print(f"Predicted Class: {predicted_class} (TB: {tb_prob:.4f}, Non-TB: {non_tb_prob:.4f})")
            #else:
            #raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
            #probs = [normal_prob, tb_prob]
            #st.write(f"Normal prob: {normal_prob}, TB prob: {tb_prob}")
            #pred_index = 1 if tb_prob > 0.5 else 0
           # pred_class = CLASS_NAMES[pred_index]
            
        # Grad-CAM for EfficientNet
            layer_name = 'top_conv'  # Replace with the actual last Conv2D layer name from model.summary()
            heatmap = make_gradcam_keras(img_array, model, layer_name)
            overlay_img = overlay_heatmap(heatmap, image)
            #display_lime_representation(overlay_img, image, pred_class, probs, CLASS_NAMES, model_name)
            
        except Exception as e:
            st.error(f"EfficientNet-B1 prediction or Grad-CAM failed: {str(e)}")
        return pred_class, probs, overlay_img

# Image statistics
def get_image_stats(image):
    img_array = np.array(image)
    stats = {
        'Dimensions': f"{image.size[0]}x{image.size[1]}",
        'Mean Pixel Values (RGB)': np.mean(img_array, axis=(0, 1)).round(2).tolist(),
        'Size (KB)': f"{os.path.getsize(io.BytesIO(image.tobytes()).tell()) / 1024:.2f}"
    }
    return stats

def make_gradcam_keras(img_array, model, layer_name):
    """
    Generate Grad-CAM heatmap for Keras-based model (e.g., DenseNet121).
    
    Args:
        img_array (np.array): Preprocessed input image array of shape (1, H, W, 3).
        model (keras.Model): Trained Keras model.
        layer_name (str): Name of the last convolutional layer (e.g., "conv5_block16_2_conv").
        
    Returns:
        heatmap (np.array): Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of class output w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling across width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to 0–1
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    return heatmap
    
def overlay_heatmap(heatmap, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap onto the original image.
    
    Args:
        heatmap (np.array): Grad-CAM heatmap.
        original_image (PIL.Image): Original image before preprocessing.
        alpha (float): Transparency factor for heatmap.
        colormap (int): OpenCV colormap to use.
        
    Returns:
        overlay (PIL.Image): Blended image.
    """
    # Convert heatmap to 0–255 range
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, original_image.size)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # Convert original PIL image to array
    original_array = np.array(original_image.convert("RGB"))

    # Blend heatmap with original image
    overlay = cv2.addWeighted(original_array, 1 - alpha, heatmap_color, alpha, 0)

    return Image.fromarray(overlay)

def display_lime_representation(lime_image, original_image, pred_class, probs, class_names):
    import streamlit as st
    from PIL import Image

    # Ensure inputs are valid
    if lime_image is None or original_image is None:
        st.error("Invalid LIME or original image provided.")
        return
 
def make_gradcam_pytorch(model, img_tensor, final_conv, model_type="vgg", image_size=(224, 224)):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        print("Forward hook triggered")
        features.append(output)
        print(f"Feature shape: {output.shape}")

    def backward_hook(module, grad_input, grad_output):
        print("Backward hook triggered")
        gradients.append(grad_output[0])
        print(f"Gradient shape: {grad_output[0].shape}")

    # Register hooks
    print(f"Registering hooks on layer: {final_conv}")
    hook_handle_fwd = final_conv.register_forward_hook(forward_hook)
    hook_handle_bwd = final_conv.register_backward_hook(backward_hook)

    # Ensure gradients are enabled for the layer
    final_conv.weight.requires_grad_(True)
    final_conv.bias.requires_grad_(True)

    model.eval()  # Set model to evaluation mode
    model.zero_grad()  # Clear any existing gradients

    # Forward pass
    print("Running forward pass")
    output = model(img_tensor)
    print(f"Model output shape: {output.shape}")

    # Compute loss
    if model_type in ["vgg"]:
        prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()
        loss = output[0][1]  # TB class score
        print(f"Loss value: {loss.item()}")
    else:
        prob = torch.sigmoid(output)[0][0].item()
        loss = output[0][0]
        print(f"Loss value: {loss.item()}")

    # Backward pass
    print("Running backward pass")
    loss.backward()

    # Check captured data
    if not features:
        hook_handle_fwd.remove()
        hook_handle_bwd.remove()
        raise RuntimeError("Forward hook failed: No features captured")
    if not gradients:
        hook_handle_fwd.remove()
        hook_handle_bwd.remove()
        raise RuntimeError("Backward hook failed: No gradients captured")

    # Process Grad-CAM
    grad = gradients[0].detach().cpu().numpy()
    fmap = features[0][0].detach().cpu().numpy()
    print(f"Gradient shape after detach: {grad.shape}")
    print(f"Feature map shape: {fmap.shape}")

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, image_size)
    cam -= np.min(cam)
    cam /= np.max(cam + 1e-8)  # Avoid division by zero

    hook_handle_fwd.remove()
    hook_handle_bwd.remove()

    return cam

    

def explain_with_lime(model, image_pil, transform, class_names):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from PIL import Image
    import streamlit as st

    try:
        # Ensure model is in eval mode and on correct device
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #st.write(f"Model device: {device}")

        def batch_predict(images):
            try:
                batch = []
                for img in images:
                    # Convert image to uint8 if needed
                    img = img.astype(np.uint8) if img.dtype != np.uint8 else img
                    pil_img = Image.fromarray(img).convert('RGB')
                    tensor = transform(pil_img).unsqueeze(0)  # [1, 3, 224, 224]
                    batch.append(tensor)
                
                batch = torch.cat(batch).to(device)
                #st.write(f"Batch shape: {batch.shape}, device: {batch.device}")
                
                with torch.no_grad():  # Explicitly disable gradients
                    logits = model(batch)
                    probs = F.softmax(logits, dim=1)
                    #st.write(f"Logits shape: {logits.shape}, Probs shape: {probs.shape}")
                    probs_np = probs.cpu().numpy()  # Convert to NumPy
                    #st.write(f"Probs NumPy shape: {probs_np.shape}")
                
                return probs_np
            
            except Exception as e:
                st.error(f"batch_predict failed: {str(e)}")
                raise RuntimeError(f"batch_predict error: {str(e)}")

        # Convert PIL image to NumPy
        image_np = np.array(image_pil.convert('RGB'))
        #st.write(f"Input image shape: {image_np.shape}, dtype: {image_np.dtype}")

        # Run LIME explanation
        #st.write("Starting LIME explanation...")
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_np,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=100  # Reduced for faster debugging
        )
        #st.write("LIME explanation completed")

        # Generate overlaid image
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        temp = np.uint8(temp)
        overlaid_image = mark_boundaries(temp, mask,color=(0, 1, 0))
        overlaid_image = np.uint8(overlaid_image * 255)
        
        #st.write("LIME overlay generated")
        return Image.fromarray(overlaid_image)
    
    except Exception as e:
        st.error(f"LIME explanation failed at: {str(e)}")
        raise RuntimeError(f"LIME explanation failed: {str(e)}")


# Streamlit app
def main():
    # Model paths (update these to your local directory)
    model_paths = {
        'VGG16': 'best_vgg16_tuberculosis.pth',
        'ResNet50': 'best_resnet50_tuberculosis.pth',
        'EfficientNet-B1': 'my_model1 (1).keras.zip',
        'DenseNet121': 'tb_model.keras'
    }

    # Sidebar
    with st.sidebar:
        
        # File uploader
        uploaded_file = st.file_uploader("Upload an X-Ray Image...", type=['jpg', 'jpeg', 'png'])
        # Model selection dropdown
        model_name = st.selectbox("Choose a Model", list(model_paths.keys()))

    # Main content
    st.header("TB Classification")
    # st.subheader("Select Model and Upload Image")

    if uploaded_file is not None:
    # Load and display image
    
        image = Image.open(uploaded_file).convert('RGB')
             
        # Load selected model
        try:
            model = load_selected_model(model_name, model_paths)
            st.success(f"{model_name} model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
        
        # Predict
        pred_class, probs,overlay_img = predict_image(model, image,model_name)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🩻 Original X-ray")
            st.image(image, caption="Original Image", width=300)

        with col2:
            if model_name== 'ResNet50':
                st.markdown("### 🌟 LIME Explanation")
                st.image(overlay_img, caption="LIME Representation", width=300)
                st.markdown("""
                <div style='font-size: 14px; margin-top: -10px;'>
                <b>LIME Legend:</b><br>
                <span style='color:green;'>Green Boundaries</span> = Superpixels influencing the prediction<br>
                <span style='color:gray;'>Gray Areas</span> = Less relevant regions
                </div>
                """, unsafe_allow_html=True)
            else :
                st.markdown("### 🔥 Grad-CAM Heatmap")
                st.image(overlay_img, caption="Grad-CAM", width=300)
                st.markdown("""
                <div style='font-size: 14px; margin-top: -10px;'>
                <b>Color Legend:</b><br>
                <span style='color:red;'>Red/Yellow</span> = High importance (model focused here) <br>
                <span style='color:green;'>Green</span> = Moderate influence <br>
                <span style='color:blue;'>Blue</span> = Low importance (background / less relevant)
                </div>
                """, unsafe_allow_html=True)

        # Display prediction
        st.write("### Prediction Result")
        st.success(f"Predicted Class: **{pred_class}** (using {model_name})")
        st.write("Class Probabilities:")
        st.subheader("Confidence Bar")
        for cls, prob in zip(CLASS_NAMES, probs):
            st.write(f"{cls}: {prob:.4f}")
        st.progress(int(prob * 100))
        # Display image statistics
        st.write("### Image Statistics")
        stats = get_image_stats(image)
        for key, value in stats.items():
            st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()
    
 # Footer
st.markdown("""
<div class="footer">
    🚀 Capstone Project | <b>BITS Pilani WILP</b> Group 17 | <span style="color:#ffd700;">2025</span> © <b>TB Detection Project</b>
</div>
""", unsafe_allow_html=True)
