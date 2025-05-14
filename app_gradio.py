import gradio as gr
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T_vision # Use your alias from training
from PIL import Image
import numpy as np
import os
import io

# --- 1. IMPORT YOUR HybridModel DEFINITION ---
# Ensure model_definition.py contains the HybridModel WITHOUT magnification embedding
# (assuming your best_phased_model_checkpoint.pth was trained without it)
from model_definition import HybridModel

# --- 2. DEFINE CONSTANTS AND LOAD ARTIFACTS (Simplified from predictor.py) ---
DEVICE = torch.device("cpu")
MODEL_PATH = "best_phased_model_checkpoint.pth"

# --- Global variables for loaded artifacts ---
model_instance = None
LABEL_ENCODER_CLASSES = None

# --- Hardcoded Mappings (Ensure these are correct for your loaded model!) ---
# This order MUST MATCH how your LabelEncoder was fit during training of the loaded checkpoint
LABEL_ENCODER_CLASSES_HARDCODED = ['benign', 'malignant']

# Magnification is NOT used by the model version we assume you're loading (based on previous errors)
# If you were using a model WITH magnification, you'd need MAG_TO_IDX here
# MAG_TO_IDX_HARDCODED = {'40X': 0, '100X': 1, '200X': 2, '400X': 3}


def load_model_gradio():
    global model_instance, LABEL_ENCODER_CLASSES

    print("Attempting to load model for Gradio...")
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model checkpoint not found at {MODEL_PATH}")
        return False

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        print(f"Checkpoint loaded from {MODEL_PATH}.")

        if 'label_encoder_classes' in checkpoint:
            chkpt_le_classes = checkpoint['label_encoder_classes']
            if isinstance(chkpt_le_classes, np.ndarray):
                LABEL_ENCODER_CLASSES = chkpt_le_classes.tolist()
            else:
                LABEL_ENCODER_CLASSES = chkpt_le_classes
            print(f"Label encoder classes from checkpoint: {LABEL_ENCODER_CLASSES}")
        else:
            print("Warning: 'label_encoder_classes' not found in checkpoint. Using hardcoded fallback.")
            LABEL_ENCODER_CLASSES = LABEL_ENCODER_CLASSES_HARDCODED
        
        if not LABEL_ENCODER_CLASSES or len(LABEL_ENCODER_CLASSES) == 0:
            print("FATAL: Label classes could not be determined.")
            return False

        NUM_CLASSES = len(LABEL_ENCODER_CLASSES)

        effnet_base = models.efficientnet_v2_s(weights=None)
        vit_base = models.vit_b_16(weights=None)

        # Instantiate the HybridModel WITHOUT magnification arguments
        model_instance = HybridModel(
            backbone1=effnet_base,
            backbone2=vit_base,
            num_classes=NUM_CLASSES
        )
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.to(DEVICE)
        model_instance.eval()
        print("Model loaded successfully for Gradio.")
        return True

    except Exception as e:
        print(f"FATAL: Error loading model for Gradio: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 3. DEFINE IMAGE PREPROCESSING (Same as predictor.py) ---
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

test_image_transform = T_vision.Compose([
    T_vision.Resize((224, 224)),
    T_vision.ToTensor(),
    T_vision.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# --- 4. GRADIO PREDICTION FUNCTION ---
# This function will be wrapped by Gradio.
# It takes inputs compatible with Gradio input components and returns outputs for Gradio output components.
def predict_breast_cancer(image_pil: Image.Image): # Gradio's Image input gives a PIL Image
    if model_instance is None:
        return {"error": "Model not loaded."} # Gradio can display dictionaries as JSON output

    try:
        # Image is already a PIL Image from Gradio input, ensure RGB
        image_pil_rgb = image_pil.convert("RGB")
        image_tensor = test_image_transform(image_pil_rgb).unsqueeze(0).to(DEVICE)

        # No magnification needed for this model version
        with torch.no_grad():
            logits = model_instance(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            
            if not (0 <= predicted_idx < len(LABEL_ENCODER_CLASSES)):
                 return {"error": f"Model prediction index {predicted_idx} out of bounds for labels."}

            predicted_label = LABEL_ENCODER_CLASSES[predicted_idx]
            # Gradio's Label output component expects a dictionary of {label: confidence}
            confidences = {LABEL_ENCODER_CLASSES[i]: float(probabilities[0, i]) for i in range(len(LABEL_ENCODER_CLASSES))}
            
            return confidences # Return confidences for Gradio Label output
            # Or, if you prefer simpler output for a text box:
            # return f"Predicted: {predicted_label} (Confidence: {confidences[predicted_label]:.4f})"

    except Exception as e:
        print(f"Error during Gradio prediction: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}


# --- 5. CREATE AND LAUNCH GRADIO INTERFACE ---
if __name__ == '__main__':
    if load_model_gradio(): # Load model on startup
        # Define Gradio input and output components
        image_input = gr.Image(type="pil", label="Upload Histopathology Image")
        # Output as a Label component to show class confidences nicely
        label_output = gr.Label(num_top_classes=len(LABEL_ENCODER_CLASSES_HARDCODED), label="Prediction")
        # Or as a simple JSON/Text output:
        # json_output = gr.JSON(label="Prediction Details")

        # Create the interface
        iface = gr.Interface(
            fn=predict_breast_cancer,
            inputs=image_input,
            outputs=label_output, # or json_output
            title="BreaKHis - Breast Cancer Classifier",
            description="Upload a breast histopathology image (PNG, JPG) to classify it as benign or malignant. "
                        "This demo uses a Hybrid Deep Learning Model (EfficientNetV2-S + ViT-B/16).",
            allow_flagging="never" # Disable flagging for simplicity
        )

        # Launch the interface
        # share=True creates a temporary public link if you want to share it easily (lasts 72 hours)
        # Set share=False for local-only access.
        print("Launching Gradio interface... Access it at http://127.0.0.1:7860 (or the public link if share=True)")
        iface.launch(share=False) # Set share=True for a temporary public link
    else:
        print("Could not launch Gradio interface because model loading failed.")