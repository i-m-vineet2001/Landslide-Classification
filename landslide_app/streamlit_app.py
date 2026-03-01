import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from attention_unet import AttentionUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title("🌍 Landslide Segmentation using Attention U-Net")
st.write("Upload an image and get the predicted landslide mask.")


@st.cache_resource
def load_model():
    model = AttentionUNet().to(DEVICE)
    model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
    model.eval()
    return model


model = load_model()

uploaded_file = st.file_uploader("Upload Landslide Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img_np, caption="Original Image", width=800)

    # Resize
    img_resized = cv2.resize(img_np, (256, 256))

    # ---------------------------
    # 🔥 FIX: Correct Normalization
    # ---------------------------
    img_norm = img_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std

    # To tensor
    img_tensor = torch.tensor(img_norm).permute(2,0,1).float().unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0].cpu().numpy()[0]

    pred_mask = (pred > 0.5).astype("uint8")

    # Resize back
    pred_big = cv2.resize(pred_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = img_np.copy()
    overlay[:,:,1] = np.clip(overlay[:,:,1] + pred_big*180, 0, 255)

    col1, col2 = st.columns(2)
    col1.image(pred_big * 255, caption="Predicted Mask", width=400)
    col2.image(overlay, caption="Overlay Output", width=400)

    st.success("Prediction complete ✔")
